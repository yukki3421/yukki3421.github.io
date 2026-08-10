---
  title: "LingBot-VLA 推理部署"
  date: 2026-08-10T14:00:00+08:00
  draft: false
  categories: ["LingBot VLA"]
  tags: ["VLA", "机器人", "部署"]
  summary: "LingBot-VLA 推理部署的主要流程与原理。"
  ShowToc: true
  TocOpen: false
---


本文只记录 VLA 算法工程师需要掌握的推理部署主流程，不展开 WebSocket、SSH、异步编程等底层实现。

## 1. 整体架构

```text
工控机
├── 采集三路相机
├── 读取双臂 state
├── 提供任务 instruction
└── 通过 WebSocket 发送 observation
             │
             │ msgpack + SSH 隧道
             ▼
GPU 服务器
├── 输入预处理
├── LingBot-VLA 推理
├── 动作反归一化
└── 返回 action chunk
             │
             ▼
工控机
├── 动作限幅与安全检查
└── 通过 CAN/机器人 SDK 执行动作
```

核心分工：

- `real_deploy`：相机采集、机器人状态读取、动作执行。
- `deploy/websocket_client_policy.py`：工控机网络客户端。
- `deploy/websocket_policy_server.py`：GPU 网络服务。
- `deploy/lingbot_vla_policy.py`：模型加载、预处理和推理。

## 2. 推理服务启动流程

入口：

```text
deploy/lingbot_vla_policy.py::main()
```

调用链：

```text
解析命令行参数
  ↓
LingbotVLAServer(...)
  ↓
load_vla() 加载配置和模型
  ↓
WebsocketPolicyServer(...)
  ↓
serve_forever() 监听端口
```

常用参数：

```text
--model_path           后训练 HF checkpoint
--norm_path            训练数据归一化统计
--use_length           每次返回 action chunk 的前多少步
--num_denoising_step   Flow Matching 去噪步数
--port                 WebSocket 服务端口
--use_compile          是否使用 torch.compile
```

## 3. 模型加载必须理解的内容

checkpoint 目录至少需要：

```text
config.json
lingbotvla_cli.yaml
*.safetensors
```

配置来源：

- `config.json`：Hugging Face/LeRobot 标准模型配置。
- `lingbotvla_cli.yaml`：训练时的 model、train、data 完整记录。
- `QWEN25_PATH`：Qwen2.5-VL 基座配置、tokenizer 和图像处理器。

加载顺序：

```text
读取 config.json
  ↓
使用 lingbotvla_cli.yaml 的 model/train 字段覆盖
  ↓
合并 Qwen 文本与视觉配置
  ↓
创建 LingbotVlaPolicy
  ↓
加载 safetensors
  ↓
移动到 CUDA 并切换 eval 模式
```

部署默认使用 BF16 推理，模型输出在回到 CPU 后转换为 FP32。

## 4. reset 的作用

客户端首次推理前需要发送：

```python
policy.reset("dual_piper")
```

服务端据此加载：

```text
configs/robot_configs/dual_piper.yaml
```

一句话概括：

> `reset()` 根据机器人类型加载输入输出映射和归一化规则，为后续推理初始化 `FeatureTransform`。

## 5. dual_piper 输入格式

工控机需要提供：

```python
{
    "observation.images.top": top_image,
    "observation.images.wrist_left": left_image,
    "observation.images.wrist_right": right_image,
    "observation.state": robot_state,
    "task": instruction,
}
```

图像要求：

```text
格式：HWC
通道：RGB
类型：uint8
范围：0～255
```

state 必须严格按以下顺序排列：

```text
[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
```

共 14 维。相机左右映射、state 顺序和 instruction 文本错误，通常比数值精度更容易造成真机效果下降。

## 6. 输入预处理流程

核心入口：

```text
FeatureTransform.apply(observation)
```

处理过程：

```text
三路图像
  → resize 到 224×224
  → Qwen image processor
  → 视觉 patch

14维 state
  → 拆分 arm.position / effector.position
  → 使用训练统计量归一化
  → padding 到 max_state_dim=75

task 字符串
  → 添加 <bos> 和换行符
  → tokenizer
  → lang_tokens / lang_masks
```

推理代码创建的全零“假动作”只用于兼容训练/推理共用的 `FeatureTransform`，不是 Flow Matching 的起始噪声。

## 7. 动作生成流程

模型输入主要包括：

```text
images
img_masks
lang_tokens
lang_masks
state
```

模型首先编码图像和语言，并保存 KV Cache。随后生成高斯随机动作噪声：

```text
shape = (batch=1, chunk=50, max_action_dim=75)
```

Flow Matching 使用默认 10 步 Euler 更新：

```text
随机噪声
  → 根据图像、语言、state 预测速度
  → 多步更新
  → 得到归一化动作 chunk
```

模型完整生成 50 步，`use_length` 只决定最后返回前多少步。例如：

```text
模型输出       (1, 50, 75)
去除 batch     (50, 75)
反向特征转换   (50, 14)
use_length=25  (25, 14)
```

## 8. 动作后处理

核心入口：

```text
FeatureTransform.unapply()
```

执行：

```text
删除 padding
  ↓
反归一化
  ↓
关节增量加回当前 state
  ↓
恢复 [L6, L夹爪, R6, R夹爪]
```

最终返回的是机器人空间中的绝对目标动作。工控机仍必须进行关节限幅、速度限制和安全检查。

## 9. 网络收发需要掌握到什么程度

只需理解：

```text
Python字典/NumPy数组
  → msgpack 打包成字节
  → WebSocket 发送
  → 服务端解包并推理
  → 动作字典打包返回
```

服务端网络收发使用异步 WebSocket，但 GPU 推理是同步的，当前设计更适合单客户端。

`api_key` 客户端参数目前没有对应的服务端校验，不能视为有效鉴权。

## 10. SSH 隧道与健康检查

生产模式通常是：

```text
工控机 localhost:6006
  → SSH 隧道
  → GPU localhost:6006
```

VS Code 当前调试配置使用云端 8006，可建立：

```bash
ssh -i ~/.ssh/id_ed25519_real_deploy \
  -p 39134 \
  -N \
  -L 6006:127.0.0.1:8006 \
  root@connect.westd.seetacloud.com
```

必须记住两级检查：

```text
GPU 本机 8006 返回 OK → 推理服务正常
工控机 6006 返回 OK   → SSH 隧道也正常
```

对应命令：

```bash
# GPU 服务器
curl http://127.0.0.1:8006/healthz

# 工控机
curl http://127.0.0.1:6006/healthz
```

健康检查只能证明服务与网络正常，不等于真实 observation 一定正确。

## 11. 没有真机时如何调试

可分三层：

1. `01 Feature Mapping (fake obs)`：检查特征映射、归一化和 padding。
2. Open-loop evaluation：使用真实数据集观测检查预测动作。
3. Fake WebSocket client：发送假图像和假 state，验证完整网络推理。

重要断点：

```text
WebsocketPolicyServer._handler()   查看收到的 obs
LingbotVLAServer.infer()           查看预处理前后数据
PolicyPreprocessMixin.select_action()
FlowMatching.sample_actions()      查看去噪过程
FeatureTransform.unapply()         查看最终机器人动作
```

## 12. 常见问题排查顺序

```text
服务无法连接
  → GPU 本机 healthz
  → SSH 隧道
  → 工控机 healthz

模型无法启动
  → config.json
  → lingbotvla_cli.yaml
  → safetensors 是否完整
  → QWEN25_PATH
  → GPU 显存

真机效果差
  → 相机是否接反/卡帧
  → state/action 顺序
  → norm_stats 是否匹配
  → instruction 是否与训练一致
  → checkpoint 对比
  → use_length 与控制频率
  → 机器人标定和动作限幅
  → 最后再比较 BF16/FP32
```

## 13. 算法工程师必须能说明的内容

- VLA 的图像、语言、state 如何形成模型输入。
- action chunk、Flow Matching 去噪和 `use_length` 的关系。
- 训练归一化必须在部署时保持一致。
- 模型输出如何恢复为机器人真实动作。
- 工控机与 GPU 推理服务的职责边界。
- WebSocket、msgpack 和 SSH 隧道各自解决什么问题。
- 如何通过 health check、假输入和 open-loop eval 分层定位故障。
- 真机部署必须具备限幅、急停、日志、录像和 dry-run。
