---
  title: "LingBot-VLA 双臂实机远程部署手册"
  date: 2026-08-10T14:10:00+08:00
  draft: false
  categories: ["LingBot VLA"]
  tags: ["VLA", "双臂机器人", "实机部署"]
  summary: "AutoDL 训练机与工控机 Piper 双臂的远程部署手册。"
  ShowToc: true
  TocOpen: false
---


本文汇总 AutoDL 训练机 + 工控机 Piper 双臂的远程部署流程、命令与排障，便于查阅。

**架构：**

```text
工控机 real_deploy（无 GPU）
  采图 + 读 14 维关节 → WebSocket
       │
  127.0.0.1:6006  ←── SSH -L 隧道 ──→  AutoDL:6006
                                           │
                              lingbot_vla_policy（GPU 推理）
                                           │
       ←────────── action (T,14) ──────────┘
  展平下发 can_s2(左) / can_s1(右)
```

工控机**没有**单独的「客户端服务」：跑的是 `main_real_xvla.py` 主程序。  
云端跑的才是常驻推理服务。

---

## 0. 关键路径

| 角色 | 路径 |
| --- | --- |
| 云端 lingbot-vla | `/root/autodl-fs/cxy/projects/lingbot-vla` |
| 云端 real_deploy（源码同步用） | `/root/autodl-fs/cxy/projects/real_deploy` |
| 训练输出 | `.../lingbot-vla/output/dual_piper_8tasks_v30/` |
| 10K 权重 | `.../checkpoints/global_step_10000/hf_ckpt` |
| 10K 备份 | `.../checkpoints_backup/global_step_10000/` |
| Norm | `.../lingbot-vla/assets/norm_stats/dual_piper_8tasks.json` |
| 机器人映射 | `.../lingbot-vla/configs/robot_configs/dual_piper.yaml` |
| 工控机 real_deploy | `/home/tuoyuan/code/real_deploy` |
| 工控机 forge | `/home/tuoyuan/code/forge`（启动时需 `PYTHONPATH`） |
| 工控机部署配置 | `~/code/real_deploy/configs/STACK/LingBotVLA_REAL_PIPER_8tasks.yaml` |
| 训练数据 | `/root/autodl-fs/cxy/data/v2_6cam/merged_all_8tasks_v30` |

**SSH（经笔记本端口转发，端口/密钥以现网为准）：**

| 方向 | 示例 |
| --- | --- |
| 云端 → 工控机 | `ssh -i ~/.ssh/id_ed25519 -p 22122 tuoyuan@127.0.0.1` |
| 工控机 → 云端 | `ssh -i ~/.ssh/id_ed25519_real_deploy -p 39134 root@connect.westd.seetacloud.com` |

**注意：** 在云端 root 下写 `DST=~/code/...` 会展开成 `/root/...`。拷到工控机请用绝对路径 `/home/tuoyuan/code/real_deploy`。

---

## 1. 前置条件清单

- [ ] 已有可用 `hf_ckpt`（如 `global_step_10000`）
- [ ] 训练占满 4 卡时**不要**同机起推理；先停训或留 1 张卡
- [ ] 工控机 conda：`dual_arm`（含 websockets、msgpack；forge 用 `~/code/forge`）
- [ ] CAN：`can_s1`（右）、`can_s2`（左）正常
- [ ] 云 ↔ 工控机 SSH 互通
- [ ] AutoDL 自定义服务端口 **6006** 已放行

### 8 个训练任务指令（`instruction` 必须原句）

1. `hold the tray with both hands`
2. `pack the toothbrush holder`
3. `place objects into desk drawer`
4. `put in pen container`
5. `put the pencil case into the schoolbag`
6. `put the shoes back`
7. `tidy up the makeup table`
8. `tie a knot`

### 动作维度（14）

`[左臂6, 左爪1, 右臂6, 右爪1]` → `piper_left` / `piper_right`。

### 相机（现场曾确认，重启后可能漂移）

| 名称 | 设备 |
| --- | --- |
| top | `/dev/video4` |
| wrist_left | `/dev/video8` |
| wrist_right | `/dev/video6` |

---

## 2. 开环检测（上真机前，在云端）

腾出 GPU0，停掉占卡训练后：

```bash
conda activate lingbotvla
cd /root/autodl-fs/cxy/projects/lingbot-vla

export QWEN25_PATH=/root/autodl-fs/cxy/projects/lingbot-vla/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python scripts/open_loop_eval.py \
  --model_path /root/autodl-fs/cxy/projects/lingbot-vla/output/dual_piper_8tasks_v30/checkpoints/global_step_10000/hf_ckpt \
  --norm_path /root/autodl-fs/cxy/projects/lingbot-vla/assets/norm_stats/dual_piper_8tasks.json \
  --data_path /root/autodl-fs/cxy/data/v2_6cam/merged_all_8tasks_v30 \
  --robo_name dual_piper \
  --traj_ids 0 100 200 \
  --use_length 25 \
  --save_plot_path ./open_loop_test/step10000/
```

- 输出：终端 MSE/MAE；图在 `./open_loop_test/step10000/`
- MAE / MSE 不同是指标定义（MSE 对尖峰更敏感）；优先看 **MAE + 曲线**是否大致跟上 GT
- 约 3 条 episode × 每条最多 10 次推理；首次含加载约 10–20 分钟

---

## 3. 云端：启动推理服务

### 3.1 写启动脚本

```bash
cat > /tmp/start_lingbot_infer.sh << 'EOF'
#!/bin/bash
set -e
export PATH=/root/miniconda3/envs/lingbotvla/bin:/root/miniconda3/bin:$PATH
export QWEN25_PATH=/root/autodl-fs/cxy/projects/lingbot-vla/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct
cd /root/autodl-fs/cxy/projects/lingbot-vla
CUDA_VISIBLE_DEVICES=0 python -m deploy.lingbot_vla_policy \
  --model_path /root/autodl-fs/cxy/projects/lingbot-vla/output/dual_piper_8tasks_v30/checkpoints/global_step_10000/hf_ckpt \
  --norm_path /root/autodl-fs/cxy/projects/lingbot-vla/assets/norm_stats/dual_piper_8tasks.json \
  --port 6006 \
  --use_length 25
EOF
chmod +x /tmp/start_lingbot_infer.sh
```

**注意：** `--model_path` 与路径必须在同一逻辑行（行续用 `\`），不要把路径单独拆成下一行且漏 `\`。

换更新 ckpt 时只改 `--model_path` 中的 `global_step_XXXX`。

### 3.2 用 screen 后台跑

```bash
screen -dmS lingbot_infer /tmp/start_lingbot_infer.sh
screen -r lingbot_infer
```

看到：

```text
INFO:websockets.server:server listening on 0.0.0.0:6006
```

后 **Ctrl+A D** 挂起（不要 Ctrl+C）。

`screen -dmS 名字 命令`：新建名为 `lingbot_infer` 的 screen，后台执行脚本，SSH 断了服务仍在。

### 3.3 确认端口（本机可能没有 `ss`）

```bash
python3 -c "import socket;s=socket.socket();print('ok' if s.connect_ex(('127.0.0.1',6006))==0 else 'fail');s.close()"
nvidia-smi   # GPU0 应有显存占用
ps aux | grep lingbot_vla_policy | grep -v grep
```

应打印 `ok`。日志里偶发 `connection closed` 多为探测连接断开，服务仍在听即可忽略。

### 3.4 关闭推理

```bash
screen -r lingbot_infer
# Ctrl+C 停掉进程，再 exit 或 Ctrl+A D

# 或：
ps aux | grep lingbot_vla_policy | grep -v grep
kill <PID>
screen -S lingbot_infer -X quit
```

再测 6006 应为 `fail`，GPU 显存下降。

---

## 4. 工控机：SSH 隧道（必须常开）

```bash
ssh -i ~/.ssh/id_ed25519_real_deploy \
  -L 6006:127.0.0.1:6006 \
  -p 39134 \
  root@connect.westd.seetacloud.com
```

该窗口保持登录。另开终端测：

```bash
python3 -c "import socket;s=socket.socket();print('ok' if s.connect_ex(('127.0.0.1',6006))==0 else 'fail');s.close()"
```

YAML 中：`server_url: "ws://127.0.0.1:6006"`。

---

## 5. 工控机：配置与代码

### 5.1 从云端拉取（工控机执行）

云端改代码后，工控机拉取（端口/密钥按现网）：

```bash
HOST=root@connect.westd.seetacloud.com
PORT=39134
KEY=~/.ssh/id_ed25519_real_deploy
SRC=/root/autodl-fs/cxy/projects/real_deploy
DST=/home/tuoyuan/code/real_deploy

scp -P $PORT -i $KEY $SRC/main_real_xvla.py $DST/main_real_xvla.py
scp -P $PORT -i $KEY $SRC/robot/robot_base.py $DST/robot/robot_base.py
scp -P $PORT -i $KEY $SRC/task/task_base.py $DST/task/task_base.py
scp -P $PORT -i $KEY $SRC/policy/__init__.py $DST/policy/__init__.py
scp -P $PORT -i $KEY -r $SRC/policy/lingbotvla_remote $DST/policy/
scp -P $PORT -i $KEY $SRC/configs/STACK/LingBotVLA_REAL_PIPER_8tasks.yaml $DST/configs/STACK/
```

拉取 YAML 后务必再改回你的 `instruction`（云端模板可能是 TODO）。

### 5.2 必改 YAML 项

文件：`~/code/real_deploy/configs/STACK/LingBotVLA_REAL_PIPER_8tasks.yaml`

- `policy.instruction`：当前任务英文原句  
- `env.cameras[*].port`：现场确认的 `/dev/video*`  
- 建议首测：`require_batch_confirmation: true`  
- `execute_action_steps: 25`  
- 旧 Forge 需要顶层 `robot:` + 双臂 `robots:`（已在模板中）

### 5.3 策略注册（已合入源码时一般不用再改）

- `policy/lingbotvla_remote/`：远程客户端  
- `policy/__init__.py`：lazy 导出 `LINGBOTVLA_REMOTE_IN_FORGE`  
- `task/task_base.py`：`type: lingbotvla_remote` 分支；策略按需 import（避免工控机急加载本地大模型）

服务端返回的是合并后的 **`action` (T,14)**，客户端已按此解析（不是 `action.arm.position` 拆分键）。

---

## 6. 现场启动客户端

### 6.1 启动前

1. 云端推理已起，6006 = `ok`  
2. 工控机隧道开着，本地 6006 = `ok`  
3. 场景已按 `instruction` 摆好  
4. 急停就绪；CAN 正常；本机有显示器（看相机拼接）

### 6.2 启动命令

```bash
conda activate dual_arm
export PYTHONPATH=/home/tuoyuan/code/forge:$PYTHONPATH
cd ~/code/real_deploy
python main_real_xvla.py --config configs/STACK/LingBotVLA_REAL_PIPER_8tasks.yaml
```

**每次新开终端都要带 `PYTHONPATH`**，否则会落到 site-packages 旧 forge，易再报 `robot Field required` 等错误。

### 6.3 正常现象

1. `Creating robot ... piper_left ... can_s2` / `piper_right ... can_s1`  
2. 三路相机拼接画面  
3. `[LingBotRemote] connected ws://127.0.0.1:6006, reset robo_name=dual_piper`  
4. 首次推理可能要等几十秒；云端 screen 可见 `sample action time`  
5. `[Action] 开始执行: {...}`（含 `piper_left` / `piper_right`）  
6. `[PolicyGate] ... [y/n]`：输入 **`y`** 继续下一批；**`n`** 再 **`q`** 可退出

### 6.4 一次启动能跑多久？

- 进程会**持续循环**控臂，不是自动「只跑一个任务就退出」  
- 同一进程里 `instruction` 固定；换任务请改 YAML 后重启  
- 结束：终端 **Ctrl+C**，或 PolicyGate 里 `q`

### 6.5 任务多久算失败？

代码**没有**任务级超时（`timeout_s` 只是 WebSocket）。需人工约定，例如：

- 参考训练时长：多数任务中位约 40–60s，P90 约 50–95s  
- 建议：超过该任务 P90×1.5，或统一 **2 分钟**仍未完成 → 失败  
- 碰撞/乱甩/长时间无进展 → 立即失败并急停  

---

## 7. 推荐完整顺序（检查清单）

1. [ ]（可选）开环评测 10K/更新 ckpt  
2. [ ] 停训或腾出 1 张 GPU  
3. [ ] 云端 `screen` 起 `lingbot_infer`，确认 `listening on 0.0.0.0:6006`  
4. [ ] 工控机 SSH `-L 6006:...`，测 `ok`  
5. [ ] YAML：`instruction` + 相机 port  
6. [ ] 摆场景、急停、CAN  
7. [ ] `PYTHONPATH=.../forge` + `main_real_xvla.py`  
8. [ ] 三路画面 + `[Action]` + PolicyGate 确认  
9. [ ] 测完：客户端 Ctrl+C；需要时再关云端推理  

---

## 8. 常见问题

### 8.1 `ForgeConfig` / `robot Field required`

工控机 forge 要顶层 `robot`。处理：YAML 同时写 `robot` + `robots`；`main_real_xvla.py` 用兼容加载；并设置 `PYTHONPATH=/home/tuoyuan/code/forge`。

### 8.2 `dict has no attribute type`

`config.robots` 可能是 dict 列表。已用 `_iter_robot_configs` 兼容。

### 8.3 `PickAndPlaceTaskRobot` 抽象方法 / 无 `robots` 属性

工控机 forge 的 `BaseTaskRobot` 与新版不一致。已在 `robot/robot_base.py` 实现 `step` / `get_observation`，并规范化 `self.robots`。

### 8.4 `RealRuntime ... missing argument: robot`

旧 Runtime 需要 `robot=`。`main_real_xvla.py` 已按签名兼容传入。

### 8.5 缺相机 / `select() timeout`（如 video6）

- 三路不齐时，任务会**空等**，不会推理、臂不动  
- 设备号可能漂移；或单独能看、三路同开因 USB 带宽失败  
- 排查：`v4l2-ctl --list-formats-ext`；OpenCV 单独/三路同开测试；必要时换 USB 口或 MJPEG  

### 8.6 `missing action features ... expected action.arm.position`

服务端返回的是合并键 **`action` (T,14)**。更新 `policy/lingbotvla_remote/policy.py` 即可（不必重启推理服务，重启客户端）。

### 8.7 只有左臂动

1. 看 `[Action]` 里 `piper_right` 关节是否在变  
2. 变而不动 → 查 `can_s1` / 使能 / 急停  
3. 几乎不变 → 可能是当前策略/任务下右臂保持，或状态编码问题  
4. 用遥操作确认右臂硬件正常  

### 8.8 卡在 `connected` 没有 `[Action]`

- 等 20–30s（首次推理慢）  
- 查云端是否有 `sample action time`  
- 查隧道是否仍 `ok`  
- 客户端 Ctrl+C 后按 隧道 → 客户端 顺序重启  

### 8.9 QFontDatabase 字体警告

OpenCV/Qt 字体提示，可忽略。

---

## 9. 训练相关（简要）

- 配置：`configs/vla/dual_piper_8tasks.yaml`  
- 输出：`output/dual_piper_8tasks_v30/`，`save_steps` 可改为 `5000` 后续训  
- `enable_resume: true`：从最新 `global_step_*` 接着训（如 10K → 再存 15K/20K…）  
- 续训 screen 示例：`/tmp/start_lingbot_resume.sh` + `screen -dmS lingbot_8tasks_resume ...`  
- 盯进度：`tail -f output/dual_piper_8tasks_v30/checkpoints/loss.jsonl`  
- 10K 存完后建议先 `rsync` 备份整份 `global_step_10000` 再停训/改配置  

---

## 10. 相关文件（real_deploy）

| 文件 | 作用 |
| --- | --- |
| `main_real_xvla.py` | 真机入口；config / RealRuntime 兼容 |
| `policy/lingbotvla_remote/` | WebSocket 远程策略客户端 |
| `task/task_base.py` | 任务循环、相机等待、策略注册 |
| `robot/robot_base.py` | `PickAndPlaceTaskRobot` 双臂兼容 |
| `configs/STACK/LingBotVLA_REAL_PIPER_8tasks.yaml` | 实机配置 |

---

*文档随 8-task 双臂远程部署实践整理。端口、密钥、`/dev/video*` 以现场为准。*
