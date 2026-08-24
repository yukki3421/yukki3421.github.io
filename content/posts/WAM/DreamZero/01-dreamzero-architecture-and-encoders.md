---
title: "DreamZero 架构阅读笔记（一）：图像编码器、VAE 与 DiT 输入"
date: 2026-08-14
draft: false
math: true
---

# DreamZero 架构阅读笔记（一）：图像编码器、VAE 与 DiT 输入

本文记录 DreamZero 的整体架构，以及图像编码器、视频 VAE 和 DiT 输入的形状变化。本文主要讨论默认的 Wan2.1 I2V 14B 路线；Wan2.2 5B 的差异放在文末。

## 1. 整体架构

DreamZero 不是简单的“视觉编码器 + LLM + 动作头”，而是把 Wan 视频生成 DiT 攦造成同时预测未来视频和机器人动作的 World Action Model。

```text
语言指令 ── T5 ─────────────────┐
当前首帧 ── CLIP Image Encoder ─┤
视频/首帧 ── Video VAE ─────────┼─→ Causal Wan DiT ─┬─→ 视频 flow prediction
机器人状态 ── State Encoder ────┤                   └─→ 动作 flow prediction
带噪动作 ── Action Encoder ─────┘
```

顶层 `VLA` 只是编排器。默认的 backbone 是 `IdentityBackbone`，真正的主要模型都位于 `WANPolicyHead` 中：

```text
VLA
├── IdentityBackbone
└── WANPolicyHead
    ├── T5 text encoder
    ├── CLIP image encoder
    ├── Video VAE
    ├── FlowMatchScheduler
    └── CausalWanModel
```

训练损失为：

\[
L=L_{\text{video dynamics}}+L_{\text{action}}
\]

## 2. CLIP 图像编码器

DreamZero 使用 OpenCLIP ViT-H/14 的视觉部分，只处理当前首帧，用于提取与语言相关的高层视觉语义。

主要参数：

| 参数 | 数值 |
|---|---:|
| 输入分辨率 | 224 × 224 |
| Patch 大小 | 14 × 14 |
| Patch token 数量 | 16 × 16 = 256 |
| CLS token 数量 | 1 |
| 总 token 数 | 257 |
| 模型隐藏维度 | 1280 |
| Transformer block | 定义 32 层，DreamZero 使用前 31 层 |
| 注意力头数 | 16 |
| 每头维度 | 1280 / 16 = 80 |
| FFN 中间维度 | 1280 × 4 = 5120 |

形状变化：

```text
[B,3,224,224]
    ↓ Conv2d(kernel=14, stride=14)
[B,256,1280]
    ↓ 添加 CLS token 和位置编码
[B,257,1280]
    ↓ 前 31 层 Vision Transformer
[B,257,1280]
```

### 2.1 CLS token

CLS token 是一个不对应具体图像区域的可训练向量，被放在所有 patch token 前面：

```text
[CLS, patch₁, patch₂, ..., patch₂₅₆]
```

它通过 self-attention 汇总整张图像的信息。DreamZero 不只使用 CLS token，而是保留 CLS 和全部 256 个 patch token，从而同时获得全局语义与局部空间信息。

### 2.2 注意力不是 GQA

图像编码器使用标准 Multi-Head Attention：

```text
Q heads = 16
K heads = 16
V heads = 16
head_dim = 80
```

Q、K、V 由一个融合投影产生：

```text
[B,257,1280]
    ↓ Linear(1280, 3840)
Q,K,V 各为 [B,257,1280]
    ↓ 分成 16 个 head
Q,K,V 各为 [B,16,257,80]
```

这里没有 `num_kv_heads` 或 KV head 共享，因此不是 GQA/MQA。

## 3. 图像编码器与 VAE 的区别

| 对比项 | Image Encoder | Video VAE |
|---|---|---|
| 主体结构 | ViT Transformer | 因果 3D 卷积、残差块和 bottleneck attention |
| 处理内容 | 当前首帧 | 完整视频；也用于构造首帧条件 |
| 输出 | `[B,257,1280]` 语义 token | `[B,16,T_z,H_z,W_z]` 连续 latent |
| 主要目标 | 理解图像语义 | 压缩并重建视频 |
| 是否包含 CLS | 是 | 否 |

简化记忆：

```text
Image Encoder：画面表达了什么？
VAE：如何压缩并重建这段视频？
```

## 4. 默认 Wan2.1 Video VAE

假设输入一段视频：

```text
[B,3,33,176,320]
```

VAE 输出：

```text
[B,16,9,22,40]
```

这表示：

```text
RGB 通道：3 → latent 通道：16
时间：33 → 9
高度：176 → 22（8倍压缩）
宽度：320 → 40（8倍压缩）
```

### 4.1 编码步骤

```text
[B,3,33,176,320]
    ↓ CausalConv3d，提取低层视频特征
[B,96,33,176,320]
    ↓ 第一次空间下采样
[B,96,33,88,160]
    ↓ 第二次时空下采样，通道增加
[B,192,约17,44,80]
    ↓ 第三次时空下采样，通道增加
[B,384,9,22,40]
    ↓ ResBlock + Attention + ResBlock
[B,384,9,22,40]
    ↓ RMSNorm + SiLU + CausalConv3d(384→32)
[B,32,9,22,40]
    ↓ 沿通道拆成两组，只保留 μ
[B,16,9,22,40]
```

中间的 `384 → 32` 不是选择前 32 个通道，而是 32 个可学习的 3D 卷积核共同读取全部 384 个输入通道及其局部时空邻域，生成 32 个重新组合的输出特征。卷积权重形状可理解为：

```text
[32,384,3,3,3]
```

随后：

```text
32 = 16 + 16
├── 前16通道：μ，保留
└── 后16通道：另一组分布参数，当前编码路径不使用
```

因此最终的 16 是 VAE 预先设定的 latent channel `z_dim=16`，不是根据图像尺寸计算得到的。

### 4.2 为什么 33 帧变成 9

Wan VAE 单独保留第一帧，后续每 4 帧压缩成一个 latent 时间位置：

```text
frame 0      → latent 0
frame 1...4  → latent 1
frame 5...8  → latent 2
...
frame 29...32 → latent 8
```

因此：

\[
T_z=1+\frac{T-1}{4}=1+\frac{32}{4}=9
\]

第一帧被单独保留，是为了适应因果视频建模和 I2V 条件生成。

## 5. 为什么一次训练调用 VAE 两次

同一个 VAE 会处理两份内容不同的输入。

### 5.1 第一次：编码完整真实视频

```text
[真实首帧, 真实未来帧]
    ↓ VAE
完整视频的干净 latent x₀：[B,16,9,22,40]
```

`x₀` 是模型需要学习生成的真实目标。它会与随机噪声插值得到带噪 latent：

\[
x_t=(1-\alpha_t)x_0+\alpha_t\epsilon
\]

```text
x₀：[B,16,9,22,40]
    ↓ 与噪声插值
xₜ：[B,16,9,22,40]
```

### 5.2 第二次：构造干净首帧条件

第二次输入不是完整真实视频，而是：

```text
[真实首帧, 零帧, 零帧, ..., 零帧]
    ↓ VAE
首帧条件 latent：[B,16,9,22,40]
```

再加入 4 通道 mask：

```text
mask：[B,4,9,22,40]
时间含义：[已知,未知,未知,...]
```

沿通道维拼接：

```text
首帧 latent：16通道
mask：        4通道
─────────────────
y：          20通道
```

这条条件路径不加噪。它只允许模型看到真实首帧，不允许看到真实未来，从而避免答案泄漏。

## 6. 两份 latent 如何输入 DiT

第一次产生待去噪对象 `xₜ`，第二次产生首帧条件 `y`。二者沿通道维 `C` 拼接：

```text
xₜ：[B,16,9,22,40]
y ：[B,20,9,22,40]
          ↓ 沿 C 维拼接
[B,36,9,22,40]
```

这不是两组独立 token。它们在 token 化之前已经合并，同一个时空位置同时包含：

```text
16维带噪视频特征
+ 16维干净首帧条件特征
+ 4维条件 mask
= 36维输入特征
```

## 7. 视频、动作、状态和语言如何对齐维度

### 7.1 视频 token

36 通道输入经过 DiT patch embedding：

```text
[B,36,9,22,40]
    ↓ Conv3d(kernel=(1,2,2), stride=(1,2,2), out=5120)
[B,5120,9,11,20]
    ↓ flatten 时空维并转置
[B,1980,5120]
```

其中：

\[
1980=9\times11\times20
\]

### 7.2 动作和状态 token

```text
原始动作：[B,32,32]
    ↓ Action Encoder
动作 token：[B,32,5120]

原始状态：[B,1,44]
    ↓ State Encoder
状态 token：[B,1,5120]
```

三类 token 统一为 5120 维后，沿序列长度维拼接：

```text
video：[B,1980,5120]
action：[B,32,5120]
state： [B,1,5120]
       ↓ 沿 token 维拼接
[B,2013,5120]
```

序列布局：

```text
[video tokens | action tokens | state tokens]
```

这条序列进入 DiT 的 self-attention，使视频、动作和状态相互交换信息。

### 7.3 语言和 CLIP 图像 token

语言由 T5 编码并投影到 5120 维；CLIP 首帧 token 也会投影到 5120 维。它们不加入上述 self-attention 主序列，而是作为 cross-attention context：

```text
Query： [video | action | state]
Key/Value：[CLIP image tokens | language tokens]
```

因此要区分两种拼接：

```text
latent 通道拼接：
xₜ + y → [B,36,T,H,W]

Transformer 序列拼接：
video + action + state → [B,L_total,5120]
```

语言和 CLIP 特征走 cross-attention，不属于第二种拼接的主序列。

## 8. 每层 DiT 中的 Self-Attention 与 Cross-Attention

这里有两个序列：

```text
主序列 x     = [video | action | state]
条件 context = [image | language]
```

每个 `CausalWanAttentionBlock` 依次执行：

```text
1. 主序列做 Self-Attention
2. 主序列向 context 做 Cross-Attention
3. 主序列经过 FFN
```

### 8.1 Self-Attention

Q、K、V 都来自主序列：

$$
\begin{aligned}
Q &= xW_Q, \\
K &= xW_K, \\
V &= xW_V, \\
x^{\prime} &= x + \operatorname{SelfAttention}(Q,K,V)
\end{aligned}
$$

所以 video、action、state 可以在因果 mask 允许的范围内交换信息。

### 8.2 Cross-Attention

Q 来自主序列，K/V 来自条件序列：

令条件序列为：

$$
C=[\text{image}\mid\text{language}]
$$

$$
\begin{aligned}
Q &= xW_Q, \\
K &= CW_K, \\
V &= CW_V, \\
A &= \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right), \\
x^{\prime} &= x + AV
\end{aligned}
$$

这使每个 video/action/state token 都能选择需要的图像区域和语言词语。

### 8.3 FFN

$$
x^{\prime}=x+\operatorname{FFN}(\operatorname{Norm}(x))
$$

一层结束后，输出仍然只有 `[video | action | state]`，随后进入下一层。

### 8.4 Context 是只读条件

DiT block 逐层更新主序列 `x`，但不更新 `context`。信息流是单向的：

```text
image ────┐
          ├──→ video / action / state
language ─┘
```

image/language 会影响视频和动作预测，但不会反过来接收主序列信息。它们是稳定的外部条件。

context 虽然固定，但每层主序列不同，所以每层 Query 也不同：

```text
x₀ → Self-Attention → Cross-Attention(context) → FFN → x₁
x₁ → Self-Attention → Cross-Attention(context) → FFN → x₂
```

需要区分：

```text
联合建模：video、action、state 在主序列中做 Self-Attention。
条件控制：image、language 通过 Cross-Attention 单向影响主序列。
```

## 9. 推理时的视频 Latent 去噪循环

![DreamZero 推理时的视频 latent 去噪循环](/images/WAM/DreamZero/dreamzero-inference-denoising-loop.svg)

推理从随机视频 latent 和随机动作开始。每一轮 DiT 同时预测视频 flow 与动作 flow；视频 scheduler 根据当前 latent、视频 flow 和 timestep 得到下一步 latent：

$$
z_{t-1}=\operatorname{SchedulerStep}(z_t,v_t,t)
$$

新得到的 ��接进入下一轮 DiT。循环过程始终位于 VAE latent 空间：

```text
当前视频 latent zₜ
    ↓ DiT
视频 flow vₜ
    ↓ FlowUniPC Scheduler
下一步视频 latent zₜ₋₁
    ↓ 再次进入 DiT
...
```

DiT 输出的是 flow prediction，不是 RGB，也不是已经更新完成的 latent。真正更新 latent 的是 `FlowUniPCMultistepScheduler`。

只有所有去噪步骤结束后，才会按需调用一次 VAE Decoder：

```text
最终未来视频 latent z₀
    ↓ VAE Decoder
预测的未来 RGB 视频
```

因此，循环中不会执行“latent → RGB → latent”。RGB 解码只用于最终可视化、保存和诊断；动作与视频 latent 在共享 DiT 中同步去噪，动作预测不需要读取最终解码出的 RGB。

## 10. Wan2.2 5B 的主要差异

Wan2.2 路线使用 `WanVideoVAE38`：

| 项目 | Wan2.1 默认 | Wan2.2 5B |
|---|---:|---:|
| VAE latent 通道 | 16 | 48 |
| 空间压缩倍率 | 8 | 16 |
| DiT hidden dimension | 5120 | 3072 |
| DiT layers | 40 | 30 |
| DiT heads | 40 | 24 |

Wan2.2 配置还设置 `concat_first_frame_latent: false`，因此不能把本文 Wan2.1 的 `16+20=36` 输入通道规则直接套用到 Wan2.2。

## 11. 关键源码位置

- `groot/vla/model/dreamzero/base_vla.py`：VLA 顶层编排。
- `groot/vla/model/dreamzero/backbone/identity.py`：IdentityBackbone。
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`：视频预处理、两次 VAE 调用、加噪与联合损失。
- `groot/vla/model/dreamzero/modules/wan_video_image_encoder.py`：ViT-H/14 图像编码器。
- `groot/vla/model/dreamzero/modules/wan_video_vae.py`：Wan2.1/Wan2.2 VAE。
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`：Causal Wan DiT、动作/状态 register。

## 12. 当前阶段的核心结论

1. Image Encoder 负责首帧语义，VAE 负责可生成的视频 latent。
2. 完整视频经过 VAE 得到 `x₀`，再与噪声插值得到 `xₜ`。
3. 首帧加零未来帧经过同一 VAE，得到不加噪的条件 `y`。
4. 默认 Wan2.1 中，`xₜ` 和 `y` 沿通道维拼成 36 通道，再转成视频 token。
5. 视频、动作、状态被分别投影到 5120 维后组成 self-attention 主序列。
6. 语言和 CLIP 图像 token 作为 cross-attention context。
