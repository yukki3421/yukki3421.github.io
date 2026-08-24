---
title: "DreamZero 架构篇：Wan、联合 Flow Matching 与闭环推理"
date: 2026-08-14T16:00:00+08:00
draft: true
categories: ["WAM"]
tags: ["WAM", "DreamZero", "DiT", "VAE", "Flow Matching", "机器人"]
ShowToc: true
TocOpen: false
math: true
---

> 主题：把 DreamZero 的数据流、VAE、训练目标与闭环推理串成一条线；对照公开代码中的具体形状与顺序。
>
> 主读：DreamZero（[arXiv:2602.15922](https://arxiv.org/abs/2602.15922)，[代码](https://github.com/dreamzero0/dreamzero)）。概念背景见本系列第 1 篇。
>
> 约定：默认指 Wan2.1-I2V-14B + `WanVideoVAE`；仓库里可选 Wan2.2-5B / `WanVideoVAE38`，倍率不同，骨架同类。

---

# 1. 一页总览

DreamZero 要解决的工程问题可以压成三句：

```text
1. 视频与动作如何对齐 → 同一 DiT 上联合 flow matching
2. 长程与闭环如何稳定 → 自回归 chunk + 真观测写回
3. 大视频模型如何控机 → 异步执行 + 少步去噪（Flash 留给系统篇）
```

整条流水线：

```text
多相机画面（拼成单帧序列）
        │
        ▼
   VAE Encoder（冻结）──► 视频 latent z
语言 c ─► Text Encoder（冻结）─┐
关节 q ─► State Encoder ──────┤
动作 a ─►（加噪后）Action Enc ─┤
                               ▼
                    Autoregressive DiT（Wan）
                    Joint Flow Matching u_θ
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              未来视频 latent         动作 chunk â
           （隐式视觉规划）         （下发执行）
```

相对 π₀：不是「VLM + 只出 \(a\)」，而是「视频 DiT + 同时出 \(o_{\text{fut}}\) 与 \(a\)」。动作学习更接近隐式逆动力学——命令要对齐预测出的视觉未来。

---

# 2. 问题形式化

条件：语言 \(\mathbf{c}\)、本体感觉 \(\mathbf{q}_l\)、当前及历史观测 \(\mathbf{o}_{0:l}\)。  
输出：未来一段视频与动作，共享 horizon \(H\)：

\[
\pi(\mathbf{o}_{l:l+H},\,\mathbf{a}_{l:l+H}\mid \mathbf{o}_{0:l},\,\mathbf{c},\,\mathbf{q}_l)
\]

概念上可写成「视频预测 × 隐式 IDM」，但实现上是 **一个** \(u_\theta\) 端到端联合去噪，而不是视频模型与 IDM 两段式拼接。作者认为共享表征更有利于 video–action 对齐；预训练视频模型已具备世界如何演化的能力，机器人阶段主要补本机视频分布，并学会从视觉未来抽出动作。

---

# 3. 模块：尽量少改 Wan

| 模块 | 训练时 | 作用 |
|------|--------|------|
| VAE | 冻结 | 像素 ↔ 视频 latent |
| Text / Image Encoder | 冻结 | 语言与（I2V 相关）图像条件 |
| DiT（Wan） | 更新 | 联合去噪骨干 |
| State / Action Encoder、Action Decoder | 更新 | 接入机器人状态与动作 |

多视角不做复杂 fusion，直接 **拼成一张图** 再走原骨干。LoRA 在作者实验里效果不佳，主结果走全量更新 DiT。

---

# 4. VAE：压缩世界，供 DiT 使用

## 4.1 角色

DiT 不在原始 RGB 上做生成，而在 Wan 的 latent 上做 flow matching。VAE 是冻结的时空压缩器：

```text
编码: 像素视频 → latent z
解码: latent z → 像素（可视化用；控制主路径不依赖解码）
```

## 4.2 内部结构（WanVideoVAE）

本质是 **3D 因果卷积自编码器**：

```text
Encoder3d
  CausalConv stem
  → 多层 ResidualBlock + Resample 下采样
  → Middle（含轻量 Attention）
  → 输出 mu（取均值作为 z，再按固定 mean/std 归一化）

Decoder3d：对称上采样回到 RGB
```

默认压缩：

| 维度 | 倍率 | 含义 |
|------|------|------|
| 空间 | **8×**（`upsampling_factor=8`） | \(H'=H/8,\ W'=W/8\) |
| 时间 | **约 4×** | 见下一小节公式 |
| 通道 | **16**（`z_dim=16`） | 信息挤进通道 |

**下采样**：用 stride 把时间/高/宽变稀（常见每次边长 ÷2）。空间连做三次 `/2` 得到 `/8`；时间方向配合 `downsample3d` 得到约 `/4`。

Wan2.2 的 `WanVideoVAE38` 则是 48 通道、空间 16×，读 14B 主叙事时先按上表即可。

## 4.3 时间长度为什么是 \(T'=1+(T-1)/4\)

实现不是简单的整段 \(T/4\)，而是：

```text
第 1 帧单独编码 → 1 个 latent
其余 (T-1) 帧每 4 帧一组 → 每组 1 个 latent
```

因此：

\[
T' = 1 + \frac{T-1}{4}
\]

训练脚本里常见 `num_frames=33`（满足 \(8n+1\)）：

```text
输入:  [3, 33, 176, 320]   # C, T, H, W
输出:  [16, 9, 22, 40]     # 16, 1+8, 176/8, 320/8
```

`frame_seqlen=880` 对应每帧 latent 平面 \(22\times40\)。

---

# 5. Chunk：时间如何对齐

策略按 **chunk / block** 工作，而不是单帧单步：

```text
一段约 1.6s 的未来
  · 视频：若干 latent 帧（论文里常按 K=2 等设定对齐）
  · 动作：一段 action horizon（如 30Hz 下 H=48）
```

源码用 `num_frame_per_block`、`num_action_per_block` 对齐「一截视频 ↔ 一截动作」。自回归主要加在 **视频时间维** 上，避免错误动作在闭环里滚雪球；动作与当前视频块一起生成。

最大视觉 context 约数秒量级（论文写到约 6.6s / 8 latent；具体随配置变）。长程记忆不是本工作的重点评测项。

---

# 6. 训练：Teacher Forcing + 联合 Flow Matching

## 6.1 加噪顺序（务必分清）

```text
视频:  像素 → VAE → 干净 z₁ → add_noise → noisy z → 进 DiT
动作:  归一化 a → add_noise → noisy a → Action Encoder(+t) → 进 DiT
```

- 视频：**先编码，再对 latent 加噪**（不在像素上加噪）。  
- 动作：**先加噪，再进 Action Encoder**（Encoder 同时吃 timestep 正弦特征，风格近 π 系 action encoding）。

## 6.2 Teacher forcing：历史是「干净的」

预测第 \(k\) 个 chunk 时，上下文是数据集里的真值，而不是模型自己上一步的生成：

\[
\mathcal{C}_k=\{(\mathbf{z}_1^j,\,\mathbf{a}_1^j)\}_{j<k}
\]

「干净」= 记号里的 \(z_1,a_1\)（未加噪的真 latent / 真动作）。当前块才是带噪的 \(z_t,a_t\)。这与语言模型左边放真 token、右边学下一个是同一类稳定训练技巧。训练 **不依赖** DiT 的 KV cache；靠整段输入 + causal / block mask 一次算完。

（VAE 编码长视频时内部的 `feat_cache` 是因果卷积实现细节，与推理用的 DiT KV cache 不是一回事。）

## 6.3 联合目标与共享时间步

对当前 chunk，干净量与噪声插值（论文记号）：

\[
\mathbf{z}_t = t\,\mathbf{z}_1+(1-t)\,\mathbf{z}_0,\quad
\mathbf{a}_t = t\,\mathbf{a}_1+(1-t)\,\mathbf{a}_0
\]

模型 \(u_\theta\) 预测联合 velocity，对视频与动作一起做 flow matching 损失。

| 模式 | 时间步 |
|------|--------|
| 默认 DreamZero | video / action **共用** \(t_k\)（仓库默认 `decouple_video_action_noise=false`） |
| Flash / 解耦 | video 偏向高噪声，action 仍可均匀独立采样 |

同一 chunk 内帧共享时间步；不同 chunk 可独立采样。默认共享 \(t\) 有利于早期对齐与收敛；解耦是为少步推理服务，系统篇再展开。

---

# 7. 推理：KV Cache 与 GT 写回

训练是开环 teacher forcing；部署是闭环：

```text
1) Prefill
   当前真图 → VAE → 干净 z 写入 KV

2) 循环
   a. 从噪声出发，联合去噪 N 步 → (预测视频, â)
   b. 滤波后异步下发 â
   c. 执行结束后取新真图 o_real
   d. z_real = VAE(o_real)，更新 KV          ← GT 写回
   e. 丢掉本轮预测视频 latent               ← 不用梦境当历史
```

### 写回什么、何时写

| 问题 | 答案 |
|------|------|
| 何时 | 每个 action chunk 执行完、新观测到达之后 |
| 写什么 | 真像素经 VAE 的 **干净视频 latent**（及由其写入的 K/V） |
| 不写什么 | 模型生成的假未来视频 |

### KV cache 存什么

与 LLM 类似：历史 token 的 Key / Value，避免每步重算整段历史。内容来自 **已经发生的真观测**（Prefill 与 GT 写回），供当前 noisy 视频块与动作 register 做 attention。这是推理加速与短时视觉记忆；训练阶段不需要这套缓存。

### 去噪步数

| 配置 | 典型步数 |
|------|----------|
| 朴素大模型 | 可达 16 步，延迟秒级 |
| 常用优化后 | **约 4 步**（配置里 `num_inference_timesteps` 常见为 4） |
| DreamZero-Flash | 可压到 **1 步**，换速度–精度折中 |

异步执行把约束从「算完才能动」改成「在当前 chunk 耗尽前算完下一段」。Flash 与系统栈（CFG 并行、DiT cache、量化等）见系列后续系统篇。

---

# 8. 自回归而非双向的要点

双向扩散常固定长度，长程示教里为对齐语言 caption 容易被迫 subsample，扭曲原生帧率，伤害 video–action 时间对齐。自回归按 chunk 条件于历史视觉，可保留更接近原生的 FPS，并用 KV cache 加速。消融里 AR 与 BD 的 task progress 可接近，但 AR 动作更平滑、推理明显更快。DreamZero 把自回归用在视频侧，并在闭环用真观测截断开环视频生成的误差累积——这是 WAM 相对「纯开环做梦再 IDM」的结构优势。

---

# 9. 与 π₀ 的架构对照（概念级）

```text
π₀:
  图+文 → VLM → Action Expert（FM）→ a

DreamZero:
  视频 → VAE → z
  z（加噪）+ a（加噪）+ 文/状态 → 同一视频 DiT（联合 FM）
       → o_future 与 a
  闭环：真 z 写 KV，预测视频丢弃
```

相似点：都用 flow matching 出动作 chunk，都吃语言与视觉，都有大预训练骨干。  
不同点：骨干类型（VLM vs 视频 DiT）、是否预测未来、加噪是否跨模态联合、是否依赖 AR 闭环写回。

---

# 10. 要点收束

```text
VAE          先把世界压成 z；时间按「首帧 + 每 4 帧」计数
训练         历史真值 + 当前 video/action 加噪；默认同 t；无 DiT KV
加噪顺序     视频：VAE→加噪；动作：加噪→Action Encoder
推理         联合去噪出 â；真观测写 KV；预测视频不作历史
设计中心     一个 DiT 上对齐「未来画面」与「电机命令」
```

下一篇（数据与泛化）会说明：这样的架构如何配合「宽而少重复」的数据，在未见环境 / 未见运动上相对 π₀.₅ 等 VLA 读出差异。系统篇再专门讲 Flash 与 38× 推理栈。
