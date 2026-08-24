---
title: "真机部署踩坑 01：双臂左右语义标反，两只手为什么向外抓"
date: 2026-08-19T12:00:00+08:00
draft: false
categories: ["机器人真机部署踩坑"]
tags: ["VLA", "LingBot", "双臂机器人", "真机部署", "数据质量"]
summary: "一次双臂 VLA 真机动作左右相反的完整排查：如何证明原始数据的字段名与物理手臂不一致，如何兼容已训好的权重，以及今后如何验收左右语义。"
ShowToc: true
TocOpen: true
---

## 1. 现象：目标在中间，两只手却向外走

双臂 LingBot VLA 权重训练完成后，真机抓取出现了一个很典型的错误：

```text
抓取目标在双臂中间
        ↓
模型看起来在让两只手向目标靠拢
        ↓
真机却是两只手分别向外运动
```

这类“左右对称但方向完全相反”的动作，通常不是 MAE 偏高或训练不足，而是左右手臂的语义在某一层被交换了。

本次硬件已经通过分别断电确认：

```text
can_s1 = 物理右臂
can_s2 = 物理左臂
```

左右腕部相机的物理连接也已单独确认。

## 2. 先说结论

该批原始数据在数值上的实际顺序是：

```text
[物理右臂6轴, 右夹爪, 物理左臂6轴, 左夹爪]
```

但 `meta/info.json` 把相同的 14 个数声明成了：

```text
[piper_left 6轴, piper_left_gripper,
 piper_right 6轴, piper_right_gripper]
```

因此，不是 `info.json` 里没写“左臂在前”，而是**“左臂在前”这个字段标签本身与物理数据不符**。

已经排除的环节包括：

- 后续 8 任务 merge 没有重排 `observation.state` 和 `action`；
- V1 与 V2 的 `dual_piper.yaml` 都保留原始块顺序；
- `FeatureTransform` 只做拆分、拼接和归一化，没有交换左右；
- 归一化统计不会改变维度顺序；
- 推理服务返回 14 维动作时没有隐式交换；
- 真机框架按 `robots` 配置顺序拼接 state，也按同样的顺序拆分 action。

所以，模型没有“自己学反”。它只是忠实地学习了训练数据中实际存在的顺序。

## 3. 不要把字段名当成物理事实

这次问题最容易误判的地方是：

> `info.json` 明明写了 `piper_left` 在前，为什么还说原始数据左右标反？

因为 metadata 只是 schema，它声明“我认为第 0～6 维是左臂”，但它不会自动验证这些数是否真的来自物理左臂。

必须分清三层语义：

```text
物理层：机器人面向工作区时，哪一台是左臂/右臂
   ↕
接口层：can_s1、can_s2、USB 拓扑和 SDK 对象分别连到哪只手
   ↕
数据层：向量的第 0～6 维和第 7～13 维被叫作什么
```

只检查任意一层，都不足以证明整条链路正确。

## 4. 如何从原始轨迹证明左右标反

核验使用的是原始单任务数据，而不是后来的 8 任务合并集。方法是：

1. 在 parquet 中找到“第 1 个 7 维块明显运动，第 2 块基本不动”的时间窗口；
2. 找到“第 2 块明显运动，第 1 块基本不动”的时间窗口；
3. 提取相同时刻的 `wrist_left`、`wrist_right` 和固定顶视相机帧；
4. 观察哪路腕部画面发生全局运动，并用顶视角确认物理手臂；
5. 至少在 3 条独立轨迹上重复，避免被另一只手、夹爪或物体的局部运动误导。

三条原始轨迹的结果如下。表中的运动量为对应时间窗口内 6 个关节的 L1 变化，夹爪不参与判定。

| 原始轨迹 | 时间窗口 | 第 1 块运动 | 第 2 块运动 | 主要移动的腕部相机 | 判定 |
|---|---:|---:|---:|---|---|
| episode 0 | 19.633～20.133 s | 1.136 | 0.000 | `wrist_right` | 第 1 块 = 物理右臂 |
| episode 0 | 5.300～5.800 s | 0.125 | 0.702 | `wrist_left` | 第 2 块 = 物理左臂 |
| episode 1 | 6.800～7.300 s | 1.487 | 0.012 | `wrist_right` | 第 1 块 = 物理右臂 |
| episode 1 | 4.033～4.533 s | 0.000 | 0.434 | `wrist_left` | 第 2 块 = 物理左臂 |
| episode 2 | 7.800～8.300 s | 2.091 | 0.002 | `wrist_right` | 第 1 块 = 物理右臂 |
| episode 2 | 5.533～5.867 s | 0.000 | 0.098 | `wrist_left` | 第 2 块 = 物理左臂 |

还可以用 PSNR/SSIM 作为辅助指标。对固定场景中的腕部相机，相机随手臂运动时，前后帧的 PSNR 和 SSIM 通常会明显下降。但这只能作为辅助证据：另一只手或物体经过画面时也会降低指标，最终仍应与固定顶视画面交叉验证。

### 4.1 8 个任务、每任务 10 条的批量复核

上面 3 条轨迹确认问题后，又对全部 8 个原始任务做了分层抽样。每个任务都有 100 条轨迹，固定抽取：

```text
episode = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
```

这样可以覆盖每批采集的开头、中间和结尾，比只检查前 10 条更容易发现采集中途重启、换线或 USB 枚举变化。

每条轨迹对两个关节块各选 6 个不重叠、高隔离度的运动窗口，总计：

```text
8 个任务 × 10 条轨迹 = 80 条轨迹
80 条 × 12 个窗口 = 960 个运动窗口
```

批量结果：

- 80/80 条轨迹都高置信判定为“物理右臂在前”；
- 0 条被判定为正常左臂在前；
- 0 条为歧义结果；
- 960 个单独窗口中 943 个方向一致，窗口级一致率为 98.23%；
- 80/80 条轨迹的 `action` 都与 `observation.state` 保持同块顺序，没有在 action 一侧再交换回来。

| 原始任务 | 抽样轨迹 | 右臂在前 | 正常左臂在前 | 歧义 | 中位交换分数 | 最低窗口得票率 |
|---|---:|---:|---:|---:|---:|---:|
| hold the tray with both hands | 10 | 10 | 0 | 0 | 5.829 | 91.7% |
| pack the toothbrush holder | 10 | 10 | 0 | 0 | 5.065 | 91.7% |
| place objects into desk drawer | 10 | 10 | 0 | 0 | 4.209 | 83.3% |
| put in pen container | 10 | 10 | 0 | 0 | 3.422 | 83.3% |
| put the pencil case into the schoolbag | 10 | 10 | 0 | 0 | 3.094 | 91.7% |
| put the shoes back | 10 | 10 | 0 | 0 | 3.937 | 91.7% |
| tidy up the makeup table | 10 | 10 | 0 | 0 | 4.313 | 100.0% |
| tie a knot | 10 | 10 | 0 | 0 | 3.773 | 91.7% |

交换分数越大，越支持“第 1 块对应右腕、第 2 块对应左腕”。本次 80 条的最小分数为 2.619，高于 0.45 的判定阈值。最低得票轨迹仍有 10/12 个窗口方向一致；少数单窗口受双臂同时运动或物体经过画面干扰，不影响整条轨迹的中位判定。

需要注意，这是对 800 条原始轨迹中 80 条的均匀抽样，不等于逐条检查了全部 800 条。但 8 个任务、采集全时段的抽样结果都同向，说明这是系统性的数据协议问题，而不是某一条轨迹的偶发损坏。

完整批处理脚本保存为：

```text
scripts/verify_dual_arm_dataset_semantics.py
```

本次执行命令：

```bash
python scripts/verify_dual_arm_dataset_semantics.py \
  --episodes-per-task 10 \
  --windows-per-block 6 \
  --gpu 7 \
  --workers 4 \
  --output-dir output/dual_arm_semantics_audit_8tasks_10each
```

## 5. 可复用的“单臂运动窗口”搜索脚本

下面的脚本会对 14 维 `observation.state` 搜索 0.5 秒窗口，并分别找出两个 7 维块最孤立的运动区间。

```python
import numpy as np
import pyarrow.parquet as pq

PARQUET = "data/chunk-000/episode_000000.parquet"
FPS = 30
WINDOW_SECONDS = 0.5

table = pq.read_table(
    PARQUET,
    columns=["timestamp", "observation.state"],
)
timestamps = np.asarray(table["timestamp"], dtype=float)
states = np.asarray(table["observation.state"].to_pylist(), dtype=float)

window = round(FPS * WINDOW_SECONDS)
candidates = []

for start in range(len(states) - window):
    stop = start + window

    # 只比较双臂的 6 个关节，不让夹爪开合干扰判定。
    delta = np.abs(states[stop] - states[start])
    block_1 = float(delta[0:6].sum())
    block_2 = float(delta[7:13].sum())
    candidates.append((start, stop, block_1, block_2))

def most_isolated(block_index, min_motion=0.3):
    ranked = []
    for start, stop, block_1, block_2 in candidates:
        active, other = (
            (block_1, block_2)
            if block_index == 1
            else (block_2, block_1)
        )
        if active < min_motion:
            continue

        # 0.02 用于防止静止块为 0 时除零。
        score = active / (other + 0.02) + active
        ranked.append((score, start, stop, active, other))

    return max(ranked)

for block_index in (1, 2):
    score, start, stop, active, other = most_isolated(block_index)
    print(
        f"block={block_index}, "
        f"time={timestamps[start]:.3f}-{timestamps[stop]:.3f}s, "
        f"active={active:.4f}, other={other:.4f}"
    )
```

选出时间窗口后，用 ffmpeg 从三路视频中抽取同一时刻的帧：

```bash
ffmpeg -ss 6.800 -i episode_000001.mp4 -frames:v 1 before.png
ffmpeg -ss 7.300 -i episode_000001.mp4 -frames:v 1 after.png
```

注意：只比较窗口起止点可能遗漏“中间运动后又回到原位”的情况。对存疑窗口，应把指标改为逐帧路径长度：

```python
segment = states[start : stop + 1]
path = np.abs(np.diff(segment, axis=0)).sum(axis=0)
block_1_path = path[0:6].sum()
block_2_path = path[7:13].sum()
```

## 6. 错误最早出现在哪一层

先把这里的两个问题分开：

```text
问题 A：原始数据的物理左右是否标反？
答案：是，已经由 8 个任务、每任务 10 条，共 80 条原始轨迹复核。

问题 B：采集链路中的哪一个具体操作首先把它标反？
答案：现有历史快照不足，无法在 CAN 运行时绑定和数据写入命名之间唯一定位。
```

因此，下面保留不确定性的只是“最早的具体触发点”，不是“原始数据是否标反”这个结论。

现有证据能够精确定：

```text
原始单任务 parquet 中
数值已经是 [物理右臂, 物理左臂]

同一数据集的 info.json 却把它声明成
[piper_left, piper_right]
```

因此，错误肯定早于后续合并和训练。最可能的触发点是采集运行时的 CAN 别名与物理手臂绑定，或写入数据时使用的机器人对象名称。

但由于当时没有保留以下快照，还无法在这两个点之间做唯一定位：

- 采集当天 `can_s1/can_s2` 对应的 USB 拓扑；
- 两只物理手臂的设备序列号；
- 采集进程最终展开后的 robot list；
- 一段“单独移动左臂/右臂”的采集前验收轨迹。

这里必须保持边界：**已经证明的是原始数据标签与物理语义不一致；尚未能唯一证明的是采集当天究竟哪一行运行时配置首先引入了错误。**

## 7. 已训两周的权重要不要重训

不需要因为这个问题立即重训。

只要训练链路中没有另一次左右交换，现有 V1/V2 checkpoint 学到的稳定协议就是：

```text
模型第 1 个 7 维块 = 物理右臂
模型第 2 个 7 维块 = 物理左臂
```

部署端可以使用兼容映射：

| 模型块 | 部署名称 | CAN 接口 | 实际物理手臂 |
|---|---|---|---|
| 第 1 块 | `piper_left` | `can_s1` | 右臂 |
| 第 2 块 | `piper_right` | `can_s2` | 左臂 |

这个配置在命名上是反的，但会让数值正确地路由到物理手臂。

有三个必须注意的细节：

1. **state 输入和 action 输出必须同时使用相同的兼容顺序。**只换输出动作，模型看到的本体状态仍然是错的。
2. **夹爪必须跟随各自的 6 轴手臂一起交换。**不能只换 6 个关节。
3. **已确认物理正确的腕部相机不要跟着交换。** `wrist_left` 仍应是物理左腕，`wrist_right` 仍应是物理右腕。

长期来看，比“故意写反机器人名称”更安全的做法，是保持真机配置的物理语义正确，并在策略客户端增加显式兼容层：

```text
真机 state [左, 右]
        ↓ checkpoint 兼容置换
模型 state [右, 左]
        ↓ 模型推理
模型 action [右, 左]
        ↓ checkpoint 兼容置换
真机 action [左, 右]
```

并且这个兼容层应该与 checkpoint ID 绑定，不要设置成所有模型的全局默认行为。

## 8. 以后如何验收左右语义

### 8.1 先定义“左”和“右”的观察视角

所有人、配置和文档必须使用同一定义：

> 以机器人自身面向工作区的视角定义物理左臂和物理右臂。

不要使用“摄像头画面左侧”或“人站在机器人对面时的左侧”作为默认语义。

### 8.2 硬件层：通过断电和序列号确认 CAN

验收时不要只看 `can_s1` 这种别名，应实际执行：

1. 只给物理左臂上电，读取所有 CAN 接口的设备响应；
2. 只给物理右臂上电，重复读取；
3. 记录 USB 拓扑、CAN 适配器序列号和物理手臂之间的关系；
4. 优先建立基于设备序列号的持久化 udev 别名，而不是依赖可能随插拔变化的 USB 端口号。

建议保存机器可读的硬件清单：

```yaml
physical_arms:
  left:
    can_alias: can_s2
    adapter_serial: "<record-me>"
  right:
    can_alias: can_s1
    adapter_serial: "<record-me>"
```

### 8.3 相机层：遮挡镜头，不要只看 `/dev/videoN`

左右腕部相机的验收方法：

1. 用手完全遮住物理左腕相机；
2. 同时查看采集程序中哪个 key 变黑；
3. 对物理右腕相机重复操作；
4. 记录相机 USB 序列号和稳定 symlink。

`/dev/video6`、`/dev/video7` 这类编号可能随重启和插拔变化，不应作为唯一证据。

### 8.4 SDK state 层：每次只手动一只手臂

电机不下发动作，只读取 SDK state：

1. 手动小幅移动物理左臂；
2. 确认只有声明为 left 的 7 维块发生变化；
3. 手动小幅移动物理右臂；
4. 确认只有 right 块发生变化；
5. 把原始 14 维向量和各维名称一起写入验收日志。

这一步必须在数据采集前执行，而不是训练出问题之后才检查。

### 8.5 Dataset 层：用三路证据做交叉验证

数据集制作后，随机抽取至少 3 条轨迹，每条都要同时对照：

```text
14 维 state/action 变化
        +
左右腕部相机的全局运动
        +
固定顶视相机中的物理手臂
```

如果三者不一致，数据不得进入 merge 和训练。

还应同时检查 `observation.state` 和 `action`。它们可能来自不同的 SDK 对象或不同的拼接代码，不能因为 state 正确就默认 action 一定正确。

### 8.6 部署层：同时记录“模型顺序”和“物理路由”

启动策略时，打印且保存：

```text
checkpoint ID
checkpoint 期望的 arm order
SDK state 的 arm order
14 维 state 的前后两个块
14 维 action 被拆分后的目标 SDK 对象
can_s1/can_s2 当前对应的物理手臂
三路相机的序列号与语义 key
```

第一次下发动作前，先使用不上电或不执行的 dry-run 模式检查数值路由。真机测试时必须使用低速、小幅限幅、无障碍安全位姿，并保证急停可立即触发。

## 9. 采集前的最小验收清单

以后每次更换 USB 端口、CAN 适配器、摄像头、机器人或采集电脑后，都重新执行：

- [ ] 以机器人自身视角标记物理左/右臂；
- [ ] 通过分别断电确认两个 CAN 接口；
- [ ] 记录 CAN 适配器序列号和 USB 拓扑；
- [ ] 通过遮挡镜头确认左右腕部相机；
- [ ] 记录相机序列号，不只记录 `/dev/videoN`；
- [ ] 手动移动物理左臂，检查 state 只有 left 块变化；
- [ ] 手动移动物理右臂，检查 state 只有 right 块变化；
- [ ] 在安全条件下分别验证两个 action 块的物理目标；
- [ ] 录制一条左臂单独运动、一条右臂单独运动的验收轨迹；
- [ ] 从 parquet 和三路视频反向核对上述轨迹；
- [ ] 对最终部署路径执行 dry-run，检查 state/action 双向映射；
- [ ] 将验收结果、配置版本和设备清单与数据集一起归档。

## 10. 本次经验

这次最重要的经验不是“左右反了就在部署端换一下”，而是：

1. metadata 是声明，不是物理真值；
2. `can_s1`、`/dev/video6` 等系统名字也不是稳定的物理语义；
3. 左右语义必须用硬件、state、action 和视频做端到端闭环验证；
4. 对已训好的 checkpoint，保持它学到的数值协议并增加显式兼容层，通常比盲目重训更合理；
5. 对新数据，必须先修复物理语义并重新生成统计量，不应继续传播这个历史兼容顺序。

一句话总结：

> 在机器人系统中，一个变量叫 `left` 并不代表它真的连到物理左臂；只有端到端的可观测验证才能证明语义正确。
