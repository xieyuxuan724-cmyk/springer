# BENI 分层避障训练与原生播放

> 场地长宽、障碍容量/分布、通路检查和超时已统一参数化。大场景命令与参数表见 [场景与压力测试](AVOIDANCE_SCENES.zh-CN.md)。

> 当前新训练默认采用 **V2**。最新观测、奖励、探索参数及重训命令见 [V2 说明](AVOIDANCE_V2.zh-CN.md)。下文原始 135 维、零命令混合采样等描述为 V1 历史设计；旧检查点仍按 V1 加载。

实现计划见 `AVOIDANCE_IMPLEMENTATION_PLAN.zh-CN.md`。当前完成代码和小规模验证；**正式两小时试验须等用户确认 GPU 空闲后再开始**。`gpu_smoke` 是单次更新的管线验证检查点，不是训练成功的避障策略。

## 如何复用现有训练代码

`beni_avoidance/training.py` 从 `beni_rl.rl_cfg.beni_velocity_flat_ppo_runner_cfg()` 读取原有 PPO 配置，使用现有 RSL-RL `OnPolicyRunner`、PPO、优化器、归一化和 TensorBoard。仅将上层网络改为 135→128→128→64→2 MLP，并通过 VecEnv 适配新的分层环境。没有另写 PPO。

环境代码位于 `scripts/beni_avoidance/`：`common.py` 共享底层推理、力矩及观测；`world.py` 生成布局；`backend.py` 提供原生 CPU/MuJoCo Warp GPU；`env.py` 实现任务。训练配置加载会使用项目依赖；原生播放不创建 mjlab 环境、不使用 Viser。

冻结的底层来自 `assets/models/policy.pt`，40 维观测、7 维动作，GRU 每环境独立。物理 500 Hz、底层 100 Hz、上层 10 Hz。沿用原生播放器的名义 PD、力矩上限、弹簧与 2 个物理步动作延迟；默认头部侧倾目标为零、无跳跃指令。底层文件 SHA256 随上层检查点保存并在加载时核对。

上层单帧为 36 条前向 180°、2 m 射线，以及方向误差 sin/cos、实际前进速度/偏航速度、两个重力分量、目标速度、上一速度命令，共 45 维；叠 3 帧为 135 维。射线是水平二维解析相交，测试已与 MuJoCo `mj_ray` 对齐。障碍物坐标不直接输入策略；目前没有射线噪声、低矮障碍、动态障碍或真实传感器模型。

动作对高斯潜变量作 tanh：前进命令映射至 [0, 目标速度]，偏航至 [-1,1] rad/s，再限命令变化率。目标速度范围 [0,.5] m/s。固定世界方向，允许绕行后恢复方向，不要求回到原直线。静态箱体和圆柱采用 8 个固定尺寸槽，随机选择槽和位置；并非连续随机尺寸。布局经保守膨胀栅格通路检查（几何可通行不等于动态可完成）。

## 小规模检查

在仓库根目录执行：

```bash
uv run python -m unittest discover -s scripts -p test_avoidance.py -v
uv run python scripts/check_avoidance.py --backend native --num-envs 1 --steps 30
uv run python scripts/check_avoidance.py --backend warp --num-envs 2 --steps 10
```

首次 Warp 编译可能较慢，初始化时间与采样速度分别记录。Torch 峰值显存不包含 Warp 分配，必须同时查看 `nvidia-smi`。不要将当前共享 GPU 上的双环境吞吐直接外推到独占 GPU。

## 正式训练（待确认 GPU 空闲）

先用 `check_avoidance.py` 比较 32/64 个环境，再选择批量；不要照搬底层训练的 4096 环境。建议首轮运行：

```bash
uv run python scripts/train_avoidance.py --backend warp --num-envs 32 \
  --iterations 2000 --steps 64 --minutes 80 --seed 42
```

80 分钟是初始化之后训练采样/优化的预算，按采样步检查；保存等收尾稍有额外耗时。迭代上限或预算先达到即停止。课程按采样进度：前 5 个迭代空场，第 5–19 个迭代 1–2 障碍，第 20 个起 3–8 障碍；新阶段在环境重置时生效。课程不是成功率门槛，需观察日志。

输出默认在 `logs/rsl_rl/beni_avoidance/YYYY-MM-DD_HH-MM-SS/`：

- `model_*.pt`、`model_final.pt`：上层 actor/critic、归一化、优化器和元数据。
- `manifest.json`：配置、底层哈希、环境数、软件版本。
- `episodes.jsonl`：逐回合原因、进度和奖励。
- TensorBoard events 与 `result.json`：训练指标和退出状态。

续训命令：

```bash
uv run python scripts/train_avoidance.py --backend warp --num-envs 32 \
  --resume logs/rsl_rl/beni_avoidance/<run>/model_final.pt --minutes 80
```

恢复优化器、网络、归一化、课程计数及随机状态；仿真回合和底层 GRU 会重新开始，不是逐物理步完全复现。预算在半个 rollout 结束时，未完成 rollout 不执行 PPO 更新。`model_final` 仅表示最后保存，**不是验证集最优模型**。

## 验证与最终测试

训练随机种子在 [0,1000000)，验证从 2000000 开始，最终测试从 3000000 开始，诊断使用 9000000 段。先用验证集选检查点，最终测试结果不可用来反复调参。

```bash
uv run python scripts/evaluate_avoidance.py --backend native --num-envs 4 \
  --policy logs/rsl_rl/beni_avoidance/<run>/model_final.pt \
  --episodes 100 --seed-base 2000000 --output logs/avoidance_validation/validation.json

uv run python scripts/evaluate_avoidance.py --backend native --num-envs 4 \
  --policy logs/rsl_rl/beni_avoidance/<run>/model_final.pt \
  --episodes 100 --seed-base 3000000 --output logs/avoidance_validation/final_test.json
```

去掉 `--policy` 即运行规则基线，同一后端、种子、速度和布局可比较。加 `--zero` 独立测零速度停稳，不计入非零速度成功率。GPU 评估改为 `--backend warp`；最终仍需原生后端验证迁移。

非零速度依次 .1/.3/.5 m/s（100 场为 34/33/33），沿目标方向推进 3 m，时限 6/v+5 秒；越侧界、后退超过 .4 m、5 秒推进不足 .05 m、跌倒或任何机器人/障碍接触失败。正常轮子/地面接触允许。最初 .5 秒为底层稳定期；其后非轮部件触地算跌倒。子步接触累积，不只检查上层最后一帧。零速度在 5 秒末位移小于 .15 m 且速度绝对值小于 .1 m/s 为成功，同时仍不能碰撞/跌倒/越界。

报告含逐场失败原因、分速度成功率、时间、净进度、平均方向误差、保守二维轮廓间距。这个间距指标不是网格碰撞的精确最短距离。100 场最终协议目标 ≥90%，代码不会保证或伪造达标。原生与 Warp 接触求解存在差异；当前 Warp 提示圆柱/圆柱及圆柱/箱体部分碰撞对只生成一个接触点，必须用双后端结果评估影响。

## 原生 MuJoCo 展示

规则基线：

```bash
uv run python scripts/play_avoidance.py --speed .3 --seed 2000001
```

训练策略：

```bash
uv run python scripts/play_avoidance.py \
  --policy logs/rsl_rl/beni_avoidance/<run>/model_final.pt
```

仅用数字小键盘：8/2 调目标速度、4/6 调目标方向并重置、5 停止、0 新布局、点号重置、Enter 暂停。原生字母键/F 键保留自己的功能。回合结束后点号重置；变更方向会重建当前场景。显示棋盘地面、箱体/圆柱、绿色射线（近障碍变红）、黄色目标方向。场地有侧向任务边界，棋盘本身不是边界指示。

无窗口检查及截图：

```bash
uv run python scripts/play_avoidance.py --headless --duration 2
# 截图需要可用 OpenGL 上下文：
uv run python scripts/play_avoidance.py --headless --duration 2 --screenshot /tmp/avoidance.png
```

## 当前验证结果

- 5 项避障回归：射线与 mj_ray、底层策略/力矩一致性、密集布局可重现且连通、部分环境重置隔离、子步碰撞锁存。
- CPU 单迭代 PPO 与续训成功；RTX 5060 Ti 双环境、8 步、单迭代 PPO 成功（16 条 transition），检查点在 `logs/rsl_rl/beni_avoidance/gpu_smoke/model_final.pt`。
- 原生无窗口加载 GPU 检查点成功；OpenGL 棋盘/障碍/射线截图验证成功。
- 规则基线 3 个诊断场景：2 次碰撞、1 次停滞；不是正式验收，也未证明上层策略有效。详情 `logs/avoidance_validation/rule_smoke.json`。
- 完整训练、100 场最终评估和成功率验收尚未执行。等用户确认 GPU 空闲后开展首轮约 20/80/20 分钟试验。

### 底层能力限制（正式试验前的决策点）

空场直接给底层固定命令，稳定期后观察 5 秒：零速度/零转向仍前进约 **0.353 m**，因此零命令停稳失败；.1/.3/.5 m/s 的进度分别约 .795/1.610/2.443 m，左右 .5 rad/s 转向短测未跌倒。完整记录 `logs/avoidance_validation/bottom_smoke.json`，其中 reason=0 仅表示短测结束时尚未触发终止，不代表通过完整导航。

冻结底层、上层不输出负速度的约束使停车漂移难以主动抵消。正式试验前应先决定是否修正/更换底层策略，再重新验证名义部署参数；当前实现没有修改底层权重、没有放宽停稳标准，也不声称避障训练能修复该问题。

## TensorBoard：新增诊断指标与周期验证

启动实时面板（不启动训练）：

```bash
uv run tensorboard --logdir logs/rsl_rl/beni_avoidance --host 127.0.0.1 --port 6006
```

浏览器访问 `http://localhost:6006`。旧运行不会补生成新指标；新训练自动记录：

| 标签组 | 含义 |
|---|---|
| `RewardStep/*` | 每个环境步的实际加权奖励贡献：推进、速度误差、方向、近障碍、命令平滑、停车速度/位移、成功及各失败惩罚；含稳定期零值 |
| `RewardEpisode/*` | 完成回合各奖励分项累计值的平均，分项之和对应平均回合回报 |
| `Avoidance/*` | 成功、进度、方向误差、速度误差、最小间距、回合持续时间 |
| `Failure/*` | 碰撞、跌倒、越界、停滞、超时的回合发生率；事件可能同时发生，因此不保证加和为 1 |
| `TrainSpeed/*` | 零速度、低速 [0.1,0.2)、中速 [0.2,0.4)、高速 [0.4,0.5] 分组成功率和碰撞率 |
| `TrainStage/*` | 按回合生成时的课程阶段分别统计成功率 |
| `Curriculum/scheduled_stage` | 当前调度阶段；正在运行的旧回合仍保留其原阶段 |
| `Stop/*` | 零速度回合的末端位移、最大位移、末端机身前进速度绝对值 |
| `Validation/native/navigation/*` | 独立验证集成功率、失败原因、0.1/0.3/0.5 m/s 分速度成功率与样本数，以及进度/误差等 |
| `Validation/native/zero/*` | 独立零速度验证成功率、停车位移和速度等 |

回合指标按**每个完成回合等权**计算，不对各仿真步的批次均值再次平均。该迭代没有对应速度组完成回合时不写该组数据，不能把缺失点解读为零成功率。`RewardStep` 是步均值，`RewardEpisode` 是回合累积均值，两者量纲和分母不同。`numerical_correction` 正常应为零或浮点舍入量，异常时反映数值回退对奖励的修正。奖励权重和任务判定未修改。

停车最大位移按 10 Hz 采样，包含初始化稳定期；末端速度指标是机身前进速度，不能代替完整平面速度/角速度停稳标准。非零速度验证的位移指标表示离起点距离，不应解读为停车漂移。

### 自动独立验证

训练默认配置：

```text
--validation-interval 50
--validation-episodes 12
--validation-zero-episodes 3
--validation-seconds 120
```

在迭代索引 50、100、150……保存检查点时验证一次（索引从 0 开始）；首个 0 号检查点不验证，最后重复保存不重复验证。设置 `--validation-interval 0` 可关闭。导航回合数须为 3 的倍数，以均衡覆盖三档速度。

导航固定种子从 2000000 开始，停车从 2100000 开始；每个检查点使用相同验证场景。使用独立原生 CPU 环境和独立确定性推理模型，不更新训练模型的归一化、不推进训练环境/底层 GRU，也保留训练随机状态。训练在验证期间暂停采样，验证耗时计入 `--minutes` 总预算，并受单轮 120 秒上限约束；原有 `Perf` 吞吐只统计训练采样/更新，不含验证开销。

每轮完整结果写入运行目录 `validation/<iteration>_navigation.json` 和 `_zero.json`。若超时，写 `_incomplete.json`，不把部分回合成功率画成完整验证结果。TensorBoard 验证曲线用于选模型/调参，不属于最终 100 场验收。

### 手动评估也可写入曲线

```bash
uv run python scripts/evaluate_avoidance.py --backend native \
  --policy logs/rsl_rl/beni_avoidance/<run>/model_100.pt \
  --episodes 12 --seed-base 2000000 \
  --output logs/avoidance_validation/manual_100.json \
  --tensorboard-dir logs/rsl_rl/beni_avoidance/<run>/manual_validation
```

横轴默认取检查点迭代，也可用 `--step` 指定。验证、最终测试、诊断种子以及不同后端使用不同标签前缀；不同策略/规则基线应使用不同事件目录。JSON 保留逐场结果。

本次 CPU 小规模验证输出在 `logs/rsl_rl/beni_avoidance/metrics_smoke/`：2 个训练迭代及 3 场导航、1 场停车验证，已读取事件文件核对指标。它只验证记录链路，不是正式训练或效果验收。
