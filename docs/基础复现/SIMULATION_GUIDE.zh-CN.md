# BENI 跳跃与行走仿真复现指南

本指南使用仓库自带的 `assets/models/policy.pt` 在 mjlab / MuJoCo Warp 中复现动作，无需先训练。文中的“行走”指轮腿机器人按速度命令在平地运动；动作由策略产生，实际速度可能与命令略有偏差。

## 1. 准备环境

推荐 Linux x86_64、Python 3.12、`uv`。GPU 训练推荐兼容 CUDA 12.8 的 NVIDIA 显卡和至少 8 GB 显存；单机器人播放可尝试 CPU。以下命令都在仓库根目录执行：

```bash
uv sync --locked
test -f assets/models/policy.pt
test -f assets/motions/base_jump_h0.18.npz
```

`uv sync --locked` 需要下载锁定依赖。若在没有桌面浏览器的远程主机运行，请转发 Viser 端口，并使用终端输出的 URL；默认端口为 8080，端口占用时以实际输出为准。

## 2. 启动仿真

```bash
uv run play Beni-Jump-Flat \
  --checkpoint-file assets/models/policy.pt \
  --num-envs 1 \
  --viewer viser
```

等待终端显示 Viser 地址，然后在浏览器打开。启动成功后应看到 BENI 机器人及 Viser 控件。若只想核对模型加载，可将参数改为 `--agent zero` 并去掉 `--checkpoint-file`；零动作代理不会复现训练出的行走或跳跃。

## 3. 复现前进与转向

在 Viser 中展开 **Controls → Commands → Twist** 并勾选 **Enable**。不勾选时命令会自动采样，不适合对照固定动作。

| 动作 | `lin_vel_x` | `ang_vel_z` | 其他设置 |
| --- | ---: | ---: | --- |
| 原地站立 | 0 | 0 | `base_height = 0.11 m`，俯仰与侧倾均为 0 |
| 前进 | 0.5 m/s | 0 | 保持默认姿态 |
| 后退 | −0.5 m/s | 0 | 保持默认姿态 |
| 原地左转 | 0 | 1 rad/s | 保持默认姿态 |
| 原地右转 | 0 | −1 rad/s | 保持默认姿态 |
| 弧线前进 | 0.5 m/s | 0.5 rad/s | 保持默认姿态 |

每次修改后观察机器人数秒。`lin_vel_y` 固定为 0，不能用于横移。若机器人跌倒，使用 **Reset Environment** 后先让它站稳，再设置命令。正的 `ang_vel_z` 从上方看表示左转。

## 4. 复现原地跳跃

1. 保持 Twist 的 **Enable** 勾选，设置 `lin_vel_x = 0`、`ang_vel_z = 0`、`base_pitch = 0`、`base_roll = 0`、`base_height = 0.11 m`。
2. 展开 **Controls → Commands → Jump**，等待 **Measured state** 显示 `ready=True`。平衡和车轮支撑条件需要连续满足 0.2 秒。
3. 点击 **Request one jump**。查看 **Last request** 是否显示 `accepted`，然后观察起跳、腾空与落地。
4. 要重复跳跃，先等待机器人重新平衡、1 秒冷却结束，以及上一次落地监测结束，再次点击。

跳跃参考高度固定为 **0.18 m 的机身抬升量**，不是保证达到的车轮离地高度。实际动作取决于策略。`Last request` 如果显示 `not ready`，等待稳定或重置环境；`cooldown` 表示冷却未结束；`busy` 表示上一次跳跃仍在监测；`disabled` 表示当前跳跃控制未启用。暂停状态下点击按钮的请求会在恢复仿真后处理。

## 5. 复现行进中跳跃

先按第 3 节让机器人以 `lin_vel_x = 0.5 m/s`、`ang_vel_z = 0` 稳定前进，再按第 4 节观察 `ready=True` 并点击 **Request one jump**。若始终不满足就绪条件，减小前进速度，先验证原地跳跃。行进中跳跃比原地跳跃更依赖策略的平衡能力。

## 6. 可选：从训练重新得到策略

仓库自带的权重可直接用于上述复现。若要自行训练，先做小规模命令检查：

```bash
uv run python -m beni_rl.train Beni-Jump-Flat \
  --env.scene.num-envs 4 --agent.max-iterations 10
```

完整训练的 README 示例为：

```bash
uv run python -m beni_rl.train Beni-Jump-Flat \
  --env.scene.num-envs 4096 --agent.max-iterations 15000 --agent.seed 42
```

训练输出位于 `logs/rsl_rl/beni_jump_flat/`。选定运行目录后，可把第 2 节命令的检查点替换为 `logs/rsl_rl/beni_jump_flat/<run>/model_best.pt`。自行训练的结果会受硬件、随机性和训练过程影响，不应预期与仓库所附权重逐帧一致。

## 7. 可选：浏览器游戏快速体验

若只想在网页中体验驾驶与跳跃，可在仓库根目录运行：

```bash
cd web
npm ci
npm run build
npm start
```

打开 `http://localhost:3000/`，进入车库并开始单人游戏。使用 W/A/S/D 或方向键驾驶，空格键跳跃。该路径是 MuJoCo WASM + ONNX 浏览器游戏；要观察机器人任务中的跳跃就绪状态、命令滑块和策略表现，请使用前述 Viser 仿真。
