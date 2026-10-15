---
title: OpenBeni
emoji: 🏁
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

# OpenBeni

[English](README.md)

项目主仓库：[Mondo-Robotics/open-beni](https://github.com/Mondo-Robotics/open-beni)。开发和拉取请求应提交到该仓库；Hugging Face Space 是此源码的部署版本。

可在 [Hugging Face Spaces](https://huggingface.co/spaces/mondo-robotics/open-beni) 玩浏览器游戏。本仓库以 MIT 许可证提供完整的网页游戏、服务器、训练源码、机器人资源和策略。有关本地开发、加载检查和部署，请参阅 [网页端 README](web/README.md)。

从源码用 Docker 构建并运行游戏：

```bash
docker build -t open-beni .
docker run --rm -p 7860:7860 open-beni
```

打开 `http://localhost:7860`。Hugging Face 会将 7860 端口代理到 Space 的 HTTPS 地址，玩家无需输入端口。设置 `MAX_ROOMS=0` 可关闭多人模式；默认最多 10 个房间。若要构建运行时不使用 CDN 的版本，传入 `--build-arg VITE_ASSET_BASE_URL=`。

# BENI mjlab

[![许可证：MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

基于 [mjlab](https://github.com/mujocolab/mjlab) 和 MuJoCo Warp 的 BENI 机器人平地速度跟踪、跌倒恢复及跳跃任务。

## 推荐环境

- Linux x86_64。
- Python **3.12**。
- 训练推荐至少 8 GB 显存、驱动兼容 CUDA 12.8 的 NVIDIA GPU。小规模运行也可使用 CPU。
- 使用 [uv](https://docs.astral.sh/uv/) 管理 Python 和依赖。

## 快速开始

在仓库根目录安装依赖：

```bash
uv sync --locked
```

## 训练

对 BENI 机器人进行 15,000 次 PPO 迭代训练：

```bash
uv run python -m beni_rl.train Beni-Jump-Flat --env.scene.num-envs 4096 --agent.max-iterations 15000 --agent.seed 42
```

进行小规模训练检查；没有 CUDA 时会自动使用 CPU：

```bash
uv run python -m beni_rl.train Beni-Jump-Flat \
  --env.scene.num-envs 4 --agent.max-iterations 10
```

CPU 运行仍使用相同的 MuJoCo Warp 后端，但比 GPU 训练慢。检查点和 TensorBoard 日志保存在 `logs/rsl_rl/beni_jump_flat/`。使用 `uv run python -m beni_rl.train Beni-Jump-Flat --help` 查看全部训练选项。

### 多 GPU 训练

`python -m beni_rl.train` 会自动使用所有可见的 CUDA 设备，最多八张卡。`--gpu-ids '[0,1]'` 选择 `CUDA_VISIBLE_DEVICES` 中的设备索引；`--gpu-ids None` 选择 CPU。

### 继续训练

从检查点继续进行 PPO 更新：

```bash
uv run --locked python -m beni_rl.train Beni-Jump-Flat \
  --agent.resume True --agent.load-run '<run>' --agent.load-checkpoint model_4000.pt \
  --agent.max-iterations 3000
```

继续训练会恢复训练阶段、采样进度、困难状态成员关系以及学习率上限。

## 评估

在仓库根目录运行 Viser，并在浏览器中播放策略：

```bash
uv run play Beni-Jump-Flat --checkpoint-file "assets/models/policy.pt" --num-envs 1 --viewer viser
```

Viser 就绪后，`play` 会在默认浏览器中自动打开新窗口或标签页。默认地址为 `http://localhost:8080`；如果端口被占用，浏览器会打开 Viser 实际选用的端口。没有可用浏览器时，使用终端打印的 URL。

### 在 Viser 中控制机器人

打开 **Controls → Commands → Twist**，勾选 **Enable**，即可通过滑块控制所选机器人。未勾选时，命令会自动采样。

| 控件 | 范围 | 请求的动作 |
| --- | --- | --- |
| `lin_vel_x` | −2 至 2 m/s | 正值前进，负值后退。 |
| `lin_vel_y` | 固定为 0 | 禁用横向运动。 |
| `ang_vel_z` | −5 至 5 rad/s | 从上往下看，正值左转，负值右转。 |
| `base_pitch (rad)` | −0.1 至 0.1 rad | 请求机身前后俯仰。 |
| `base_roll (rad)` | −0.1 至 0.1 rad | 请求机身左右侧倾。 |
| `base_height (m)` | 0.10 至 0.12 m | 请求机身高度；值越大站姿越高，中点为 0.11 m。 |

例如，将 `lin_vel_x = 0.5`、`ang_vel_z = 0` 设为前进；将 `lin_vel_x = 0`、`ang_vel_z = 1` 设为原地左转。两者都设为 0 且保持 **Enable** 勾选，可请求静止站立。同时设置前进速度和偏航角速度可请求曲线路径。这些都是策略输入命令，实际动作和姿态取决于已训练策略的跟踪能力。

打开 **Controls → Commands → Jump**（文件夹名称会显示固定参考高度），点击 **Request one jump**，让所选机器人跳跃一次。播放时须手动触发跳跃。尝试原地跳跃时，保持 Twist 的 **Enable** 勾选，将两个速度命令和机身倾角都设为 0，并将 `base_height (m) = 0.11`。等 **Measured state** 显示 `ready=True` 再点击。机器人平衡后，也可以尝试边前进边跳跃。训练、播放和 BENI 网页配置均使用固定的 0.18 m 跳跃参考值，含义是相对标称站立高度的底座抬升量，并使用相应的高度命令。检查点设置和上传模型的指纹不会改变这一参考值；实际车轮离地高度取决于训练策略。原始轨迹文件作为归档保留；早期策略须搭配其对应的源码版本才能复现行为。

机器人必须连续 0.2 秒满足平衡与车轮支撑条件，请求才会被接受。**Last request** 会显示 `accepted` 或拒绝原因：`not ready`、`cooldown`、`busy` 或 `disabled`。再次跳跃前，先等机器人恢复平衡，并等 1 秒冷却及上一次跳跃的落地监测都结束。落地后 `busy` 可能继续显示数秒；等待后再次点击即可重新提交请求。

通过 **Simulation → Pause** 暂停仿真；机器人跌倒后可用 **Reset Environment** 重置。暂停期间提交的跳跃请求会在继续仿真后处理。

### 其他播放选项

播放已训练策略，将 `<run>` 替换为保存的运行目录：

```bash
uv run play Beni-Jump-Flat \
  --checkpoint-file "logs/rsl_rl/beni_jump_flat/<run>/model_best.pt" \
  --num-envs 1 \
  --viewer viser
```

不加载检查点，仅查看机器人：

```bash
uv run play Beni-Jump-Flat \
  --agent zero \
  --num-envs 1 \
  --viewer viser
```

## 导出与部署（ONNX + MNN）

在 CPU 上导出检查点：

```bash
uv run --locked --group export python -m beni_rl.headless_export \
  --run_dir "logs/rsl_rl/beni_jump_flat/<run>" \
  --checkpoint model_best.pt
```

省略 `--checkpoint` 时，会选取运行目录中编号最大的 `model_*.pt`。该命令依次完成 `.pt → ONNX → MNN`，并在 `<checkpoint-dir>/exported/<checkpoint-stem>/` 下写入两种部署文件：

| 文件 | 部署用途 |
| --- | --- |
| `policy.onnx` | 供 ONNX Runtime 和浏览器查看器使用的策略。 |
| `policy.mnn` | 部署到机器人上的 MNN 策略。 |

## 浏览器游戏

基于浏览器 MuJoCo + ONNX 的游戏位于 [`web/`](web/README.md)。从完整仓库检出目录构建，要求 Node.js 22.13+，并可通过 `python3` 调用 Python 3.9+；无需安装训练依赖。

```bash
cd web
npm ci
npm run build
npm start
```

打开 `http://localhost:3000/`。Node 服务器监听 `0.0.0.0:3000`，默认开启多人模式；开发时使用 `npm run dev`。单独提供 `web/dist/` 静态文件只支持单人模式。公开部署多人模式时，Node 服务器须与网页同源。

## 仓库结构

主要文件和目录：

```text
.
├── assets/
│   ├── beni_base/
│   │   ├── meshes/                           # 机器人可视网格（OBJ）
│   │   ├── mjcf/beni.xml                     # MuJoCo 机器人模型
│   │   ├── textures/                         # 机器人纹理
│   │   └── urdf/beni.urdf                    # URDF 机器人模型
│   ├── models/
│   │   ├── policy.pt                         # 用于播放的检查点
│   │   ├── policy.onnx                       # ONNX 策略
│   │   ├── policy.mnn                        # MNN 策略
│   │   ├── policy_best.pt                    # 本地最佳检查点
│   │   └── policy_best.onnx                  # 本地最佳 ONNX 策略
│   └── motions/
│       ├── base_jump_h0.18.npz               # 当前跳跃参考轨迹
│       ├── base_jump_h0.25.npz               # 保留的原始跳跃参考轨迹
│       ├── recovery_initial_states_train.npz # 原始恢复状态库
│       ├── recovery_contact_states_train.npz # 按接触状态分层的训练状态
│       └── recovery_contact_states_heldout.npz # 独立的恢复能力评估状态
├── python/beni_rl/
│   ├── __init__.py                           # 任务注册
│   ├── _viser_logging.py                     # Viser WebSocket 日志过滤
│   ├── actuator.py                           # 电机和被动弹簧模型
│   ├── beni_jump_env_cfg.py                  # 跳跃训练及播放环境
│   ├── beni_robot.py                         # 机器人和执行器配置
│   ├── beni_velocity_flat_env_cfg.py         # 速度跟踪环境
│   ├── distributed.py                        # 多 GPU 共享归一化
│   ├── headless_export.py                    # ONNX/MNN 导出器
│   ├── jump.py                               # 跳跃命令、测量及奖励
│   ├── jump_evaluation.py                    # 用于选取最佳检查点的跳跃试验
│   ├── mdp.py                                # 共用观测、奖励及事件
│   ├── play.py                               # 播放入口与浏览器启动
│   ├── recovery.py                           # 恢复奖励和状态还原
│   ├── recovery_bank.py                      # 状态库校验和物理状态分层
│   ├── recovery_curriculum.py                # 恢复采样与课程
│   ├── recovery_evaluation.py                # 独立的运动能力检查
│   ├── recovery_gate.py                      # 跳跃训练前的恢复能力检查
│   ├── recovery_sampling.py                  # 困难恢复状态的自适应采样
│   ├── rl_cfg.py                             # PPO 配置和训练运行器
│   ├── train.py                              # 训练入口
│   ├── velocity_command.py                   # 速度、姿态和云台命令
│   └── wheel_safety.py                       # 车轮超速终止条件与诊断
├── web/
│   ├── app/                                  # 浏览器界面
│   ├── client/                               # 游戏会话与输入处理
│   ├── components/                           # 界面组件
│   ├── game/                                 # 浏览器仿真与渲染
│   ├── hooks/                                # React Hooks
│   ├── lib/                                  # 共用界面工具
│   ├── public/                               # 静态配置和生成的运行时资源
│   ├── scripts/                              # 资源和运行时准备脚本
│   ├── server/                               # 多人游戏服务器
│   ├── shared/                               # 客户端/服务器共用的竞速规则及协议
│   ├── tools/                                # 演示录制及验证工具
│   └── README.md                             # 网页开发说明
├── pyproject.toml                            # Python 包和依赖
├── uv.lock                                   # 锁定的 Python 依赖
├── LICENSE
└── README.md
```

## 许可证

OpenBENI 采用 [MIT 许可证](LICENSE)。包括 [mjlab](https://github.com/mujocolab/mjlab/blob/main/LICENSE) 在内的依赖各自保留其许可证。

## 致谢

本项目基于 [mjlab](https://github.com/mujocolab/mjlab) 和 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 构建。
