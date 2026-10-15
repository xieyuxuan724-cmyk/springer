# BENI 原生 MuJoCo 策略播放器

入口：`scripts/play_native.py`。直接调用 `mujoco.viewer.launch_passive`，不启动网页、Viser、mjlab、MuJoCo Warp 或训练环境。使用 CPU 推理和原生 MuJoCo 物理仿真。

## 启动

在仓库根目录执行，复用现有环境：

```bash
uv run python scripts/play_native.py
```

默认读取 `assets/models/policy.pt`。播放自己的训练检查点：

```bash
uv run python scripts/play_native.py \
  --policy logs/rsl_rl/beni_jump_flat/<运行目录>/model_15000.pt
```

实际文件名以训练保存结果为准，也可选择 `model_best.pt`。不要使用 `python -m beni_rl...` 启动这个播放器；`beni_rl` 的包初始化会导入训练框架。

这里的 `uv run` 只是复用项目依赖环境，播放器本身不导入 mjlab。也可直接运行 `.venv/bin/python scripts/play_native.py`。若需要完全不安装训练框架的独立环境：

```bash
python3.12 -m venv /tmp/beni-native-env
/tmp/beni-native-env/bin/pip install 'mujoco==3.11.0' numpy torch
/tmp/beni-native-env/bin/python scripts/play_native.py
```

以上原生 MuJoCo 版本已在本仓库环境测试。依赖安装仍需网络和平台兼容的 PyTorch 包。

ONNX 路径只需 MuJoCo、NumPy、ONNX Runtime，不需要 PyTorch：

```bash
# 现有项目环境中的 onnxruntime 在 dev 依赖组
uv sync --locked --group dev
uv run --group dev python scripts/play_native.py --policy assets/models/policy.onnx
```

## 支持的策略接口

- 当前仓库的 40 维观测、7 维动作策略，包括 Velocity 和 Jump 的当前观测布局。
- `.pt`：读取 `actor_state_dict`；单层 GRU-512，ELU MLP 512/256/128，恢复观测归一化及循环状态，使用确定性动作均值。采用 `weights_only=True` 加载。
- `.onnx`：两个 float32 输入 `[1,40]`、`[1,1,512]`，两个输出 `[1,7]`、`[1,1,512]`。网络内部负责归一化。
- 不支持旧版 42 维策略、LSTM、MNN。维度不兼容时直接报错。
- PT 的激活函数和归一化 epsilon 遵循当前仓库的 ELU / 0.01 约定；如果改变了训练架构或归一化算法，应导出符合上述接口的 ONNX，不要仅依赖相同权重形状。
- 速度任务策略可以播放运动，但没有学过跳跃的权重不应触发跳跃。

## 操作

点击原生窗口使其获得键盘焦点。按键采用增量命令，松开按键不会自动归零；终端显示当前命令及每秒状态。

| 按键 | 操作 |
| --- | --- |
| 数字小键盘 8 / 2 | 前进速度增加 / 减少 0.1 m/s |
| 数字小键盘 4 / 6 | 偏航速度增加 / 减少 0.2 rad/s |
| 数字小键盘 5 | 速度命令清零 |
| 数字小键盘 0 | 请求跳跃 |
| 数字小键盘小数点 | 重置机器人与策略状态 |
| 数字小键盘 Enter | 暂停 / 继续 |
| 窗口关闭按钮 / Ctrl+Q | 关闭窗口 |

机器人仅绑定独立数字小键盘键码；不是键盘上方的数字键。无数字小键盘时，可用下方命令行参数指定运动和定时跳跃。
字母、F1–F12 和其他按键全部保留给 MuJoCo 原生界面。特别是 W 会切换线框，F6 会切换坐标系，F7 会切换标签（当前安装的 MuJoCo 3.11.0 已实测）；它们不再控制机器人。原生 `key_callback` 不能阻止内置快捷键执行，因此不要重新将这些按键绑定为驾驶控制。

使用数字小键盘 Enter 暂停，不依赖原生界面的暂停状态。鼠标按 MuJoCo 原生方式旋转、缩放视角，相机注视点跟随机身。窗口叠加当前速度、就绪和暂停状态。暂停时可以登记跳跃请求，恢复后执行。

地面使用深浅棋盘纹理（每格约 0.25 m），物理碰撞和摩擦参数不变。碰撞平面无限；可见地面为 40×40 m 的区域，每个控制周期按 0.5 m 的完整纹理周期重定位到机器人附近，避免持续行驶后离开可见区域。整数周期平移保持方格在世界坐标中对齐。镜头过度拉远仍可能看到显示区域边缘。

初始指令示例：

```bash
uv run python scripts/play_native.py --vx 0.5
uv run python scripts/play_native.py --wz 0.5
uv run python scripts/play_native.py --head-yaw 0.5 --head-roll 0.03
```

`--head-yaw` 是策略的云台目标；`--head-roll` 是独立 PD 的局部关节目标，范围 ±0.05 rad，不能视为世界水平姿态控制。修改侧倾目标可能影响原策略表现。

## 无窗口验证与日志

```bash
uv run python scripts/play_native.py --headless --duration 10
uv run python scripts/play_native.py --headless --duration 10 --vx 0.5
uv run python scripts/play_native.py --headless --duration 10 --jump-at 3 \
  --log /tmp/beni-native-jump.json
```

`--jump-at` 仅在指定仿真时间请求一次，不会绕过就绪判断，也不会自动重试。`--duration` 使用仿真时间；不传时，窗口模式运行到关闭，无窗口模式默认运行 10 秒。无窗口模式尽快运行，不按墙钟节流。

JSON 记录末状态、最高机身绝对高度、单步就绪条件占比、跳跃请求结果以及是否检测到持续至少 50 ms 的腾空。最高机身高度不是相对起跳抬升量；初始机身高度为 0.14 m。就绪占比并非任务成功率。

## 与训练环境的对应关系及边界

- 读取原 MJCF 与网格，添加平地和光照，不覆盖源模型。
- 从 `beni_robot.py` 静态读取名义电机参数，不执行该模块。
- 将编码器参考零位设为零，初始髋角 0.3 rad、膝角 0.5 rad。
- 物理步长 0.002 s，每 5 步推理一次；观测顺序、默认角度偏置、动作缩放、膝部弹簧和力矩限制按当前代码构建。
- 延迟默认固定为 2 个物理步（4 ms），可用 `--delay-steps 2..5` 调整。它不复现训练中的随机延迟采样。
- 每个控制周期结束后调用 `mj_forward` 更新传感器，避免使用滞后一子步的观测。
- 地面和电机采用名义参数，没有训练域随机化、噪声、恢复课程或训练终止机制。
- 跳跃输入为 `[启用, 帧号/40, 0.18, 0]`，持续 41 帧。独立状态机提供连续 0.2 s 就绪、偏航稳定准备、1 s 冷却、腾空/落地检测及超时；不包含 mjlab 完整奖励、方向保护与评估状态机，也不会自动恢复训练目录中的 YAML 跳跃配置。
- 原生 MuJoCo 与 Warp 的求解路径并非逐帧等价。本工具适合策略展示和独立推理调试，不能直接替代原训练评估协议。

本次在附带 `policy.pt` 上的 10 秒测试：零速度指令下保持约 0.11 m 机身高度，但水平漂移约 0.64 m；0.5 m/s 前进指令下末位置约 `(5.15,-0.87,0.108)` m；3 秒请求跳跃后检测到持续腾空，最高机身高度约 0.256 m，结束时恢复约 0.111 m 高度。跳跃后有约 2 m 水平位移，不能据此声称实现精确原地跳跃或完整的落地质量验收。

## 故障排查

- GLFW / DISPLAY 错误：原生窗口要求可用桌面及 OpenGL；远程终端需要图形转发，纯服务器先用 `--headless`。
- macOS：MuJoCo 原生被动窗口通常需要使用同一环境的 `mjpython scripts/play_native.py`。
- 动作异常：先恢复默认头部目标、延迟和名义模型，确认使用匹配的 40 维权重，再比较 ONNX 和 PT。
- CPU 推理赶不上实时：先用单线程设置和单机器人；窗口按每个 10 ms 控制周期同步，慢机器只会降低墙钟速度，不会跳过物理步。

## 开发验证

```bash
.venv/bin/python -m unittest discover -s scripts -p test_native_play.py -v
```

包含 PT/ONNX 循环推理对照、隐藏状态重置、无训练模块导入、模型自由度与零位、动作延迟、力矩和跳跃观测接口测试。ONNX Runtime 未安装时仅跳过 ONNX 对照。
