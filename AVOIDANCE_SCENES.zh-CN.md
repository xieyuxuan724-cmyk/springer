# 可配置场地与压力测试

播放、评估、训练、短基准共用 `scripts/beni_avoidance/common.py` 的 Config 和 `scene_cli.py` 的参数入口。加载检查点后再应用显式 CLI 覆盖；不加参数时沿用检查点配置，旧检查点缺失的新字段按原场景默认值补齐。原检查点文件不会修改。

## 直接运行大场景

前进 10 m、总宽 6 m、16 个障碍槽、实际放置 12 个障碍：

```bash
uv run python scripts/play_avoidance.py \
  --policy logs/rsl_rl/beni_avoidance/2026-10-16_11-32-41/model_final.pt \
  --target-distance 10 --field-width 6 \
  --max-obstacles 16 --obstacles 12 \
  --speed 0.5 --seed 9000200
```

数字小键盘 `0` 换地图、`.` 重置，鼠标滚轮调远近。红线表示机器人中心的左右/后方任务界限，绿线表示终点；不是实体墙。当前仍限制目标速度至 .5 m/s，本次只扩展场地和障碍容量，没有扩展底层高速能力。

## 统一参数表

| CLI 参数 | Config 字段 | 默认值 | 含义 |
|---|---|---:|---|
| `--target-distance` | target_distance | 3 | 起点至终点的目标方向净进度，m |
| `--field-width` | half_width × 2 | 3 | 场地总宽，m；CLI 传完整宽度 |
| `--back-margin` | back_margin | .4 | 起点后方可用距离，m |
| `--max-obstacles` | max_obstacles | 8 | 模型障碍槽数量，支持 8–128；不是自动保证所有密度都可放置 |
| `--obstacle-start` | obstacle_start | .85 | 障碍中心最靠近起点的纵向位置，m |
| `--obstacle-end-margin` | obstacle_end_margin | .25 | 障碍中心距终点的最小纵向留白，m |
| `--obstacle-side-margin` | obstacle_side_margin | .45 | 障碍中心距名义侧界的最小横向留白，m |
| `--route-resolution` | route_resolution | .1 | 通路栅格步长，m，范围 .02–.2 |
| `--route-padding` | route_padding | .05 | 通路检查中，在机器人保守半径之外额外膨胀的余量，m |
| `--timeout-factor` | timeout_factor | 2 | 非零速度任务时间系数 |
| `--timeout-extra` | timeout_extra | 5 | 非零速度任务额外时间，秒 |
| `--stop-timeout` | stop_timeout | 5 | 独立零速度任务时限，秒 |

机器人安全半径仍为 .19 m。例如宽6 m的场地，机器人中心横向范围是 ±2.81 m；纵向范围为 −back_margin 至 target_distance。

障碍中心均匀采样区域：

```text
x ∈ [obstacle_start, target_distance - obstacle_end_margin]
y ∈ [-field_width/2 + obstacle_side_margin,
      field_width/2 - obstacle_side_margin]
```

障碍尺寸沿用8档箱体/圆柱目录，更多槽重复使用这些尺寸，不改变机器人模型或策略输入维度。未启用的障碍移至任务区域外并分散停放。两个物理后端均使用实际容量建立模型、mocap、射线数组和接触分类。

通路检查自动覆盖当前场地，从起点附近检查至终点之后两个网格步长；使用相同宽度、安全半径和障碍目录。障碍不重叠且存在栅格通路才接受布局；无路布局重采样，达到上限会明确失败。栅格通路不保证现有策略一定能动态通过。大场地/细网格超过200万个格点会拒绝，避免不受控的检查开销。

非零速度超时统一为：

```text
timeout = timeout_factor * target_distance / target_speed + timeout_extra
```

例如10 m、.5 m/s、默认参数，超时为45秒；初始 .5 秒稳定期仍不计入任务时间。原来的3 m、.5 m/s仍为17秒。

## 批量压力测试

```bash
uv run python scripts/evaluate_avoidance.py \
  --policy logs/rsl_rl/beni_avoidance/2026-10-16_11-32-41/model_final.pt \
  --backend native --num-envs 4 --episodes 30 --seed-base 9000200 \
  --target-distance 10 --field-width 6 \
  --max-obstacles 16 --obstacles 12 \
  --output logs/avoidance_validation/large_scene_30.json
```

评估仍按 .1/.3/.5 m/s 循环。`--obstacles` 固定实际障碍数；不传时从3至 max_obstacles 根据种子选择。使用独立诊断种子段，避免把压力测试用于反复查看最终测试集。

更改标准场地参数或固定评估障碍数后，报告为 `stress_test=true`、`formal_protocol=false`、`acceptance_met=null`。它仍报告成功率与失败原因，但不套用原场景98%成绩或标准90%验收标签。JSON包含有效 Config、CLI 覆盖项、逐场障碍数和种子。若写入 TensorBoard，使用 `Stress/<backend>/...` 标签；不同场地请用不同日志目录，避免曲线混合。

现有模型可以直接加载做压力测试，网络仍为147维，但边界归一化会按新场地计算，轨迹长度、障碍密度和时限也改变；不能把新场景成功率视为已训练或已保证的能力。

## 在新场地训练与短基准

训练入口支持相同场地参数，并将其写入 manifest 和检查点。示例（不会自动执行）：

```bash
uv run python scripts/train_avoidance.py \
  --backend warp --num-envs 32 --iterations 300 --minutes 30 \
  --resume logs/rsl_rl/beni_avoidance/2026-10-16_11-32-41/model_final.pt \
  --target-distance 10 --field-width 6 --max-obstacles 16 \
  --validation-zero-episodes 0
```

课程仍为：空场、1–2障碍、3至最大容量。续训延续已保存的课程步数，不会自动回到空场阶段。训练不接受固定 `--obstacles`；使用容量控制课程上限。训练和周期验证都使用有效场地配置。更多障碍会改变 GPU 内存、碰撞开销和布局生成耗时，不应直接把此前4096环境的吞吐套用到大场景。

```bash
uv run python scripts/check_avoidance.py --backend warp --num-envs 2 --steps 10 \
  --target-distance 10 --field-width 6 --max-obstacles 16 --count 12
```

## 本轮检查

- 原有12项测试、新增4项场景测试通过。
- 100个种子 × 0/4/8障碍，共300组默认布局与修改前完全一致。
- 原生模型加载、12障碍大场景回放/截图通过。
- GPU双环境、16槽、10步短基准通过。
- 10 m × 6 m、12障碍、3个诊断场景全部完成；仅为链路检查，不是大场景成功率验收。记录在 `logs/avoidance_validation/scene_stress_smoke.json`。
- 未启动大规模训练，历史检查点保留。

### 高密度布局生成修正

实际障碍数量超过8时，使用箱体/圆柱的实际二维几何距离检查8 cm间隔，避免旧包围圆检查过度占用可放置空间。仍要求不重叠且通过原来的膨胀栅格通路检查；没有自动减少障碍、缩小障碍或取消通路约束。最多8个障碍继续使用原算法，保持历史种子布局一致。

已验证 `--target-distance 3 --field-width 6 --max-obstacles 64 --obstacles 32 --seed 9000200` 能生成并渲染；连续10个种子通过可重现性和通路测试。过高密度仍可能采样失败，届时会报告实际中心采样区域、最多放置数量及通路拒绝情况，并提示增大场地或减少实际障碍。播放器启动时给出简洁错误，交互换图失败时保留旧场景，可按小键盘0尝试下一种子。
