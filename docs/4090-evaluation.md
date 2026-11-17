# RTX 4090 评测：基线与性能验收

2026-11-17 状态：候选方案，尚未通过 Sim 5 原生倒水与闭环性能验收。
不要把下面的实验配置作为已验证的最佳实践。

## 已确认的 PR #26 基线

[PR #26](https://github.com/xstar-org/RoboDojo/pull/26) 当前 head 为
`44b5ecf46e6727c1a312995e2060c0c2ec17084b`，base 为 `xstar-main`，依赖 #23。
其 `independent_isolated` 逐环境隐藏邻居几何、停用邻居灯，
`isolated_owner_products_only: true` 只更新当前环境的 product。
原实现拒绝在 Sim 5 启用该后端。

N56 job 1805350 的三份 camera backend 源文件与 PR #26 head **逐字一致**。
固定动作带 SHA256：
`8d01cbfdf9f66e43cb9344faff3360f1fdcad95bf41aaee41746fb85e951f5e8`。
原始目录：`/data/run01/scxm005/robodojo-eval/render-throughput-benchmark-20261112`。
Isaac Sim 6.0.0.1 / Lab 2.3.2，640×480 三相机、PT64、两次 delivery，
96 个动作，每个动作后采图；不含策略推理，关闭视频。
探针还额外启用了头相机深度、固定目标世界原点与 bottle rigid intervention；
因此这些数字不是未经干预的官方评测吞吐。

| 路径 | 动作中位数 | 采图中位数 | 总步骤中位数 | 96 步等长回放 episode/h |
| --- | ---: | ---: | ---: | ---: |
| 1-env tiled | 0.182 s | 2.283 s | 2.466 s | 15.229 |
| 4-env tiled 两次重复 | 0.455–0.466 s | 8.760–8.785 s | 9.218–9.246 s | 16.319–16.362 |
| 4-env isolated + owner-only 两次重复 | 0.489–0.500 s | 9.721–9.766 s | 10.200–10.263 s | 14.659–14.746 |

episode/h 使用 **实际总耗时** 换算，不能用步骤中位数代替总耗时。
正确版相对单环境吞吐为 0.963–0.968 倍，未实现多环境加速。
4-env 正确版的动作均值占总耗时约 4.8%，仅优化物理/IK 无法获得显著收益。
PR #26 的 2.75 → 1.20 秒属于隔离路径内部微基准。

## Sim 5 候选改动

目标固定为 Isaac Sim 5.1.0 / Lab 2.3.2。官方测试 Linux 驱动为
[580.65.06](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。
N56 早先节点读数为 580.82.07，已做过 Sim 5.1 合成场景启动；
每次分配仍需重新记录驱动、CUDA/Vulkan PCI 映射和实际启动结果。
`CUDA_VISIBLE_DEVICES` 不能替代 Vulkan GPU 的 PCI 匹配。

本候选移除隔离路径的 Sim 6 版本限制；底层使用 USD session layer 和
Hydra product 更新接口，没有新增 Sim 6 专属调用。
这仅开放测试入口，尚无 Sim 5 原生倒水的正确性结论。

```yaml
# env_cfg/camera/<camera>.yml；仅用于隔离实验配置
render_product_backend: independent_isolated
isolated_owner_products_only: true
multi_env_capture_render_passes: 1
```

`multi_env_capture_render_passes` 默认为 **2**，保持原有渲染次数。
候选 1-pass 用于检查是否能省去冗余帧；dispatcher delivery 只能证明帧已交付，
不能证明 PT/去噪收敛或灯光切换后的像素等价。
单环境继续使用 observation 的 `capture_render_passes`，不受这个新键影响。
缺少 zero-delay 设置时，原同步器仍自动多等一次 delivery。
dispatcher 停滞超时现在抛异常，避免把旧缓冲区当作当前图像继续评测。

## 验收顺序与退出条件

1. 在同一 Sim 5.1 版本、相同实际渲染设置下重新跑 1-env 基线，
   不能拿 Sim 6 的单环境速度作为 Sim 5 候选的分母。读取实际 spp/totalSpp、
   AA、去噪、分辨率、深度 annotator、相机位姿与物理/粒子初始状态。
2. 对比 4-env 隔离的 2-pass 与 1-pass，固定 layout 世界坐标。
   覆盖全部槽位、三路相机、0/32/64/96 步、正逆采集顺序和恢复。
   同时保存 1-env 的 1-pass/2-pass 噪声对照；先约定误差阈值，
   以已有 Sim 6 的 0.019–0.026 MAE 为参考，不能只检查非黑图。
   1-pass 如果引入帧滞后、图像残差或恢复失败，就淘汰该方案。
3. 在同一张卡串行跑各配置，至少两轮，分别记录动作、采图、总耗时、
   GPU 利用率与显存。建议的首轮回放门槛是相对同版本 1-env **至少 1.5 倍**，
   目标 2 倍；这是本轮提出的验收门槛，不是已实现的性能。
4. 用同权重、seed/layout 和策略配置跑配对闭环，包含推理、视频与 reset，
   汇报成功率、进度分、唯一 layout 完成数/小时及整批 wall time。
   相机正确性、回放速度、闭环成绩都通过，才可提升为推荐配置。

回放门槛工具使用已完成目录中的 `benchmark.json` 和 `runtime-profile.json`：

```bash
python scripts/compare_replay_throughput.py runs/sim51-one runs/sim51-four --min-speedup 1.5
```

退出码 0 表示回放吞吐达标，1 表示低于门槛，2 表示数据不完整或不可比。
工具拒绝不同 runtime、不同动作带、不同实际 render profile、失败/未完成的记录。
它不替代相机正确性、相同硬件/场景配置检查或闭环验收。

如果单卡 PT64 已接近像素计算上限，增加环境数并不保证线性加速。
后续应测独立 1-env 进程/卡的集群横向扩展及集中策略推理，
同卡多进程需单独测显存和吞吐；现有静态方块结果只支持图像控制。
不要用减少 spp、换渲染器或重复同一 layout 的 load test 冒充等价评测提速。

## 当前执行阻塞

2026-11-17，N56 用户关联 `MaxSubmitJobs=0`，无运行作业。
本轮单卡 8 CPU、3 小时申请被 `AssocMaxSubmitJobLimit` 拒绝，没有创建 GPU 作业。
在恢复提交额度并完成上述实验前，本候选 PR 应保持 Draft，不能声称已保障速度。
