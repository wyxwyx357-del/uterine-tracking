# 迁移来源清单

本文件记录 `uterine-tracking` 从 `wyxwyx357-del/end@a9cf11b95293e395e0a90a1e3340393d782a2d5d` 迁出的追踪代码。

## 原样迁移文件

以下文件在本迁移分支中保留了来源提交中的 Git blob 内容；相同 SHA 表示内容逐字节一致。

| 新仓库路径 | Git blob SHA | 作用 |
|---|---|---|
| `代码/01_底层算法/peristalsis_pipeline/tracking_lk.py` | `1448a4651f3a3a534b2ea3c683360acf71a0e1d9` | 低层 LK、视频读取、FB/位移门限 |
| `代码/01_底层算法/peristalsis_pipeline/tracking_mesh.py` | `8b0ad51c3701b536b7c2102e9ec1f727bc7ac0ae` | 网格邻域、拓扑检查/修复 |
| `代码/01_底层算法/peristalsis_pipeline/radial_pair_geometry.py` | `91c91ab674d384270c342b63dea3ae3edd114b35` | Inner–Outer 径向几何与径向长度变化 |
| `代码/01_底层算法/peristalsis_pipeline/tracking_huang_fusion.py` | `ca3af56a3061936f2890426f373dd3cbbe2ea829` | 全局刚体参考及辅助函数 |
| `代码/01_底层算法/peristalsis_pipeline/tracking_huang_wall_fusion.py` | `9e85c827b8460285b7add7dfd9bb67a13602dcba` | 相邻帧 wall LK / PCC / FB / 曲线距离 |
| `代码/01_底层算法/peristalsis_pipeline/tracking_huang_radial_pairs.py` | `f7a712a93c59217c5245301bb75e69472b248b57` | P3 引导径向点对相邻帧 LK |
| `代码/01_底层算法/peristalsis_pipeline/p3_image_boundary_correction.py` | `1acfb6f709f2a6a0680d4363c5fad61c0bb28365` | 图像边界证据、P3 纠偏候选、LK 质量门控 |
| `代码/01_底层算法/peristalsis_pipeline/p3_anatomical_position_qc.py` | `07fef88f32da191462bc87b90b587f51255faf59` | P3 局部位置 QC |
| `代码/01_底层算法/peristalsis_pipeline/tracking_artifact_qc_v2.py` | `9f85bf9c57202ed7f8acaa833d1d79e88c4bca9e` | 多证据 artifact QC v2.1 |
| `代码/01_底层算法/peristalsis_pipeline/project_layout.py` | `63e98a85a9fc7d3e0ce9afd95dd20761cbd9c2f6` | 原项目输出路径映射兼容层 |
| `代码/02_已并入主流程/01_P3与径向追踪/run_step1_4_huang_radial_pair_pilot.py` | `0e23733195d2b408de6e0da75603047d93ff765e` | 原径向追踪运行/复核脚本 |
| `代码/02_已并入主流程/01_P3与径向追踪/run_step1_4_p3_image_boundary_correction_v2_5case.py` | `9bf7528e1e5a45714d7ce2baa13529e91b2c343d` | 原 P3 图像边界纠偏运行脚本 |
| `代码/02_已并入主流程/01_P3与径向追踪/run_step1_4_p3_anatomical_position_qc_v1_5case.py` | `100143fed5cadba52dc8d3460cae101058abce3f` | 原 P3 anatomical position QC 脚本 |
| `代码/02_已并入主流程/02_伪影质控与人工复核/run_step1_4_artifact_qc_v2_5case.py` | `330637ac4276b1d670d25b7d934b68113ad5f98a` | 原 artifact QC v2.1 运行脚本 |
| `代码/03_实验与历史代码/run_final_sparse_anchor_wall_tracking.py` | `c03421d9f54b0fb20dd77092fe406b6a19008b43` | 310 例实际批处理调用的五锚帧壁追踪入口 |

## 新增胶水代码

`代码/04_运行入口/run_tracking_only_pipeline.py` 是新仓库新增的独立编排入口，不是来源仓库文件的逐字节复制。它的目的仅是：

- 把原 15 阶段新患者流程裁剪为追踪相关的前 7 阶段；
- 去掉 DICOM 曲率、患者级特征、RSR 后处理、传播/方向、建模依赖；
- 保持核心追踪算法和阈值来自上述原样迁移模块；
- 使用与原 310 例新患者批处理相同的“自动候选待人工复核、无人工确认行默认不执行人工排除”语义。

## 明确未迁移范围

以下内容不是本仓库目标，因此不应被误认为遗漏：

- F01–F20 / V2 患者级特征提取；
- DICOM 曲率物理标定；
- bandpass/事件/四路 RSR 正式分析链；
- 传播方向与 null 分析；
- 临床特征整合、Elastic Net、AUC；
- 冻结预训练编码器、ResNet/R3D 等深度学习流程。

## 合并前验证要求

在将迁移分支合并到 `main` 前，至少完成：

1. `pytest` 跑通本仓库已迁移的底层测试；
2. 使用同一去标识化病例、同一五锚帧 JSON/PNG、同一 MP4，分别运行原仓库和新仓库；
3. 对 wall tracking NPZ、raw radial NPZ、corrected radial NPZ、P3 position QC NPZ、final artifact QC NPZ 做关键数组逐元素/容差比较；
4. 任何差异必须能追溯到新编排路径，而不是核心算法或阈值变化。
