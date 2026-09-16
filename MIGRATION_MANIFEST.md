# 迁移来源清单

本文件记录 `uterine-tracking` 从 `wyxwyx357-del/end@a9cf11b95293e395e0a90a1e3340393d782a2d5d` 迁出的追踪代码。

迁移原则：核心算法、阈值、公式和 QC 逻辑不得重新设计；仅允许独立仓库所需的入口、路径、批处理编排和来源记录作为 glue code。患者视频、DICOM、真实标签、真实结果及身份信息不迁移。

## 原样迁移文件

以下文件在本迁移分支中与来源提交的 Git blob SHA 一致，因此内容逐字节一致。

| 新仓库路径 | Git blob SHA | 作用 |
|---|---|---|
| `代码/01_底层算法/peristalsis_pipeline/__init__.py` | `34cf7ea92bae39242cc491f98ce59597e412c00a` | 包初始化 |
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
| `代码/03_实验与历史代码/run_final_sparse_anchor_wall_tracking.py` | `277b1a7e3675b3134605987ed37128525eccff0a` | 310 例批处理调用的五锚帧壁追踪入口 |
| `代码/05_环境与校验/requirements.txt` | `24538552ea80f2fc0952e9878d6ebe06ceb03b6f` | 原运行依赖版本 |
| `测试代码/01_底层算法测试/test_radial_pair_geometry.py` | `cf28871982b1ec19c85903a0616dc246788bb36a` | 径向几何/RSR基础测试 |
| `测试代码/conftest.py` | `c7b9c6c39cab797eae6a46ad9c88fff8c2f33318` | pytest路径初始化 |

## 新增胶水代码

`代码/04_运行入口/run_tracking_only_pipeline.py` 为 `GLUE_ADAPTED`，不是来源仓库文件的逐字节复制。它只负责把原仓库 `代码/03_实验与历史代码/run_new_patient_full_pipeline.py` 中的追踪阶段 01–07 串联起来，并在第 07 阶段后停止。核心 LK、网格、径向点对、P3 纠偏、P3 position QC 和 artifact QC 均调用上表逐字迁移模块。

该入口对应原 310 例新患者批处理中的“无既往人工确认记录”语义：自动候选保存为待人工复核，不能因为自动判定本身直接获得人工排除动作。

## 壁追踪入口核验结果

`代码/03_实验与历史代码/run_final_sparse_anchor_wall_tracking.py` 已恢复为来源提交中的逐字内容，当前目标 blob SHA 与 `end@a9cf11` 完全一致：

`277b1a7e3675b3134605987ed37128525eccff0a`

因此该核心入口现在可以标记为 `EXACT_MIGRATION`。它所包含的五锚帧壁追踪、narrow section、宫底冗余、双向 LK/RSTC 融合及壁追踪有效性规则没有在迁移过程中重写。

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
4. 任何差异必须能追溯到新编排路径，而不是核心算法、阈值、公式或 QC 语义变化。
