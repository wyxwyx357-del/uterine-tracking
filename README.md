# uterine-tracking

本仓库用于保存从原始项目 `wyxwyx357-del/end` 中独立迁移出的子宫短视频 P3 壁追踪、径向 LK 测量与追踪质量控制流程。

## 迁移基线

- 来源仓库：`wyxwyx357-del/end`
- 来源提交：`a9cf11b95293e395e0a90a1e3340393d782a2d5d`
- 迁移原则：核心算法、阈值、公式与 QC 逻辑直接来自上述提交，不重新实现；仅新增独立运行所需的编排与路径胶水代码。
- 核心迁移文件保留来源提交中的 Git blob 内容；详见 `MIGRATION_MANIFEST.md`。

## 本仓库范围

本仓库只覆盖以下追踪链：

1. 五锚帧人工前壁、后壁、宫颈点、宫底点；
2. 宫颈→宫底统一方向与 10 个候选 section；
3. 窄 section 节点规则、宫底冗余 section 处理；
4. 前向/反向逐帧 LK、RSTC 融合、邻域与拓扑约束；
5. 逐帧 P3 前后壁轨迹；
6. 基于 P3 建立固定 21 analysis-px Inner–Outer 径向点对；
7. 相邻帧 pairwise LK 径向测量及 validity/PCC/FB/curve-distance 等质量量；
8. 基础 artifact QC；
9. 图像支持的 P3 边界纠偏候选及原位置 vs 候选位置 LK 质量门控；
10. 接受纠偏后重新建立径向点对并重新运行 LK；
11. P3 anatomical position QC；
12. 纠偏后 artifact QC。

流程到此结束。患者级 F01–F20/V2 特征、传播/方向分析、DICOM 曲率、AUC 建模、深度学习等不属于本仓库。

## 核心运行入口

独立批处理入口：

```bash
python "代码/04_运行入口/run_tracking_only_pipeline.py" \
  --external-label-dir <五锚点标签目录> \
  --output-dir <新输出目录> \
  --cases CASE_001 CASE_002 \
  --resize-factor 0.5 \
  --sections 10 \
  --skip-videos
```

输入目录需要包含每个病例的：

- `<case_id>__five_anchor_manifest.csv`；
- manifest 指向的 5 组 PNG + LabelMe JSON；
- manifest 中 `source_video` 指向可读取的 MP4。

默认参数保持原 310 例新患者批处理使用的主要设置：`resize_factor=0.5`、`sections=10`、径向点对固定偏移 `21 analysis px`。这些是当前项目工程设置，不应表述为临床阈值或已外部验证的参数。

## 输出阶段

每个病例输出以下 7 个阶段目录：

```text
01_wall_tracking_reference
02_radial_tracking_raw
03_base_artifact_qc
04_p3_boundary_correction
05_radial_tracking_corrected
06_p3_position_qc
07_artifact_qc
```

当前独立入口与原 310 例新患者批处理一致地采用“无人工确认记录”语义：自动候选保留为待复核，不会因为自动候选本身直接执行人工排除动作。

## 重要方法学边界

- 壁追踪中的 `feature_valid` / `tracking_measurement_valid` 与径向 LK 的 `radial_pair_valid` 是不同阶段的有效性掩膜。
- 径向 LK 每个相邻帧对都重新从前一帧 P3 参考位置出发；它不证明跨长时间的材料点身份连续性。
- 径向 FB error 在该分支主要是质量诊断，不是壁追踪阶段那个 `FB <= 2 px` admission gate。
- P3 边界纠偏需要图像证据与 tracking quality gate；通过纠偏后会重新建立 Inner–Outer 点对并重新跑 LK，而不是平移旧测量结果。
- P3 anatomical position QC 的自动候选只用于复核定位，不自动修改 P3。
- artifact QC 的自动 3 级候选在未人工确认时不会直接作为正式删除动作。

## 环境

原项目冻结环境依赖保存在：

```text
代码/05_环境与校验/requirements.txt
```

## 验证状态

迁移分支保留了部分底层测试，并记录了逐文件来源 SHA。当前迁移工作不包含患者数据，也没有在 GitHub 端批量重跑 310 例；合并前应在本地使用去标识化/允许使用的数据做至少一例端到端 smoke test，并与原仓库同输入结果进行数组级比较。
