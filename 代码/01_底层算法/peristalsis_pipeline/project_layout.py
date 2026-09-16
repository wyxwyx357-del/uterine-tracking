from __future__ import annotations

from pathlib import Path


OUTPUT_GROUP_BY_NAME = {
    "step1_4_p3_assisted_radial_tracking_v1_5case": "01_径向LK追踪",
    "step1_4_p3_image_boundary_correction_v2_5case": "02_P3图像边界纠偏",
    "step1_4_p3_anatomical_position_qc_v1_image_corrected_5case": "03_P3位置质控",
    "step1_4_p3_assisted_artifact_qc_v2_1_5case": "04_伪影质控",
    "step1_4_p3_image_corrected_artifact_qc_v2_1_5case": "04_伪影质控",
    "step1_4_p3_image_corrected_rsr_local_deformation_review_v1_5case": "05_RSR局部形变与事件",
    "step1_4_pair_quality_shadow_qc_v1_5case": "06_影子质控与四路RSR_敏感性",
    "step1_5f_rsr_fourway_v1_5case": "06_影子质控与四路RSR_敏感性",
    "step1_5f_rsr_processing_fair_comparison_v1_5case": "06_影子质控与四路RSR_敏感性",
    "step1_5f_anatomical_deformation_v1_5case": "07_解剖形变_敏感性",
    "step1_6a_direction_null_v1_5case": "08_方向零假设_探索性",
    "明早_218帧人工复核包": "09_人工复核待处理结果",
    "step1_4_artifact_qc_v2_1_synthetic": "10_模拟与验证结果",
    "step1_4_lk_parameter_ablation_v1": "10_模拟与验证结果",
    "step1_4_p3_assisted_artifact_qc_v2_1_image_texture_validation": "10_模拟与验证结果",
    "p3_hidden_middle_holdout_h1_v1": "10_模拟与验证结果",
    "step1_4_weak_rsr_and_propagation_synthetic_validation_v1": "10_模拟与验证结果",
    "step1_6a_real_texture_semisynthetic_v1": "10_模拟与验证结果",
    "step1_4_p3_guided_tracklet_pilot_v1_5case": "90_实验与历史输出",
    "step1_4_p3_guided_tracklet_pilot_v1_smoke": "90_实验与历史输出",
}


def output_dir(project_root: Path, logical_name: str) -> Path:
    """Return the physical directory for a logical output name.

    Unknown names are isolated under the experimental/history group so an
    experimental script cannot silently publish into a formal result group.
    """

    group = OUTPUT_GROUP_BY_NAME.get(logical_name, "90_实验与历史输出")
    return project_root / "输出" / group / logical_name


def all_output_roots(project_root: Path) -> list[Path]:
    return [
        project_root / "输出" / group
        for group in sorted(set(OUTPUT_GROUP_BY_NAME.values()) | {"90_实验与历史输出"})
    ]


def input_video_dir(project_root: Path) -> Path:
    return project_root / "输入" / "01_原始视频"


def middle_label_dir(project_root: Path) -> Path:
    return project_root / "输入" / "02_人工标签" / "01_中帧解剖标签"


def anchor_label_dir(project_root: Path) -> Path:
    return project_root / "输入" / "02_人工标签" / "02_P3五锚点标签"


def manual_label_dir(project_root: Path) -> Path:
    return project_root / "输入" / "02_人工标签" / "03_质控人工记录"


def upstream_anatomy_dir(project_root: Path) -> Path:
    return project_root / "输入" / "03_上游解剖节点"


def frozen_baseline_dir(project_root: Path) -> Path:
    return project_root / "输入" / "04_冻结基线"
