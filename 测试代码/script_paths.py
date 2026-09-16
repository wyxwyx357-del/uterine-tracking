from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = PROJECT_ROOT / "代码" / "02_已并入主流程"
EXPERIMENT_ROOT = PROJECT_ROOT / "代码" / "03_实验与历史代码"

MAIN_GROUP_BY_SCRIPT = {
    "run_step1_4_huang_radial_pair_5case.py": "01_P3与径向追踪",
    "run_step1_4_huang_radial_pair_pilot.py": "01_P3与径向追踪",
    "run_step1_4_radial_common_motion_diagnostic.py": "01_P3与径向追踪",
    "run_step1_4_p3_image_boundary_correction_v2_5case.py": "01_P3与径向追踪",
    "run_step1_4_p3_anatomical_position_qc_v1_5case.py": "01_P3与径向追踪",
    "run_step1_4_artifact_qc_v2_5case.py": "02_伪影质控与人工复核",
    "run_step1_4_pair_quality_shadow_qc_5case.py": "02_伪影质控与人工复核",
    "prepare_218_candidate_manual_review.py": "02_伪影质控与人工复核",
    "export_218_manual_review.py": "02_伪影质控与人工复核",
    "apply_218_manual_review.py": "02_伪影质控与人工复核",
    "run_step1_4_rsr_local_deformation_review_v1_5case.py": "03_RSR与信号处理",
    "run_step1_5f_rsr_fourway_comparison_5case.py": "03_RSR与信号处理",
    "run_step1_5f_rsr_processing_fair_comparison_5case.py": "03_RSR与信号处理",
    "run_step1_4_lk_parameter_ablation_real_texture.py": "03_RSR与信号处理",
    "run_step1_5f_anatomical_deformation_5case.py": "04_形变事件与方向审查",
    "run_step1_6a_direction_null_5case.py": "04_形变事件与方向审查",
    "run_step1_6a_real_texture_semisynthetic_propagation.py": "04_形变事件与方向审查",
    "backfill_semisynthetic_direction_null_details.py": "04_形变事件与方向审查",
    "audit_literature_aligned_outputs.py": "05_输出审查与发布",
    "build_organized_result_view.py": "05_输出审查与发布",
}


def script_path(filename: str) -> Path:
    group = MAIN_GROUP_BY_SCRIPT.get(filename)
    if group is None:
        return EXPERIMENT_ROOT / filename
    return MAIN_ROOT / group / filename
