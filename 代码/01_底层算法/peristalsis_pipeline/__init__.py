"""Independent peristalsis analysis module.

This package is intentionally separated from the static PyRadiomics v2
pipeline. It audits and, when data are available, models dynamic peristalsis
features without modifying existing static radiomics outputs.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "results" / "peristalsis_analysis"
LOG_DIR = PROJECT_ROOT / "logs" / "peristalsis"

