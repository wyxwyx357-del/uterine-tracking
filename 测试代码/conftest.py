from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

(PROJECT_ROOT / "输出" / "90_实验与历史输出").mkdir(parents=True, exist_ok=True)
