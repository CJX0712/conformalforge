"""pytest 根配置（作者：晨星）：确保仓库根在 sys.path，绝对导入可用。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
