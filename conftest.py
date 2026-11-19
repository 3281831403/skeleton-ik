"""把仓库根加入 sys.path，使 `import anim` 在任意工作目录下可用。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
