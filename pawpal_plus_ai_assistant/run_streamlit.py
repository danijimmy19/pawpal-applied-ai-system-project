from pathlib import Path
import runpy
import sys

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

runpy.run_path(str(SRC_DIR / "pawpal_ai" / "app.py"), run_name="__main__")
