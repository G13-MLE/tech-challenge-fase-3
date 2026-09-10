import sys
from pathlib import Path

# Must insert before any src imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.stopwords import ENGLISH_STOPWORDS
print("OK:", len(ENGLISH_STOPWORDS))
