import os
import sys
from pathlib import Path

# Get absolute path to project root
project_root = Path(__file__).parent.parent.absolute()
src_path = project_root

# Add to Python path
sys.path.insert(0, str(src_path)) 