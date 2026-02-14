import sys
from pathlib import Path

# Add the src directory to sys.path so we can import modules from src/
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))
