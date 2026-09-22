import sys
from pathlib import Path

# Add bench/indicjevbench to sys.path so `import indicjevbench` works
# when pytest is run from the nirnaya project root.
sys.path.insert(0, str(Path(__file__).parent.parent))
