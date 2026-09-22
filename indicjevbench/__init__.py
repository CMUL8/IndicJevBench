"""IndicJevBench — benchmark harness for Indic structured decision AI."""
from pathlib import Path
__version__ = "1.0.0"
BENCH_ROOT = Path(__file__).parent.parent
DATASETS_DIR = BENCH_ROOT / "datasets" / "v1"
RESULTS_DIR = BENCH_ROOT / "results" / "v1"
