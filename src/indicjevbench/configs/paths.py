"""Path configuration for IndicJevBench datasets and results."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchPaths:
    """Resolved filesystem locations for the benchmark.

    Attributes:
        bench_root: Repository/install root containing ``datasets/``.
        datasets_dir: Directory with the versioned JSONL task files.
        results_dir: Directory where run results and raw logs are written.
    """

    bench_root: Path
    datasets_dir: Path
    results_dir: Path

    @classmethod
    def default(cls) -> "BenchPaths":
        """Resolve the standard layout relative to this package.

        In an editable/source checkout the datasets live at the repository
        root (``<root>/src/indicjevbench/configs/paths.py`` → root is the
        first ancestor containing ``datasets/``). In an installed wheel the
        datasets are not shipped, so we fall back to the grandparent of the
        ``indicjevbench`` package directory.

        Returns:
            BenchPaths with resolved absolute paths.
        """
        here = Path(__file__).resolve()
        root: Path | None = None
        for parent in here.parents:
            if (parent / "datasets").is_dir():
                root = parent
                break
        if root is None:
            # installed layout: site-packages/indicjevbench/configs/paths.py
            root = here.parents[2]
        return cls(
            bench_root=root,
            datasets_dir=root / "datasets" / "v1",
            results_dir=root / "results" / "v1",
        )

    def dataset_files(self) -> list[Path]:
        """Return all JSONL dataset files, sorted by name.

        Returns:
            Sorted list of dataset file paths (empty if none exist).
        """
        if not self.datasets_dir.is_dir():
            return []
        return sorted(self.datasets_dir.glob("*.jsonl"))

    def results_file(self, run_id: str) -> Path:
        """Return the results JSON path for a run.

        Args:
            run_id: Run identifier (e.g. ``run_20250101_000000``).

        Returns:
            Path to ``<results_dir>/<run_id>.json``.

        Raises:
            ValueError: If ``run_id`` is empty.
        """
        if not run_id:
            raise ValueError("run_id must be non-empty")
        return self.results_dir / f"{run_id}.json"
