"""Tests for indicjevbench/core/packaging.py — raw JSONL → v1 task files + manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indicjevbench.core.dataset import load_tasks
from indicjevbench.core.packaging import DatasetPackager

_RAW_ROWS = [
    # MASSIVE intent row, two questions
    {
        "source": "massive",
        "lang": "hi-Deva",
        "state": "बुधवार को अलार्म लगाओ",
        "questions": [
            {
                "qid": "q0",
                "type": "choice",
                "instructions": "Select the intent.",
                "options": ["alarm_set", "alarm_remove"],
                "label": 0,
            },
            {
                "qid": "q1",
                "type": "choice",
                "instructions": "Which domain?",
                "options": ["alarm", "music"],
                "label": 0,
            },
        ],
    },
    # synthetic row: score + noul (bool label normalizes to 0/1)
    {
        "source": "synthetic",
        "lang": "hi-Latn",
        "state": "mera refund nahi aaya",
        "questions": [
            {
                "qid": "q0",
                "task": "urgency_score",
                "type": "score",
                "instructions": "Rate urgency 1-5.",
                "options": ["1 - very low", "2 - low", "3 - medium", "4 - high", "5 - critical"],
                "label": 4,
            },
            {
                "qid": "q1",
                "task": "escalation_noul",
                "type": "noul",
                "instructions": "Does this need a human?",
                "label": True,
            },
        ],
    },
    # non-redistributable source → skipped
    {"source": "internal-corpus", "lang": "hi-Deva", "state": "x", "questions": []},
    # unknown question type → question skipped, row counted
    {
        "source": "comilingua",
        "lang": "hi-Latn",
        "state": "kya haal hai bhai",
        "questions": [
            {"qid": "q0", "type": "freeform", "instructions": "translate", "label": 0},
        ],
    },
]


def _write_raw(path: Path) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in _RAW_ROWS]
    lines.insert(2, "not-valid-json")  # exercises the skip path
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_package_round_trip(tmp_path: Path) -> None:
    """Packaging writes per-dataset JSONL + manifest; output loads via load_tasks."""
    raw = tmp_path / "test.jsonl"
    out_dir = tmp_path / "v1"
    manifest_path = tmp_path / "manifest.json"
    _write_raw(raw)

    packager = DatasetPackager(data_final=raw, datasets_dir=out_dir, manifest_path=manifest_path)
    manifest = packager.package()

    assert manifest["schema_version"] == 1
    assert manifest["total_items"] == 4  # 2 massive + 2 synthetic; others skipped
    assert manifest["skipped_rows"] == 2  # bad JSON line + internal-corpus row

    splits = manifest["splits"]
    assert set(splits) == {"intent_massive", "synthetic_enterprise"}
    assert splits["intent_massive"]["n_items"] == 2
    assert splits["intent_massive"]["langs"] == ["hi-Deva"]
    assert splits["synthetic_enterprise"]["families"] == ["escalation", "urgency"]

    # Output files exist at declared paths and parse through the strict loader
    for name, split in splits.items():
        tasks = load_tasks(out_dir / f"{name}.jsonl")
        assert len(tasks) == split["n_items"]
        for task in tasks:
            assert task.license == "CC BY 4.0"
            assert task.provenance["origin"] == task.source

    # Noul bool label normalized to int
    synthetic = load_tasks(out_dir / "synthetic_enterprise.jsonl")
    noul_task = next(t for t in synthetic if t.q_type == "noul")
    assert noul_task.expected == 1
    score_task = next(t for t in synthetic if t.q_type == "score")
    assert score_task.expected == 4
    assert score_task.family == "urgency"


def test_package_missing_input(tmp_path: Path) -> None:
    """Missing raw input raises FileNotFoundError."""
    packager = DatasetPackager(
        data_final=tmp_path / "absent.jsonl",
        datasets_dir=tmp_path / "v1",
        manifest_path=tmp_path / "manifest.json",
    )
    with pytest.raises(FileNotFoundError):
        packager.package()


def test_package_no_items(tmp_path: Path) -> None:
    """Input producing zero items raises ValueError."""
    raw = tmp_path / "test.jsonl"
    raw.write_text('{"source": "internal-only", "questions": []}\n', encoding="utf-8")
    packager = DatasetPackager(
        data_final=raw,
        datasets_dir=tmp_path / "v1",
        manifest_path=tmp_path / "manifest.json",
    )
    with pytest.raises(ValueError, match="No items produced"):
        packager.package()
