"""Tests for indicjevbench/core/dataset.py — JSONL loading and validation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from indicjevbench.core.dataset import load_manifest, load_tasks
from indicjevbench.schemas.contracts import Task


def _task_row(i: int) -> dict:
    return {
        "id": f"t-{i:03d}",
        "family": "intent",
        "lang": "hi-Deva",
        "state": f"message {i}",
        "question": {
            "type": "choice",
            "instructions": "Pick the intent.",
            "options": ["a", "b", "c"],
            "criteria": {},
        },
        "expected": i % 3,
        "split": "v1",
        "source": "massive",
        "license": "CC BY 4.0",
        "provenance": {"origin": "test"},
    }


def _write_jsonl(path: Path, rows: list[dict], blank_lines: tuple[int, ...] = ()) -> Path:
    lines = []
    for i, row in enumerate(rows):
        if i in blank_lines:
            lines.append("")
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_tasks
# ---------------------------------------------------------------------------


def test_load_tasks_roundtrip(tmp_path: Path):
    rows = [_task_row(i) for i in range(5)]
    path = _write_jsonl(tmp_path / "tasks.jsonl", rows)
    tasks = load_tasks(path)
    assert len(tasks) == 5
    assert all(isinstance(t, Task) for t in tasks)
    assert [t.id for t in tasks] == [r["id"] for r in rows]
    # Round-trip equality: asdict -> JSON -> Task preserves all fields.
    for task, row in zip(tasks, rows):
        assert task == Task.from_dict(json.loads(json.dumps(dataclasses.asdict(task))))
        assert task.q_type == row["question"]["type"]
        assert task.options == tuple(row["question"]["options"])


def test_load_tasks_skips_blank_lines(tmp_path: Path):
    rows = [_task_row(i) for i in range(3)]
    path = _write_jsonl(tmp_path / "tasks.jsonl", rows, blank_lines=(1,))
    tasks = load_tasks(path)
    assert len(tasks) == 3


def test_load_tasks_max_examples(tmp_path: Path):
    rows = [_task_row(i) for i in range(10)]
    path = _write_jsonl(tmp_path / "tasks.jsonl", rows)
    tasks = load_tasks(path, max_examples=4)
    assert len(tasks) == 4
    assert tasks[0].id == "t-000"


def test_load_tasks_invalid_json_reports_line(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_task_row(0)) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:.*invalid JSON"):
        load_tasks(path)


def test_load_tasks_bad_row_reports_line(tmp_path: Path):
    rows = [_task_row(0), _task_row(1)]
    rows[1].pop("expected")
    path = _write_jsonl(tmp_path / "tasks.jsonl", rows)
    with pytest.raises(ValueError, match=r":2:.*missing required keys"):
        load_tasks(path)


def test_load_tasks_validation_error_reports_line(tmp_path: Path):
    rows = [_task_row(0), _task_row(1)]
    rows[1]["expected"] = 99  # out of range for 3 options
    path = _write_jsonl(tmp_path / "tasks.jsonl", rows)
    with pytest.raises(ValueError, match=r":2:.*out of range"):
        load_tasks(path)


def test_load_tasks_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "nope.jsonl")


def test_load_tasks_bad_max_examples_type(tmp_path: Path):
    path = _write_jsonl(tmp_path / "tasks.jsonl", [_task_row(0)])
    with pytest.raises(TypeError):
        load_tasks(path, max_examples="3")  # type: ignore[arg-type]


def test_load_tasks_bad_max_examples_value(tmp_path: Path):
    path = _write_jsonl(tmp_path / "tasks.jsonl", [_task_row(0)])
    with pytest.raises(ValueError):
        load_tasks(path, max_examples=0)


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_roundtrip(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": "v1", "files": ["a.jsonl"]}), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest["version"] == "v1"
    assert manifest["files"] == ["a.jsonl"]


def test_load_manifest_invalid_json(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_manifest(path)


def test_load_manifest_not_object(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="object"):
        load_manifest(path)


def test_load_manifest_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")
