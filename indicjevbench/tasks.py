from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class Task:
    id: str
    family: str
    lang: str
    state: str
    question: dict
    expected: Any
    split: str
    source: str
    license: str
    provenance: dict

    @property
    def q_type(self) -> str:
        return self.question["type"]

    @property
    def options(self) -> list[str] | None:
        return self.question.get("options")

def load_tasks(path: str | Path, max_examples: int | None = None) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                tasks.append(Task(**d))
                if max_examples and len(tasks) >= max_examples:
                    break
    return tasks
