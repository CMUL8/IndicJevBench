"""Atomic filesystem write helpers."""

import os
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + rename).

    Args:
        path: Destination file path.
        text: Full text content to write.

    Raises:
        OSError: If the write or rename fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_append_line(path: Path, line: str) -> None:
    """Append a single line to ``path``, creating parents as needed.

    The line is written and flushed immediately (line-buffered) so the raw
    log survives interruptions; a trailing newline is added if missing.

    Args:
        path: Destination file path.
        line: Line content without trailing newline.

    Raises:
        OSError: If the append fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(line if line.endswith("\n") else line + "\n")
