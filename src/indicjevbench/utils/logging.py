"""Logging helpers for IndicJevBench entrypoints."""

import logging
import sys
from typing import TextIO

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Logger name (pass ``__name__`` from the calling module).

    Returns:
        The named logger. Handlers/levels are configured only by
        ``configure_logging`` at entrypoints, never here.
    """
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Configure root logging for CLI entrypoints.

    Idempotent: safe to call multiple times.

    Args:
        level: Minimum log level for the root handler.
        stream: Destination for log records (default: ``sys.stdout`` so CLI
            progress lines stay on stdout; pass ``sys.stderr`` to detach them).
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
