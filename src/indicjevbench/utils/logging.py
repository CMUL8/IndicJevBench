"""Logging helpers for IndicJevBench entrypoints."""

import logging

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


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI entrypoints.

    Idempotent: safe to call multiple times.

    Args:
        level: Minimum log level for the root handler.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
