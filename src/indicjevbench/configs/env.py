"""Environment variable access helpers.

Never logs or exposes secret values — only presence is checked.
"""

import os


def get_api_key(*names: str) -> str | None:
    """Return the first non-empty environment variable among ``names``.

    Args:
        names: Candidate environment variable names, in priority order.

    Returns:
        The first non-empty value, or ``None`` if none are set.

    Raises:
        ValueError: If no names are given.
    """
    if not names:
        raise ValueError("at least one environment variable name is required")
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
