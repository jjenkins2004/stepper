"""Optional logging setup for pipeline runs.

`configure_logging` wires the stdlib root logger so `[STEP_*]` / `[MODULE_*]` lines
print. It mirrors a typical CLI setup: `basicConfig` if nothing's configured yet, else
just raise the level. Tracing/observability is out of scope — add spans via `Hooks`.
"""

from __future__ import annotations

import logging


def configure_logging(*, level: int = logging.INFO, fmt: str = "%(message)s") -> None:
    """Set up stdlib logging so step/module lines print. `level` sets verbosity; `fmt`
    is the log-line format (default: just the message). `basicConfig` if the root logger
    has no handlers yet, otherwise just set its level."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=fmt)
    else:
        root_logger.setLevel(level)
