from __future__ import annotations

import os
import sys
from typing import Final

_RESET: Final[str] = "\x1b[0m"
_DIM: Final[str] = "\x1b[2m"
_CYAN: Final[str] = "\x1b[36m"
_GREEN: Final[str] = "\x1b[32m"
_RED: Final[str] = "\x1b[31m"
_MAGENTA: Final[str] = "\x1b[35m"
_BLUE: Final[str] = "\x1b[34m"


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    term = os.getenv("TERM", "")
    if term.lower() == "dumb":
        return False
    return sys.stderr.isatty()


def _colorize(value: str, color: str) -> str:
    if not _supports_color():
        return value
    return f"{color}{value}{_RESET}"


def _event_label(event: str, color: str) -> str:
    return _colorize(event, color)


def format_step_start(*, step_name: str, input_type: str, output_type: str) -> str:
    return (
        f"[{_event_label('STEP_START', _CYAN)}] "
        f"step={step_name} input={input_type} output={output_type}"
    )


def format_step_end(*, step_name: str, elapsed_ms: int, output_type: str) -> str:
    timing = _colorize(f"{elapsed_ms}ms", _DIM)
    return (
        f"[{_event_label('STEP_DONE', _GREEN)}] "
        f"step={step_name} output={output_type} elapsed={timing}"
    )


def format_step_fail(*, step_name: str, elapsed_ms: int, error_type: str) -> str:
    timing = _colorize(f"{elapsed_ms}ms", _DIM)
    return (
        f"[{_event_label('STEP_FAIL', _RED)}] "
        f"step={step_name} error={error_type} elapsed={timing}"
    )


def format_module_start(*, module_name: str, step_count: int) -> str:
    return (
        f"[{_event_label('MODULE_START', _MAGENTA)}] "
        f"module={module_name} steps={step_count}"
    )


def format_module_end(*, module_name: str, elapsed_ms: int) -> str:
    timing = _colorize(f"{elapsed_ms}ms", _DIM)
    return f"[{_event_label('MODULE_DONE', _MAGENTA)}] module={module_name} elapsed={timing}"


def format_debug_fetch(*, payload_type: str) -> str:
    return f"[{_event_label('FETCH', _BLUE)}] payload={payload_type}"


def format_debug_persist(*, payload_type: str) -> str:
    return f"[{_event_label('PERSIST', _BLUE)}] payload={payload_type}"
