"""Shared console logging setup for every Python entry point in this
project, so errors and progress show up consistently (timestamp, level,
module) instead of relying on scattered `print` calls that silently
disappear on a crash.

Usage:
    from logging_setup import get_logger
    log = get_logger(__name__)
    log.info("...")

Output is colored (dim timestamp/module, level-colored tag) when stdout is
a terminal; piping/redirecting (as smoke_train.py's subprocess capture and
`--metrics-csv`-style file redirects do) or setting `NO_COLOR` disables it
automatically, so log files and test output stay plain ASCII. Use
`color_enabled()` / `paint()` for colors elsewhere (e.g. a script's own
per-iteration summary line) so they always agree with this decision.
"""

import logging
import os
import sys

_CONFIGURED = False

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_COLORS = {
    "black": "30", "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
}
_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[34m",
    logging.INFO: "\x1b[32m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[1;31m",
    logging.CRITICAL: "\x1b[1;41m",
}


def color_enabled() -> bool:
    """Whether ANSI colors should be used on stdout right now."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def paint(color: str, text: str, *, bold: bool = False, dim: bool = False) -> str:
    """Wrap `text` in an ANSI color (one of `_COLORS`) when `color_enabled()`;
    otherwise return it unchanged. `bold`/`dim` stack with the color."""
    if not color_enabled():
        return text
    code = _COLORS[color]
    prefix = f"\x1b[{'1;' if bold else ''}{'2;' if dim else ''}{code}m"
    return f"{prefix}{text}{_RESET}"


class _ColorFormatter(logging.Formatter):
    def __init__(self, color: bool):
        super().__init__(datefmt="%H:%M:%S")
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"
        message = record.getMessage()
        if self._color:
            level_color = _LEVEL_COLORS.get(record.levelno, "")
            line = (
                f"{_DIM}{ts}{_RESET} {level_color}[{record.levelname}]{_RESET} "
                f"{_DIM}{record.name}:{_RESET} {message}"
            )
        else:
            line = f"{ts} [{record.levelname}] {record.name}: {message}"
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line = f"{line}\n{record.exc_text}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


def configure(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_ColorFormatter(color_enabled()))
    logging.basicConfig(level=level, handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
