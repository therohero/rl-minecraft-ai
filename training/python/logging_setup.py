"""Shared console logging setup for every Python entry point in this
project, so errors and progress show up consistently (timestamp, level,
module) instead of relying on scattered `print` calls that silently
disappear on a crash.

Usage:
    from logging_setup import get_logger
    log = get_logger(__name__)
    log.info("...")
"""

import logging
import sys

_CONFIGURED = False


def configure(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
