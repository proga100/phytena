"""Logging setup with secret redaction."""

from __future__ import annotations

import logging
import re

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"KGAT_[A-Za-z0-9]{10,}"),
)

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[37m",
    logging.INFO: "\033[36m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m\033[1m",
}
_RESET = "\033[0m"


class _RedactFilter(logging.Filter):
    """Strip anything that looks like an API key from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pattern in _SECRET_PATTERNS:
            msg = pattern.sub("[REDACTED]", msg)
        record.msg = msg
        record.args = ()
        return True


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def configure(level: int = logging.INFO) -> None:
    """Configure root logger once. Safe to call multiple times."""
    root = logging.getLogger()
    if getattr(root, "_growz_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter("[%(levelname)s] %(message)s"))
    handler.addFilter(_RedactFilter())

    root.addHandler(handler)
    root.setLevel(level)
    root._growz_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)


def step_banner(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
