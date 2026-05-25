"""Structured JSON logging configuration (Sprint 8 §5).

All proxy decisions are logged as JSON lines to stdout/stderr
for collection by systemd journal, Docker, or any log aggregator.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter.

    Output: {"timestamp": "...", "level": "INFO", "message": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields") and record.extra_fields:
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure structured JSON logging to stdout.

    Replaces default handlers. Suppresses noisy third-party loggers.
    Returns the root logger.

    Args:
        level: Log level as string ("INFO", "DEBUG") or int (logging.INFO).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

    # Suppress noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("fakeredis").setLevel(logging.WARNING)

    return root_logger
