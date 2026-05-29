"""Structured JSON logging configuration (Feature).

All proxy decisions are logged as JSON lines to stdout/stderr
for collection by systemd journal, Docker, or any log aggregator.

Supports file logging to /var/log/proxy-cesar or user-defined location
via LOG_FILE environment variable.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

_setup_logger = logging.getLogger(__name__)


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
    """Configure structured JSON logging to stdout + optional file.

    Replaces default handlers. Suppresses noisy third-party loggers.
    Returns the root logger.

    Reads LOG_FILE environment variable:
    - If set: logs to that file path (with rotation)
    - If unset: checks /var/log/proxy-cesar (if writable)
    - Fallback: stdout only

    Args:
        level: Log level as string ("INFO", "DEBUG") or int (logging.INFO).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handlers = []

    # Always add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JSONFormatter())
    handlers.append(stdout_handler)

    # Try to add file handler
    log_file = os.environ.get("LOG_FILE")
    if not log_file:
        # Default to /var/log/proxy-cesar if writable
        default_log_dir = "/var/log/proxy-cesar"
        if os.path.isdir(default_log_dir) and os.access(default_log_dir, os.W_OK):
            log_file = os.path.join(default_log_dir, "proxy.log")
        elif os.path.isdir("/var/log") and os.access("/var/log", os.W_OK):
            # Fallback: write directly to /var/log
            log_file = "/var/log/proxy-cesar.log"

    if log_file:
        try:
            # Create directory if needed
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, mode=0o755, exist_ok=True)

            # Use rotating file handler: 100MB max, keep 5 backups
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=100 * 1024 * 1024,  # 100 MB
                backupCount=5,
            )
            file_handler.setFormatter(JSONFormatter())
            handlers.append(file_handler)
            _setup_logger.info("Logging to file: %s", log_file)
        except Exception as e:
            _setup_logger.warning("Could not setup file logging to %s: %s", log_file, e)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = handlers

    # Suppress noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("fakeredis").setLevel(logging.WARNING)

    return root_logger
