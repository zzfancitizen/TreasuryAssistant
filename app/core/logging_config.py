from __future__ import annotations

import logging
from logging.config import dictConfig
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LOGGING_CONFIG_PATH = Path("app/config/logging.yaml")

logger = logging.getLogger(__name__)


def configure_logging(config_path: str | Path | None = None) -> None:
    path = Path(config_path or os.getenv("LOGGING_CONFIG_PATH", DEFAULT_LOGGING_CONFIG_PATH))
    if not path.exists():
        logging.basicConfig(level=logging.INFO)
        logger.warning("logging.config.missing", extra={"config_path": str(path)})
        return

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dictConfig(_with_defaults(payload))
    logger.info("logging.config.loaded", extra={"config_path": str(path)})


def _with_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    configured = dict(payload)
    configured.setdefault("version", 1)
    configured.setdefault("disable_existing_loggers", False)
    return configured
