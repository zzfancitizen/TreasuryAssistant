import logging

from app.core.logging_config import _with_defaults


def test_logging_config_defaults_disable_existing_loggers_to_false() -> None:
    payload = _with_defaults({"version": 1})

    assert payload["disable_existing_loggers"] is False


def test_logging_config_preserves_explicit_values() -> None:
    payload = _with_defaults({"version": 1, "disable_existing_loggers": True})

    assert payload["disable_existing_loggers"] is True
    assert logging.WARNING == 30
