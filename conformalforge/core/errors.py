"""Error taxonomy for ConformalForge (E100..E500).

Every error carries a stable code so failures are greppable and testable.

Author: 晨星
"""

from __future__ import annotations


class ConformalError(Exception):
    """Base class for all ConformalForge errors."""

    code = "E000"

    def __init__(self, message: str, *, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(ConformalError):
    code = "E100"


class DataError(ConformalError):
    code = "E200"


class ValidationError(ConformalError):
    code = "E300"


class ModelError(ConformalError):
    code = "E400"


class PipelineError(ConformalError):
    code = "E500"
