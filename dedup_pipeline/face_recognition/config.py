"""Client configuration. The service address has no default: it must be supplied."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional
from urllib.parse import urlsplit

from .errors import FrConfigError

ENV_BASE_URL = "FR_DOPAMS_BASE_URL"
ENV_CONNECT_TIMEOUT = "FR_DOPAMS_CONNECT_TIMEOUT"
ENV_READ_TIMEOUT = "FR_DOPAMS_READ_TIMEOUT"
ENV_RETRIES = "FR_DOPAMS_RETRIES"
ENV_ALLOW_LOG_WRITES = "FR_DOPAMS_ALLOW_AUDIT_LOG_WRITES"
ENV_API_KEY = "FR_DOPAMS_API_KEY"  # sent as X-API-Key to POST /compare


@dataclass(frozen=True)
class FrConfig:
    base_url: str
    api_key: str = field(default="", repr=False)
    connect_timeout: float = 5.0
    read_timeout: float = 60.0  # CPU embedding of one image can take seconds
    retries: int = 2
    backoff: float = 0.5
    # /match, /match/person and /search append a row to FR-DOPAMS ``match_logs`` on success.
    allow_audit_log_writes: bool = False

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise FrConfigError(f"{ENV_BASE_URL} must be an http(s) URL, got {self.base_url!r}")
        if self.connect_timeout <= 0 or self.read_timeout <= 0 or self.retries < 0:
            raise FrConfigError("timeouts must be positive and retries must not be negative")

    def __repr__(self) -> str:
        parts = urlsplit(self.base_url)  # never echo user:password@ if someone put it in the URL
        where = f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "") + parts.path
        return f"FrConfig(base_url={where!r}, allow_audit_log_writes={self.allow_audit_log_writes})"

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "FrConfig":
        env = os.environ if environ is None else environ
        base_url = (env.get(ENV_BASE_URL) or "").strip()
        if not base_url:
            raise FrConfigError(f"{ENV_BASE_URL} is not set; it must point at the FR-DOPAMS service.")
        try:
            return cls(
                base_url=base_url,
                api_key=(env.get(ENV_API_KEY) or "").strip(),
                connect_timeout=float(env.get(ENV_CONNECT_TIMEOUT, cls.connect_timeout)),
                read_timeout=float(env.get(ENV_READ_TIMEOUT, cls.read_timeout)),
                retries=int(env.get(ENV_RETRIES, cls.retries)),
                allow_audit_log_writes=env.get(ENV_ALLOW_LOG_WRITES, "").strip().lower() in ("1", "true", "yes"),
            )
        except ValueError as exc:
            raise FrConfigError(f"invalid numeric setting: {exc}") from None
