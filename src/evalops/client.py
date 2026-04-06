"""EvalOpsClient — fire-and-forget submission of EvalRunReport objects.

The client **never raises**: all errors are logged at WARNING level and
swallowed so that a misconfigured or unreachable EvalOps endpoint cannot
break an evaluation run.

Configuration is read from environment variables:

- ``EVALOPS_ENDPOINT`` — HTTP(S) URL to POST the JSON payload to.
- ``EVALOPS_API_KEY``  — Bearer token added as ``Authorization`` header.

When ``EVALOPS_ENDPOINT`` is not set the client is a no-op.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from src.evalops.schema import EvalRunReport

logger = logging.getLogger(__name__)


class EvalOpsClient:
    """Submit :class:`EvalRunReport` payloads to a remote EvalOps endpoint.

    Args:
        endpoint: Full URL to POST to, or ``None`` to no-op.
        api_key: Optional bearer token.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._endpoint = endpoint or ""
        self._api_key = api_key or ""

    @classmethod
    def from_env(cls) -> "EvalOpsClient":
        """Construct from ``EVALOPS_ENDPOINT`` / ``EVALOPS_API_KEY`` env vars."""
        return cls(
            endpoint=os.environ.get("EVALOPS_ENDPOINT"),
            api_key=os.environ.get("EVALOPS_API_KEY"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, report: EvalRunReport) -> None:
        """Submit *report* to the configured endpoint.

        Errors are caught and logged at WARNING level — this method never
        raises so it cannot break the calling evaluation run.
        """
        try:
            self._do_submit(report)
        except Exception as exc:
            logger.warning("EvalOpsClient.submit failed (ignored): %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_submit(self, report: EvalRunReport) -> None:
        if not self._endpoint:
            logger.debug("EvalOpsClient: no endpoint configured, skipping submit")
            return

        import urllib.request

        payload = asdict(report)
        data = json.dumps(payload, ensure_ascii=False, default=str).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        with urllib.request.urlopen(req, timeout=5) as _resp:
            pass

        logger.info(
            "EvalOpsClient: submitted run %r to %s", report.run_id, self._endpoint
        )
