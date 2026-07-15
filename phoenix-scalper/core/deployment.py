import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Release:
    version: str
    timestamp: str
    changelog: str
    validation_report: str
    status: str
    rollback_point: Optional[str] = None


class DeploymentManager:
    def __init__(self, risk_governor=None, on_notify=None):
        self._releases: list[Release] = []
        self._rollback_version: Optional[str] = None
        self._risk_governor = risk_governor
        self.on_notify = on_notify

    def create_release(self, version: str, changelog: str, validation_report: str) -> Release:
        release = Release(
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            changelog=changelog,
            validation_report=validation_report,
            status="pending",
        )
        self._releases.append(release)
        return release

    def promote_to_shadow(self, version: str) -> bool:
        release = self._get_release(version)
        if not release:
            return False
        release.status = "shadow"
        logger.info(f"Release {version} promoted to shadow mode")
        if self.on_notify:
            self.on_notify("deployment", event_type="promote_shadow", version=version, status="shadow")
        return True

    def promote_to_canary(self, version: str) -> bool:
        release = self._get_release(version)
        if not release:
            return False
        release.status = "canary"
        logger.info(f"Release {version} promoted to canary (10% traffic)")
        if self.on_notify:
            self.on_notify("deployment", event_type="promote_canary", version=version, status="canary")
        return True

    def promote_to_full(self, version: str) -> bool:
        release = self._get_release(version)
        if not release:
            return False
        release.status = "live"
        self._rollback_version = version
        logger.info(f"Release {version} promoted to full deployment")
        if self.on_notify:
            self.on_notify("deployment", event_type="promote_full", version=version, status="live")
        return True

    def check_rollback_needed(self) -> Optional[str]:
        if not self._risk_governor:
            return None
        state = self._risk_governor.get_state()
        if state.level.name in ("PAUSED", "EMERGENCY"):
            logger.warning(f"Rollback triggered by risk level: {state.level.name}")
            return self._rollback_version
        return None

    def rollback(self) -> Optional[str]:
        if not self._rollback_version:
            logger.warning("No rollback point available")
            return None
        logger.info(f"Rolling back to {self._rollback_version}")
        for release in self._releases:
            if release.version == self._rollback_version:
                release.status = "rollback_target"
            elif release.status == "live":
                release.status = "rolled_back"
        if self.on_notify:
            self.on_notify("deployment", event_type="rollback", version=self._rollback_version, status="rolled_back")
        return self._rollback_version

    def _get_release(self, version: str) -> Optional[Release]:
        for r in self._releases:
            if r.version == version:
                return r
        return None

    def get_release_history(self) -> list[Release]:
        return list(self._releases)
