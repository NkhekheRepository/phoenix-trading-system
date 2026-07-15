import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    feature: str
    psi: float
    kl_divergence: float
    wasserstein_distance: float
    drift_detected: bool
    severity: str
    recommendation: str


class ConceptDriftDetector:
    def __init__(self, psi_threshold: float = 0.2, kl_threshold: float = 0.1,
                 wasserstein_threshold: float = 0.5, window_size: int = 500,
                 on_notify=None):
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.window_size = window_size
        self.on_notify = on_notify
        self._reference_distributions: Dict[str, np.ndarray] = {}
        self._current_windows: Dict[str, deque] = {}
        self._drift_history: Dict[str, List[float]] = {}

    def set_reference(self, feature: str, values: np.ndarray):
        if len(values) < 50:
            logger.warning(f"Reference for {feature} too small ({len(values)}), skipping")
            return
        self._reference_distributions[feature] = values.copy()
        self._drift_history[feature] = []

    def update(self, feature: str, value: float) -> Optional[DriftResult]:
        if feature not in self._current_windows:
            self._current_windows[feature] = deque(maxlen=self.window_size)

        self._current_windows[feature].append(value)
        window = self._current_windows[feature]

        if len(window) < 100 or feature not in self._reference_distributions:
            return None

        if len(window) % 50 != 0:
            return None

        current = np.array(window)
        reference = self._reference_distributions[feature]

        psi = self._compute_psi(reference, current)
        kl = self._compute_kl(reference, current)
        wasserstein = self._compute_wasserstein(reference, current)

        drift_detected = psi > self.psi_threshold or kl > self.kl_threshold

        self._drift_history[feature].append(psi)
        if len(self._drift_history[feature]) > 100:
            self._drift_history[feature] = self._drift_history[feature][-100:]

        if drift_detected:
            if psi > self.psi_threshold * 2:
                severity = "critical"
                recommendation = "retrain and validate"
            elif psi > self.psi_threshold * 1.5:
                severity = "warning"
                recommendation = "monitor closely, consider retrain"
            else:
                severity = "info"
                recommendation = "monitor"
        else:
            severity = "normal"
            recommendation = "no action"

        if drift_detected and self.on_notify and severity != "normal":
            self.on_notify("drift", feature=feature, psi=psi, kl=kl,
                           wasserstein=wasserstein, severity=severity,
                           recommendation=recommendation)

        return DriftResult(
            feature=feature,
            psi=round(psi, 4),
            kl_divergence=round(kl, 4),
            wasserstein_distance=round(wasserstein, 4),
            drift_detected=drift_detected,
            severity=severity,
            recommendation=recommendation,
        )

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        bins = 20
        combined = np.concatenate([reference, current])
        bin_edges = np.percentile(combined, np.linspace(0, 100, bins + 1))

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        psi = np.sum((ref_pct - cur_pct) * np.log((ref_pct + 1e-10) / (cur_pct + 1e-10)))
        return float(psi)

    def _compute_kl(self, reference: np.ndarray, current: np.ndarray) -> float:
        bins = 30
        combined = np.concatenate([reference, current])
        bin_edges = np.linspace(min(combined), max(combined), bins + 1)

        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)

        ref_hist = ref_hist + 1e-10
        cur_hist = cur_hist + 1e-10

        ref_pdf = ref_hist / np.sum(ref_hist)
        cur_pdf = cur_hist / np.sum(cur_hist)

        kl = np.sum(ref_pdf * np.log(ref_pdf / cur_pdf))
        return float(kl)

    def _compute_wasserstein(self, reference: np.ndarray, current: np.ndarray) -> float:
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)

        n = min(len(ref_sorted), len(cur_sorted))
        ref_sorted = ref_sorted[:n]
        cur_sorted = cur_sorted[:n]

        distance = np.mean(np.abs(ref_sorted - cur_sorted))
        return float(distance)

    def get_drift_summary(self) -> Dict:
        summary = {}
        for feature in self._reference_distributions:
            psi_vals = self._drift_history.get(feature, [])
            if psi_vals:
                summary[feature] = {
                    "current_psi": psi_vals[-1],
                    "max_psi": max(psi_vals),
                    "mean_psi": np.mean(psi_vals),
                    "trend": "increasing" if len(psi_vals) > 5 and psi_vals[-1] > np.mean(psi_vals[-5:]) else "stable",
                }
        return summary
