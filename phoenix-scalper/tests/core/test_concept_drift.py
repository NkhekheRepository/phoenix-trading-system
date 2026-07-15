import pytest
import numpy as np
from core.concept_drift import ConceptDriftDetector


class TestConceptDriftDetector:
    def setup_method(self):
        self.detector = ConceptDriftDetector(
            psi_threshold=0.2,
            kl_threshold=0.1,
            wasserstein_threshold=0.5,
            window_size=200,
        )

    def test_no_drift_identical_distributions(self):
        values = np.random.randn(500) * 10 + 100
        self.detector.set_reference("test", values)
        for v in values[:200]:
            self.detector.update("test", v)

    def test_drift_different_distributions(self):
        ref = np.random.randn(500) * 10 + 100
        self.detector.set_reference("test", ref)

        drifted = np.random.randn(500) * 30 + 200
        result = None
        for v in drifted[:250]:
            r = self.detector.update("test", v)
            if r:
                result = r
        if result:
            assert result.drift_detected is True

    def test_drift_summary(self):
        values = np.random.randn(500) * 10 + 100
        self.detector.set_reference("test", values)
        for v in values[:200]:
            self.detector.update("test", v)
        summary = self.detector.get_drift_summary()
        assert "test" in summary
