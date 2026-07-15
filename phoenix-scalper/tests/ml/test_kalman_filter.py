import pytest
import numpy as np
from ml.kalman_filter import KalmanPricePredictor, compute_kalman_features


class TestKalmanFilter:
    def test_update_returns_required_keys(self):
        kf = KalmanPricePredictor()
        result = kf.update(50000.0)
        assert "kf_price" in result
        assert "kf_trend" in result
        assert "kf_prediction" in result
        assert "kf_confidence" in result
        assert "kf_direction" in result

    def test_confidence_range(self):
        kf = KalmanPricePredictor()
        prices = 50000 + np.cumsum(np.random.randn(100) * 10)
        confidences = []
        for p in prices:
            result = kf.update(p)
            confidences.append(result["kf_confidence"])
        assert all(0 <= c <= 1.0 for c in confidences)

    def test_direction_detection(self):
        kf = KalmanPricePredictor()
        prices = 50000 + np.arange(100) * 10
        result = kf.update(prices[-1])
        assert result["kf_direction"] in (-1, 0, 1)

    def test_compute_kalman_features(self):
        np.random.seed(42)
        n = 200
        prices = 50000 + np.cumsum(np.random.randn(n) * 10)
        atr_pct = np.abs(np.random.randn(n)) * 0.001 + 0.005
        volume_ratio = np.random.randn(n) * 0.2 + 1.0

        features = compute_kalman_features(prices, atr_pct, volume_ratio)
        assert len(features["kf_price"]) == n
        assert len(features["kf_trend"]) == n
        assert len(features["kf_prediction"]) == n
        assert len(features["kf_confidence"]) == n
