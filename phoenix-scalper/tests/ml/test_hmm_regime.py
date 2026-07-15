import pytest
import numpy as np
from ml.hmm_regime import RegimeDetector, compute_hmm_features


class TestRegimeDetector:
    def setup_method(self):
        self.hmm = RegimeDetector(n_states=3, n_iter=5)

    def test_fit_and_predict(self):
        np.random.seed(42)
        n = 200
        returns = np.random.randn(n) * 0.01
        vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
        volume_change = np.random.randn(n) * 0.1

        obs = np.column_stack([returns, vol, volume_change])
        self.hmm.fit(obs)

        probs = self.hmm.predict_regime_probs_fast(obs)
        assert probs.shape == (n, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_regime_fast(self):
        np.random.seed(42)
        n = 100
        returns = np.random.randn(n) * 0.01
        vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
        volume_change = np.random.randn(n) * 0.1

        obs = np.column_stack([returns, vol, volume_change])
        self.hmm.fit(obs)

        regime = self.hmm.predict_regime_fast(obs)
        assert len(regime) == n
        assert all(r in [0, 1, 2] for r in regime)

    def test_fit_insufficient_data(self):
        obs = np.random.randn(5, 3)
        self.hmm.fit(obs)
        assert self.hmm.pi[0] == pytest.approx(0.3, abs=0.01)

    def test_compute_hmm_features(self):
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.01
        vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
        volume_change = np.random.randn(n) * 0.1

        train_n = n // 2
        features = compute_hmm_features(
            returns[:train_n], vol[:train_n], volume_change[:train_n],
            full_returns=returns, full_vol=vol, full_volume_change=volume_change,
        )

        assert len(features["hmm_regime"]) == n
        assert len(features["hmm_p_bull"]) == n
        assert len(features["hmm_p_bear"]) == n
        assert "hmm_regime_stability" in features

    def test_compute_hmm_features_no_full(self):
        np.random.seed(42)
        n = 200
        returns = np.random.randn(n) * 0.01
        vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
        volume_change = np.random.randn(n) * 0.1

        features = compute_hmm_features(returns, vol, volume_change)

        assert len(features["hmm_regime"]) == n

    def test_emission_prob_stability(self):
        np.random.seed(42)
        obs = np.random.randn(50, 3)
        probs = self.hmm._emission_prob(obs)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
