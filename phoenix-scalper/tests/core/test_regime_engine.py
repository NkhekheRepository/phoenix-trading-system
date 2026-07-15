import pytest
import pandas as pd
import numpy as np
from core.regime_engine import RegimeEngine, Regime


class TestRegimeEngine:
    def setup_method(self):
        self.engine = RegimeEngine()

    def _make_dataframe(self, trend: str = "bull"):
        np.random.seed(42)
        n = 200
        if trend == "bull":
            t = np.arange(n, dtype=float)
            prices = 50000 + t * 100 + np.sin(t * 0.05) * 200
            rsi = np.full(n, 65)
            adx = 35
            hmm_p_bull = 0.8
            hmm_p_bear = 0.1
            macd_shift = 10
        elif trend == "bear":
            t = np.arange(n, dtype=float)
            prices = 55000 - t * 100 + np.sin(t * 0.05) * 200
            rsi = np.full(n, 35)
            adx = 35
            hmm_p_bull = 0.1
            hmm_p_bear = 0.8
            macd_shift = -10
        else:
            prices = 50000 + np.cumsum(np.random.randn(n) * 20)
            rsi = 50 + np.random.randn(n) * 8
            adx = 15
            hmm_p_bull = 0.35
            hmm_p_bear = 0.35
            macd_shift = 0

        df = pd.DataFrame({
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "volume": np.abs(np.random.randn(n)) * 500 + 1000,
        })

        df["ema_7"] = pd.Series(prices).ewm(span=7).mean().values
        df["ema_21"] = pd.Series(prices).ewm(span=21).mean().values
        df["ema_5"] = pd.Series(prices).ewm(span=5).mean().values
        df["ema_20"] = pd.Series(prices).ewm(span=20).mean().values
        df["rsi_14"] = np.clip(rsi, 0, 100)
        df["adx"] = adx
        df["atr_pct"] = np.full(n, 0.005)
        df["bb_width"] = np.full(n, 0.04)
        df["bb_width_sma"] = 0.04
        df["volume_ratio"] = np.random.randn(n) * 0.1 + 1.0
        df["plus_di"] = 28
        df["minus_di"] = 22
        df["macd"] = np.random.randn(n) * 2 + macd_shift
        df["macdsignal"] = np.random.randn(n) * 2
        df["hmm_p_bull"] = hmm_p_bull
        df["hmm_p_bear"] = hmm_p_bear

        return df

    def test_bull_regime(self):
        df = self._make_dataframe("bull")
        result = self.engine.analyze(df)
        assert result.regime in (Regime.STRONG_BULL, Regime.WEAK_BULL)

    def test_bear_regime(self):
        df = self._make_dataframe("bear")
        result = self.engine.analyze(df)
        assert result.regime in (Regime.STRONG_BEAR, Regime.WEAK_BEAR)

    def test_confidence_range(self):
        df = self._make_dataframe("bull")
        result = self.engine.analyze(df)
        assert 0 <= result.confidence <= 1.0

    def test_recommended_mode(self):
        df = self._make_dataframe("bull")
        result = self.engine.analyze(df)
        assert result.recommended_mode in ("aggressive", "normal", "capital_preservation", "scalping", "neutral")

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = self.engine.analyze(df)
        assert result.regime == Regime.SIDEWAYS
        assert result.confidence == 0.0

    def test_small_dataframe(self):
        df = self._make_dataframe("bull").iloc[:10]
        result = self.engine.analyze(df)
        assert result.regime == Regime.SIDEWAYS
