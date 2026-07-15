import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from core.data_quality import DataValidator, DatasetLineage


class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator(max_candle_age_minutes=60)

    def _make_dataframe(self, n_candles=100, with_anomaly=False):
        now = datetime.now(timezone.utc)
        dates = [now - timedelta(minutes=5 * i) for i in range(n_candles)]
        dates.reverse()

        closes = np.random.randn(n_candles) * 100 + 50000
        if with_anomaly:
            closes[50] = closes[50] * 2

        df = pd.DataFrame({
            "date": dates,
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": np.random.randn(n_candles) * 100 + 1000,
        })
        return df

    def test_valid_dataframe(self):
        df = self._make_dataframe()
        result = self.validator.validate_candles(df, "BTC/USDT")
        assert result["valid"] is True
        assert result["severity"] in ("ok", "info")

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = self.validator.validate_candles(df, "BTC/USDT")
        assert result["valid"] is False
        assert result["severity"] == "fatal"

    def test_nan_detection(self):
        df = self._make_dataframe(50)
        df.loc[10:20, "close"] = np.nan
        result = self.validator.validate_candles(df, "BTC/USDT")
        assert len(result["issues"]) > 0

    def test_indicator_warmup_sufficient(self):
        df = self._make_dataframe(200)
        result = self.validator.validate_indicator_warmup(df, 100)
        assert result["valid"] is True

    def test_indicator_warmup_insufficient(self):
        df = self._make_dataframe(50)
        result = self.validator.validate_indicator_warmup(df, 100)
        assert result["valid"] is False

    def test_price_anomaly_detection(self):
        df = self._make_dataframe(200, with_anomaly=True)
        result = self.validator.validate_candles(df, "BTC/USDT")
        anomalies = self.validator._check_price_anomalies(df)
        assert len(anomalies) > 0


class TestDatasetLineage:
    def test_to_dict(self):
        lineage = DatasetLineage(
            dataset_id="test_001",
            exchange="binance",
            pair="BTC/USDT",
            timeframe="5m",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            candle_count=8928,
            feature_version="v1.0",
        )
        d = lineage.to_dict()
        assert d["dataset_id"] == "test_001"
        assert "start_date" in d
        assert "end_date" in d
