import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class DatasetLineage:
    dataset_id: str
    exchange: str
    pair: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    candle_count: int
    feature_version: str
    training_version: Optional[str] = None
    validation_version: Optional[str] = None
    checksum: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        return d


class DataValidator:
    def __init__(self, max_candle_age_minutes: int = 10, max_price_zscore: float = 5.0, on_notify=None):
        self.max_candle_age_minutes = max_candle_age_minutes
        self.max_price_zscore = max_price_zscore
        self.on_notify = on_notify

    def validate_candles(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        issues = []

        if dataframe is None or len(dataframe) == 0:
            return {"valid": False, "issues": ["Empty dataframe"], "severity": "fatal"}

        df = dataframe.copy()

        if "date" not in df.columns:
            return {"valid": False, "issues": ["No date column"], "severity": "fatal"}

        dates = pd.to_datetime(df["date"])
        df["datetime"] = dates

        missing_candles = self._check_missing_candles(dates)
        if missing_candles:
            issues.append(f"Missing candles: {len(missing_candles)} gaps detected")

        duplicates = df[df.duplicated(subset=["date"])]
        if len(duplicates) > 0:
            issues.append(f"Duplicate candles: {len(duplicates)} found")

        stale = self._check_stale_data(dates)
        if stale:
            issues.append(f"Stale data: last candle {stale} min old")

        nan_issues = self._check_nan_values(df, pair)
        issues.extend(nan_issues)

        price_anomalies = self._check_price_anomalies(df)
        if price_anomalies:
            issues.append(f"Price anomalies: {len(price_anomalies)} abnormal moves")

        volume_anomalies = self._check_volume_anomalies(df)
        if volume_anomalies:
            issues.append(f"Volume anomalies: {len(volume_anomalies)} spikes/drops")

        severity = "ok"
        if len(issues) > 5:
            severity = "critical"
        elif len(issues) > 2:
            severity = "warning"
        elif len(issues) > 0:
            severity = "info"

        if self.on_notify and issues and severity in ("warning", "critical"):
            for issue in issues[:3]:
                self.on_notify("data_quality", pair=pair, issue=issue.split(":")[0], details=issue)

        return {
            "valid": severity != "fatal",
            "issues": issues,
            "severity": severity,
            "candle_count": len(df),
            "date_range": f"{dates.min()} to {dates.max()}" if len(dates) > 0 else "unknown",
        }

    def _check_missing_candles(self, dates: pd.Series) -> List[str]:
        if len(dates) < 2:
            return []
        gaps = []
        diffs = dates.diff()
        threshold = pd.Timedelta(minutes=15)
        large_gaps = diffs[diffs > threshold]
        for idx in large_gaps.index:
            gaps.append(f"Gap at {dates[idx]}: {diffs[idx]}")
        return gaps[:5]

    def _check_stale_data(self, dates: pd.Series) -> Optional[float]:
        if len(dates) == 0:
            return None
        last_date = dates.iloc[-1]
        if last_date.tz is None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now(last_date.tz)
        age_min = (now - last_date).total_seconds() / 60
        if age_min > self.max_candle_age_minutes:
            return round(age_min, 1)
        return None

    def _check_nan_values(self, df: pd.DataFrame, pair: str) -> List[str]:
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
            if nan_count > 0:
                pct = nan_count / len(df) * 100
                if pct > 10:
                    issues.append(f"High NaN in {col}: {nan_count}/{len(df)} ({pct:.1f}%)")
            if inf_count > 0:
                issues.append(f"Infinity values in {col}: {inf_count}")
        return issues

    def _check_price_anomalies(self, df: pd.DataFrame) -> List[int]:
        if "close" not in df.columns:
            return []
        returns = df["close"].pct_change().dropna()
        if len(returns) < 20:
            return []
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        anomaly_indices = np.where(z_scores > self.max_price_zscore)[0]
        return anomaly_indices.tolist()[:10]

    def _check_volume_anomalies(self, df: pd.DataFrame) -> List[int]:
        if "volume" not in df.columns:
            return []
        volume = df["volume"].values
        if len(volume) < 20:
            return []
        median = np.median(volume)
        mad = np.median(np.abs(volume - median))
        if mad == 0:
            return []
        modified_z = 0.6745 * (volume - median) / mad
        anomaly_indices = np.where(np.abs(modified_z) > 5)[0]
        return anomaly_indices.tolist()[:10]

    def validate_indicator_warmup(self, dataframe: pd.DataFrame, startup_candle_count: int) -> Dict:
        if len(dataframe) < startup_candle_count:
            return {
                "valid": False,
                "available": len(dataframe),
                "required": startup_candle_count,
                "issue": f"Not enough data: {len(dataframe)} < {startup_candle_count} required",
            }
        return {
            "valid": True,
            "available": len(dataframe),
            "required": startup_candle_count,
            "issue": None,
        }
