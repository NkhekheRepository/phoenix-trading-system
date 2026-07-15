import numpy as np
import pandas as pd
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Regime(Enum):
    STRONG_BULL = "strong_bull"
    WEAK_BULL = "weak_bull"
    SIDEWAYS = "sideways"
    WEAK_BEAR = "weak_bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

    def is_bullish(self) -> bool:
        return self in (Regime.STRONG_BULL, Regime.WEAK_BULL)

    def is_bearish(self) -> bool:
        return self in (Regime.STRONG_BEAR, Regime.WEAK_BEAR)

    def is_risk_on(self) -> bool:
        return self in (Regime.STRONG_BULL, Regime.LOW_VOLATILITY)

    def is_risk_off(self) -> bool:
        return self in (Regime.STRONG_BEAR, Regime.HIGH_VOLATILITY)


@dataclass
class RegimeResult:
    regime: Regime
    confidence: float
    recommended_mode: str
    transition_probability: Dict[Regime, float]
    expected_persistence: int
    signals: Dict[str, float]


class RegimeEngine:
    def __init__(self, on_notify=None):
        self._history: list[RegimeResult] = []
        self._last_regime: Optional[str] = None
        self.on_notify = on_notify

    def analyze(self, dataframe: pd.DataFrame) -> RegimeResult:
        if dataframe is None or len(dataframe) < 50:
            return self._default_result()

        df = dataframe.copy()
        last = df.iloc[-1]

        trend_signal = self._score_trend(df)
        vol_signal = self._score_volatility(df)
        volume_signal = self._score_volume(df)
        momentum_signal = self._score_momentum(df)
        hmm_signal = self._score_hmm(df)

        raw_score = (
            0.30 * trend_signal +
            0.20 * vol_signal +
            0.15 * volume_signal +
            0.20 * momentum_signal +
            0.15 * hmm_signal
        )

        vol_level = last.get("bb_width", 0) if not pd.isna(last.get("bb_width", 0)) else 0
        is_high_vol = vol_level > (last.get("bb_width_sma", vol_level) * 1.5) if not pd.isna(last.get("bb_width_sma", vol_level)) else False
        is_low_vol = vol_level < (last.get("bb_width_sma", vol_level) * 0.5) if not pd.isna(last.get("bb_width_sma", vol_level)) else False

        if is_high_vol:
            if raw_score < -0.2:
                regime = Regime.STRONG_BEAR
            elif raw_score > 0.2:
                regime = Regime.STRONG_BULL
            else:
                regime = Regime.HIGH_VOLATILITY
        elif is_low_vol and abs(raw_score) < 0.3:
            regime = Regime.LOW_VOLATILITY
        elif raw_score > 0.6:
            regime = Regime.STRONG_BULL
        elif raw_score > 0.2:
            regime = Regime.WEAK_BULL
        elif raw_score > -0.2:
            regime = Regime.SIDEWAYS
        elif raw_score > -0.6:
            regime = Regime.WEAK_BEAR
        else:
            regime = Regime.STRONG_BEAR

        confidence = min(0.5 + abs(raw_score) * 0.5, 0.95)

        persistence = self._estimate_persistence(regime)

        transitions = self._get_transition_probs(regime, raw_score)

        mode = self._recommend_mode(regime, confidence)

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            recommended_mode=mode,
            transition_probability=transitions,
            expected_persistence=persistence,
            signals={
                "trend": trend_signal,
                "volatility": vol_signal,
                "volume": volume_signal,
                "momentum": momentum_signal,
                "hmm": hmm_signal,
                "composite": raw_score,
            },
        )

        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        old = self._last_regime
        new_str = regime.value
        if old is not None and old != new_str and self.on_notify:
            self.on_notify("regime_change", old=old, new=new_str, confidence=confidence, mode=mode, signals=result.signals)
        self._last_regime = new_str

        return result

    def _score_trend(self, df: pd.DataFrame) -> float:
        ema_fast = df.get("ema_7", df.get("ema_5"))
        ema_slow = df.get("ema_21", df.get("ema_20"))
        adx = df.get("adx")

        if ema_fast is None or ema_slow is None:
            return 0.0

        last = df.iloc[-1]
        prev = df.iloc[-5] if len(df) > 5 else df.iloc[0]

        ema_fast_val = last.get("ema_7", last.get("ema_5", 0))
        ema_slow_val = last.get("ema_21", last.get("ema_20", 0))
        ema_prev = prev.get("ema_7", prev.get("ema_5", 0))

        trend = 0.0
        if ema_slow_val != 0:
            trend += np.clip((ema_fast_val - ema_slow_val) / ema_slow_val * 50, -1, 1)

        if ema_prev != 0:
            ema_slope = (ema_fast_val - ema_prev) / ema_prev * 100
            trend += np.clip(ema_slope * 2, -1, 1)

        if adx is not None:
            adx_val = last.get("adx", 0)
            if not pd.isna(adx_val):
                trend *= min(adx_val / 25, 1.0)

        return np.clip(trend * 0.5, -1, 1)

    def _score_volatility(self, df: pd.DataFrame) -> float:
        atr_pct = df.get("atr_pct")
        bb_width = df.get("bb_width")
        bb_width_sma = df.get("bb_width_sma")

        if atr_pct is None and bb_width is None:
            return 0.0

        last = df.iloc[-1]
        vol_score = 0.0

        if atr_pct is not None:
            atr_val = last.get("atr_pct", 0)
            if not pd.isna(atr_val):
                atr_mean = df["atr_pct"].mean() if len(df) > 0 else atr_val
                if atr_mean > 0:
                    vol_score += np.clip((atr_val - atr_mean) / atr_mean * 2, -1, 1)

        if bb_width is not None and bb_width_sma is not None:
            bb_val = last.get("bb_width", 0)
            bb_sma_val = last.get("bb_width_sma", bb_val)
            if not pd.isna(bb_val) and not pd.isna(bb_sma_val) and bb_sma_val > 0:
                vol_score += np.clip((bb_val - bb_sma_val) / bb_sma_val, -1, 1)

        return np.clip(vol_score * 0.5, -1, 1)

    def _score_volume(self, df: pd.DataFrame) -> float:
        vol_ratio = df.get("volume_ratio")
        if vol_ratio is None:
            return 0.0

        ratios = df["volume_ratio"].values
        recent = ratios[-5:] if len(ratios) >= 5 else ratios
        median_ratio = np.median(recent)

        score = np.clip((median_ratio - 1.0) * 2, -1, 1)
        return score

    def _score_momentum(self, df: pd.DataFrame) -> float:
        rsi_cols = [c for c in df.columns if c.startswith("rsi_")]
        if not rsi_cols:
            return 0.0

        last = df.iloc[-1]
        scores = []

        for col in rsi_cols:
            val = last.get(col)
            if val is not None and not pd.isna(val):
                normalized = (val - 50) / 50
                scores.append(normalized)

        if not scores:
            return 0.0

        macd = last.get("macd")
        macd_signal = last.get("macdsignal")
        if macd is not None and macd_signal is not None and not pd.isna(macd) and not pd.isna(macd_signal):
            macd_score = np.clip((macd - macd_signal) * 100, -1, 1)
            scores.append(macd_score * 0.5)

        plus_di = last.get("plus_di")
        minus_di = last.get("minus_di")
        if plus_di is not None and minus_di is not None and not pd.isna(plus_di) and not pd.isna(minus_di):
            di_score = np.clip((plus_di - minus_di) / 50, -1, 1)
            scores.append(di_score * 0.5)

        return np.clip(np.mean(scores), -1, 1)

    def _score_hmm(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        p_bull = last.get("hmm_p_bull")
        p_bear = last.get("hmm_p_bear")

        if p_bull is None or p_bear is None or pd.isna(p_bull) or pd.isna(p_bear):
            return 0.0

        return np.clip(p_bull - p_bear, -1, 1)

    def _estimate_persistence(self, regime: Regime) -> int:
        persistence = {
            Regime.STRONG_BULL: 12,
            Regime.WEAK_BULL: 6,
            Regime.SIDEWAYS: 8,
            Regime.WEAK_BEAR: 6,
            Regime.STRONG_BEAR: 12,
            Regime.HIGH_VOLATILITY: 4,
            Regime.LOW_VOLATILITY: 8,
        }
        return persistence.get(regime, 6)

    def _get_transition_prob(self, from_regime: Regime, to_regime: Regime) -> float:
        transitions = {
            (Regime.STRONG_BULL, Regime.WEAK_BULL): 0.15,
            (Regime.STRONG_BULL, Regime.SIDEWAYS): 0.05,
            (Regime.WEAK_BULL, Regime.STRONG_BULL): 0.20,
            (Regime.WEAK_BULL, Regime.SIDEWAYS): 0.15,
            (Regime.WEAK_BULL, Regime.WEAK_BEAR): 0.05,
            (Regime.SIDEWAYS, Regime.WEAK_BULL): 0.15,
            (Regime.SIDEWAYS, Regime.WEAK_BEAR): 0.15,
            (Regime.WEAK_BEAR, Regime.SIDEWAYS): 0.15,
            (Regime.WEAK_BEAR, Regime.STRONG_BEAR): 0.20,
            (Regime.WEAK_BEAR, Regime.WEAK_BULL): 0.05,
            (Regime.STRONG_BEAR, Regime.WEAK_BEAR): 0.15,
            (Regime.STRONG_BEAR, Regime.SIDEWAYS): 0.05,
        }
        return transitions.get((from_regime, to_regime), 0.02)

    def _get_transition_probs(self, current: Regime, score: float) -> Dict[Regime, float]:
        probs = {}
        for regime in Regime:
            prob = self._get_transition_prob(current, regime)
            if abs(score) > 0.5:
                if regime.is_bullish() and score > 0:
                    prob *= 1.5
                elif regime.is_bearish() and score < 0:
                    prob *= 1.5
            probs[regime] = min(prob, 1.0)
        total = sum(probs.values())
        if total > 0:
            for r in probs:
                probs[r] /= total
        return probs

    def _recommend_mode(self, regime: Regime, confidence: float) -> str:
        if regime in (Regime.STRONG_BULL, Regime.STRONG_BEAR) and confidence > 0.8:
            return "aggressive"
        elif regime in (Regime.WEAK_BULL, Regime.WEAK_BEAR):
            return "normal"
        elif regime == Regime.HIGH_VOLATILITY:
            return "capital_preservation"
        elif regime == Regime.LOW_VOLATILITY:
            return "scalping"
        return "neutral"

    def _default_result(self) -> RegimeResult:
        return RegimeResult(
            regime=Regime.SIDEWAYS,
            confidence=0.0,
            recommended_mode="neutral",
            transition_probability={r: 1/7 for r in Regime},
            expected_persistence=6,
            signals={"trend": 0, "volatility": 0, "volume": 0, "momentum": 0, "hmm": 0, "composite": 0},
        )

    def get_regime_history(self, n: int = 50) -> list[RegimeResult]:
        return self._history[-n:] if self._history else []
