import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngine:
    def __init__(self, forward_bars: int = 24, min_samples: int = 50000):
        self.forward_bars = forward_bars
        self.min_samples = min_samples
        self.feature_cols = None

    def generate_training_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating training data from {len(dataframe)} candles...")
        samples = []
        processed_count = 0

        for i in range(200, len(dataframe) - self.forward_bars):
            features = self._extract_features(dataframe, i)
            entry_price = dataframe.iloc[i]['close']
            future = dataframe.iloc[i+1:i+self.forward_bars+1]

            max_profit = (future['high'].max() - entry_price) / entry_price
            max_drawdown = (entry_price - future['low'].min()) / entry_price

            target = self._create_target(max_profit, max_drawdown, entry_price, future)
            features['target'] = target
            samples.append(features)
            processed_count += 1

            if len(samples) >= self.min_samples:
                break

        logger.info(f"Generated {len(samples)} training samples from {processed_count} entry points")
        result_df = pd.DataFrame(samples)
        self.feature_cols = [col for col in result_df.columns if col != 'target']
        return result_df

    def _extract_features(self, dataframe: pd.DataFrame, idx: int) -> Dict:
        row = dataframe.iloc[idx]
        close_val = row['close']
        open_val = row['open']
        high_val = row['high']
        low_val = row['low']

        candle_range = (high_val - low_val) / (close_val + 1e-10)
        body = abs(close_val - open_val) / (high_val - low_val + 1e-10)
        close_open_ratio = close_val / (open_val + 1e-10)
        bullish_candle = 1 if close_val > open_val else 0
        close_gt_open_pct = (close_val - open_val) / (open_val + 1e-10)

        features = {
            "weekday": row.get('weekday', 0),
            "hour": row.get('hour', 0),
            "trade_duration": 0,
            "leverage": 30.0,
            "open_rate": close_val,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "rsi_14": row.get('rsi_14', 50),
            "adx": row.get('adx', 20),
            "plus_di": row.get('plus_di', 20),
            "minus_di": row.get('minus_di', 20),
            "macd": row.get('macd', 0),
            "macdsignal": row.get('macdsignal', 0),
            "macdhist": row.get('macdhist', 0),
            "bb_upper": row.get('bb_upper', 0),
            "bb_middle": row.get('bb_middle', 0),
            "bb_lower": row.get('bb_lower', 0),
            "bb_width": row.get('bb_width', 0),
            "bb_position": row.get('bb_position', 0.5),
            "volume_ema": row.get('volume_ema', 0),
            "volume_ratio": row.get('volume_ratio', 1.0),
            "obv": row.get('obv', 0),
            "obv_ema": row.get('obv_ema', 0),
            "atr": row.get('atr', 0),
            "atr_pct": row.get('atr', 0) / (close_val + 1e-10),
            "ema_50": row.get('ema_50', 0),
            "ema_200": row.get('ema_200', 0),
            "is_bull_5m": row.get('is_bull', 0),
            "candle_range": candle_range,
            "candle_body": body,
            "close_open_ratio": close_open_ratio,
            "is_bull_1h": row.get('is_bull_1h', 1),
            "rsi_14_1h": row.get('rsi_14_1h', 50),
            "adx_1h": row.get('adx_1h', 20),
            "is_bull_4h": row.get('is_bull_4h', 1),
            "di_spread": row.get('plus_di', 20) - row.get('minus_di', 20),
            "macd_hist_sign": 1 if row.get('macdhist', 0) > 0 else 0,
            "volume_spike": 1 if row.get('volume_ratio', 1.0) > 1.5 else 0,
            "ema_aligned": 1 if row.get('is_bull', 0) == 1 else 0,
            "bullish_candle": bullish_candle,
            "close_gt_open_pct": close_gt_open_pct,
            "pair_enc": 0,
            "tag_enc": 0,
            "kf_price": row.get('kf_price', close_val),
            "kf_trend": row.get('kf_trend', 0.0),
            "kf_prediction": row.get('kf_prediction', close_val),
            "kf_confidence": row.get('kf_confidence', 0.5),
            "kf_direction": row.get('kf_direction', 0),
            "kf_innovation": row.get('kf_innovation', 0.0),
            "kf_S": row.get('kf_S', 0.01),
            "kf_price_momentum": row.get('kf_price_momentum', 0.0),
            "kf_trend_acceleration": row.get('kf_trend_acceleration', 0.0),
            "kf_prediction_error": row.get('kf_prediction_error', 0.0),
            "kf_regime_score": row.get('kf_regime_score', 0.0),
            "kf_vol_of_trend": row.get('kf_vol_of_trend', 0.0),
            "kf_atr_ratio": row.get('kf_atr_ratio', 0.0),
            "hmm_regime": row.get('hmm_regime', 1),
            "hmm_p_bull": row.get('hmm_p_bull', 0.5),
            "hmm_p_range": row.get('hmm_p_range', 0.3),
            "hmm_p_bear": row.get('hmm_p_bear', 0.2),
            "hmm_regime_stability": row.get('hmm_regime_stability', 0.5),
            "hmm_transition_risk": row.get('hmm_transition_risk', 0.1),
            "hmm_vol_regime": row.get('hmm_vol_regime', 1.0),
            "hmm_trend_strength": row.get('hmm_trend_strength', 1.0),
        }
        return features

    def _create_target(self, max_profit: float, max_drawdown: float,
                       entry_price: float, future: pd.DataFrame) -> int:
        ROI_THRESHOLD = 0.03
        STOPLOSS_THRESHOLD = 0.02

        if max_profit >= ROI_THRESHOLD:
            return 1
        if max_drawdown >= STOPLOSS_THRESHOLD:
            return 0

        final_price = future.iloc[-1]['close']
        final_return = (final_price - entry_price) / entry_price
        return 1 if final_return > 0 else 0

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        constant_features = []
        for col in df.columns:
            if col != 'target' and df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"Removing constant features: {constant_features}")
            df = df.drop(columns=constant_features)

        df = df.fillna(0)
        X = df.drop(columns=['target'])
        y = df['target']
        return X, y

    def get_feature_names(self) -> List[str]:
        if self.feature_cols is None:
            raise ValueError("Feature columns not set. Run generate_training_data first.")
        return self.feature_cols
