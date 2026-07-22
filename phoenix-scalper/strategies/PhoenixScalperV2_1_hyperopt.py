from datetime import datetime
from functools import reduce
import logging
import sys
import os

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.regime_engine import RegimeEngine
from core.data_quality import DataValidator
from ml.kalman_filter import compute_kalman_features
from ml.hmm_regime import compute_hmm_features

logger = logging.getLogger(__name__)


class PhoenixScalperV2_1_Hyperopt(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = True
    can_short = True
    position_adjustment_enable = False
    minimal_roi = {"0": 10.0}
    stoploss = -0.12
    use_custom_stoploss = True
    trailing_stop = False
    LEVERAGE = 10
    max_open_trades = 5

    buy_params = {
        "adx_threshold": 30,
        "ema_fast": 7,
        "ema_slow": 14,
        "rsi_oversold": 22,
        "rsi_period": 6,
        "score_high_threshold": 60,
        "score_threshold": 51,
        "short_adx_mult": 1.258,
        "short_lookback": 13,
        "short_rsi_threshold": 46,
        "short_volume_mult": 1.931,
        "sl_max": 0.006,
        "sl_min": 0.003,
        "volume_factor": 1.003,
    }

    sell_params = {
        "bleed_loss": 0.033,
        "bleed_time": 289,
        "lock_ratio": 0.359,
        "max_hold_min": 303,
        "rsi_overbought": 74,
        "tp_target": 0.071,
        "trail_threshold": 0.078,
    }

    # --- BUY PARAMS (wider ranges for hyperopt) ---
    ema_fast = IntParameter(5, 12, default=8, space="buy")
    ema_slow = IntParameter(12, 24, default=18, space="buy")
    rsi_period = IntParameter(5, 12, default=8, space="buy")
    rsi_oversold = IntParameter(22, 40, default=30, space="buy")
    rsi_overbought = IntParameter(65, 78, default=72, space="sell")
    adx_threshold = IntParameter(18, 35, default=25, space="buy")
    volume_factor = DecimalParameter(0.6, 3.0, default=1.5, space="buy")
    short_lookback = IntParameter(7, 15, default=10, space="buy")
    short_volume_mult = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    short_adx_mult = DecimalParameter(1.0, 1.5, default=1.3, space="buy")
    short_rsi_threshold = IntParameter(30, 50, default=40, space="buy")
    sl_min = DecimalParameter(0.0015, 0.0035, default=0.0025, space="buy")
    sl_max = DecimalParameter(0.0035, 0.0060, default=0.0050, space="buy")
    score_threshold = IntParameter(40, 80, default=60, space="buy")
    score_high_threshold = IntParameter(55, 75, default=65, space="buy")

    # --- SELL/EXIT PARAMS (simple, wide stops - let trades breathe) ---
    trail_threshold = DecimalParameter(0.03, 0.08, default=0.05, space="sell")
    lock_ratio = DecimalParameter(0.30, 0.70, default=0.50, space="sell")
    tp_target = DecimalParameter(0.03, 0.08, default=0.05, space="sell")
    bleed_loss = DecimalParameter(0.02, 0.08, default=0.05, space="sell")
    bleed_time = IntParameter(120, 360, default=240, space="sell")
    max_hold_min = IntParameter(120, 720, default=360, space="sell")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._last_regime_str = "unknown"
        self._data_validator = DataValidator(max_candle_age_minutes=10)
        self._regime_engine = RegimeEngine()
        self._kalman_cache = {}
        self._hmm_cache = {}
        self._hmm_update_count = 0

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, '15m') for p in pairs] + [("BTC/USDT:USDT", "1d")]

    def _calculate_entry_score(self, dataframe: DataFrame) -> DataFrame:
        rsi_col = f"rsi_{self.rsi_period.value}"
        ema_f = f"ema_{self.ema_fast.value}"
        ema_s = f"ema_{self.ema_slow.value}"

        hmm_bull = np.minimum(dataframe["hmm_p_bull"].values / 0.6, 1.0) * 20
        trend_str = np.minimum(dataframe["adx"].values / 40.0, 1.0) * 15
        kalman_c = np.minimum(dataframe["kf_confidence"].values / 0.8, 1.0) * 10
        di_diff = dataframe["plus_di"].values - dataframe["minus_di"].values
        directional = np.clip(di_diff / 20.0 + 0.5, 0, 1) * 10
        mom_a = np.minimum(np.maximum(dataframe["kf_trend_acceleration"].values, 0), 1)
        mom_p = np.minimum(np.maximum(dataframe["kf_price_momentum"].values, 0), 1)
        momentum = (mom_a + mom_p) / 2.0 * 10
        volume = np.minimum(dataframe["volume_ratio"].values / 3.0, 1.0) * 10
        stability = (1.0 - np.minimum(dataframe["hmm_regime_stability"].values / 0.5, 1.0)) * 10
        rsi_v = dataframe[rsi_col].values
        rsi_s = np.maximum(0, 1.0 - np.abs(rsi_v - 45) / 25.0) * 5
        ema_dist = np.abs(dataframe["close"].values / dataframe[ema_s].values - 1.0)
        pullback = np.maximum(0, 1.0 - ema_dist / 0.02) * 5
        trend_al = (
            (dataframe["close"].values > dataframe[ema_f].values).astype(float) +
            (dataframe["close"].values > dataframe[ema_s].values).astype(float) +
            (dataframe["close"].values > dataframe["ema_50"].values).astype(float)
        ) / 3.0 * 5

        dataframe["signal_score"] = (
            hmm_bull + trend_str + kalman_c + directional +
            momentum + volume + stability + rsi_s + pullback + trend_al
        )

        hmm_bear = np.minimum(dataframe["hmm_p_bear"].values / 0.6, 1.0) * 20
        di_s = dataframe["minus_di"].values - dataframe["plus_di"].values
        directional_s = np.clip(di_s / 20.0 + 0.5, 0, 1) * 10
        mom_a_s = np.minimum(np.maximum(-dataframe["kf_trend_acceleration"].values, 0), 1)
        mom_p_s = np.minimum(np.maximum(-dataframe["kf_price_momentum"].values, 0), 1)
        momentum_s = (mom_a_s + mom_p_s) / 2.0 * 10
        rsi_s_s = np.maximum(0, 1.0 - np.abs(rsi_v - 70) / 20.0) * 5
        bk_dist = np.abs(dataframe["close"].values / dataframe[ema_s].values - 1.0)
        breakdown = np.maximum(0, 1.0 - bk_dist / 0.02) * 5
        bear_al = (
            (dataframe["close"].values < dataframe[ema_f].values).astype(float) +
            (dataframe["close"].values < dataframe[ema_s].values).astype(float) +
            (dataframe["close"].values < dataframe["ema_50"].values).astype(float)
        ) / 3.0 * 5

        dataframe["short_score"] = (
            hmm_bear + trend_str + kalman_c + directional_s +
            momentum_s + volume + stability + rsi_s_s + breakdown + bear_al
        )
        return dataframe

    def _align_array(self, values: np.ndarray, target_len: int) -> np.ndarray:
        if len(values) == target_len:
            return values
        if len(values) < target_len:
            padded = np.full(target_len, np.nan)
            padded[:len(values)] = values
            if len(values) > 0:
                padded[len(values):] = values[-1]
            return padded
        return values[-target_len:]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair_key = metadata.get('pair', 'default')
        self._data_validator.validate_candles(dataframe, pair_key)

        for period in range(5, 13):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        for period in list(range(5, 26)) + [50, 200]:
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=17, signalperiod=5)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_middle"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (dataframe["bb_middle"] + 1e-10)
        dataframe["bb_width_sma"] = ta.SMA(dataframe["bb_width"], timeperiod=20)
        dataframe["volume_ema"] = ta.EMA(dataframe["volume"], timeperiod=10)
        dataframe["volume_ratio"] = dataframe["volume"] / (dataframe["volume_ema"] + 1e-10)
        dataframe["vwap"] = (dataframe["close"] * dataframe["volume"]).rolling(20).sum() / dataframe["volume"].rolling(20).sum()
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_ema"] = ta.EMA(dataframe["obv"], timeperiod=10)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / (dataframe["close"] + 1e-10)

        try:
            if pair_key not in self._kalman_cache:
                kf_features = compute_kalman_features(
                    dataframe['close'].values, dataframe['atr_pct'].values, dataframe['volume_ratio'].values
                )
                self._kalman_cache[pair_key] = kf_features
            else:
                kf_features = self._kalman_cache[pair_key]
            idx_len = len(dataframe)
            for col, values in kf_features.items():
                aligned = self._align_array(values, idx_len)
                dataframe[col] = aligned
        except Exception as e:
            logger.warning(f"KF: {e}")
            if pair_key in self._kalman_cache:
                kf_features = self._kalman_cache[pair_key]
                for col, values in kf_features.items():
                    dataframe[col] = self._align_array(values, len(dataframe))
            else:
                for col in ['kf_price','kf_trend','kf_prediction','kf_confidence',
                            'kf_direction','kf_innovation','kf_S','kf_price_momentum',
                            'kf_trend_acceleration','kf_prediction_error','kf_regime_score',
                            'kf_vol_of_trend','kf_atr_ratio']:
                    if col not in dataframe.columns:
                        dataframe[col] = 0.0

        try:
            if pair_key not in self._hmm_cache:
                returns = np.log(dataframe['close'] / dataframe['close'].shift(1))
                vol = returns.rolling(10).std()
                volume_change = dataframe['volume_ratio'].pct_change()
                hmm_features = compute_hmm_features(
                    returns.fillna(0).values, vol.fillna(0).values, volume_change.fillna(0).values
                )
                self._hmm_cache[pair_key] = hmm_features
            else:
                hmm_features = self._hmm_cache[pair_key]
            for col, values in hmm_features.items():
                dataframe[col] = self._align_array(values, len(dataframe))
        except Exception as e:
            logger.warning(f"HMM: {e}")
            if pair_key in self._hmm_cache:
                hmm_features = self._hmm_cache[pair_key]
                for col, values in hmm_features.items():
                    dataframe[col] = self._align_array(values, len(dataframe))
            else:
                for col, val in {
                    'hmm_regime': 1, 'hmm_p_bull': 0.5, 'hmm_p_range': 0.3,
                    'hmm_p_bear': 0.2, 'hmm_regime_stability': 0.5,
                    'hmm_transition_risk': 0.3, 'hmm_vol_regime': 1.0, 'hmm_trend_strength': 1.0,
                }.items():
                    if col not in dataframe.columns:
                        dataframe[col] = val

        dataframe['hmm_target'] = 0.5
        dataframe['hmm_sl_pct'] = self.sl_min.value

        self._hmm_update_count += 1

        try:
            regime_result = self._regime_engine.analyze(dataframe)
            if metadata.get("pair", "").startswith("BTC/"):
                self._last_regime_str = regime_result.regime.value
        except Exception as e:
            logger.warning(f"Regime: {e}")

        dataframe = self._calculate_entry_score(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi = f"rsi_{self.rsi_period.value}"
        ema_f = f"ema_{self.ema_fast.value}"
        ema_s = f"ema_{self.ema_slow.value}"
        adx = self.adx_threshold.value
        vf = self.volume_factor.value
        rc = self.rsi_oversold.value
        st = self.short_rsi_threshold.value
        lm = self.short_lookback.value
        vm = self.short_volume_mult.value
        am = self.short_adx_mult.value

        cond_pull = [
            dataframe["low"] <= dataframe[ema_s] * 1.005,
            dataframe["close"] > dataframe[ema_s],
            dataframe["close"] > dataframe["open"],
            dataframe[rsi] > 35, dataframe[rsi] < 55,
            dataframe["volume_ratio"] > vf,
            dataframe["adx"] > adx,
            dataframe["plus_di"] > dataframe["minus_di"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_pull), ["enter_long","enter_tag"]] = (1, "pullback")

        cond_rsi = [
            dataframe[rsi] > 50, dataframe["close"] > dataframe["open"],
            dataframe["close"] > dataframe[ema_f],
            dataframe["volume_ratio"] > vf,
            dataframe["adx"] > adx, dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_rsi), ["enter_long","enter_tag"]] = (1, "rsi_momentum")

        cond_bo = [
            dataframe["close"] > dataframe["high"].rolling(5).max().shift(1),
            dataframe["volume_ratio"] > vf * 1.5,
            dataframe["adx"] > adx * 1.2,
            dataframe["close"] > dataframe[ema_s], dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_bo), ["enter_long","enter_tag"]] = (1, "momentum_breakout")

        cond_kf = [
            dataframe["kf_trend"] > 0, dataframe["kf_confidence"] > 0.6,
            dataframe["close"] > dataframe["kf_prediction"],
            dataframe["kf_trend_acceleration"] > 0,
            dataframe["close"] > dataframe[ema_f],
            dataframe["volume_ratio"] > vf, dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_kf), ["enter_long","enter_tag"]] = (1, "kalman_cont")

        cond_sb = [
            dataframe["close"] < dataframe["low"].rolling(lm).min().shift(1),
            dataframe["volume_ratio"] > vf * vm,
            dataframe["adx"] > adx * am,
            dataframe["close"] < dataframe[ema_s],
            dataframe["close"] < dataframe["open"],
            dataframe["plus_di"] < dataframe["minus_di"],
            dataframe[rsi] < st, dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_sb), ["enter_short","enter_tag"]] = (1, "short_breakdown")

        cond_sr = [
            dataframe["high"] >= dataframe[ema_s] * 0.995,
            dataframe["close"] < dataframe[ema_s],
            dataframe["close"] < dataframe["open"],
            dataframe[rsi] > 55, dataframe[rsi] < 75,
            dataframe["volume_ratio"] > vf,
            dataframe["adx"] > adx,
            dataframe["plus_di"] < dataframe["minus_di"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_sr), ["enter_short","enter_tag"]] = (1, "short_rally_fail")

        cond_bm = [
            dataframe["close"] < dataframe["open"],
            dataframe["volume_ratio"] > vf * 1.3,
            dataframe["adx"] > adx * 1.1,
            dataframe["plus_di"] < dataframe["minus_di"],
            dataframe[rsi] < 45,
            dataframe["close"] < dataframe[ema_s],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda x,y: x&y, cond_bm), ["enter_short","enter_tag"]] = (1, "short_bear_momentum")

        dataframe.loc[dataframe["signal_score"] > 58, "signal_score"] = 58.0
        dataframe.loc[dataframe["short_score"] > 58, "short_score"] = 58.0
        threshold = self.score_threshold.value
        dataframe.loc[(dataframe["enter_long"]==1)&(dataframe["signal_score"]<threshold), ["enter_long","enter_tag"]] = (0, None)
        dataframe.loc[(dataframe["enter_short"]==1)&(dataframe["short_score"]<threshold), ["enter_short","enter_tag"]] = (0, None)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 60
        if current_profit >= self.tp_target.value:
            return "tp_hit"
        if current_profit < -self.bleed_loss.value and elapsed > self.bleed_time.value:
            return "bleed_exit"
        if elapsed > self.max_hold_min.value:
            return "max_hold"
        return None

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        if after_fill:
            return -0.99
        lev = trade.leverage
        if current_profit > self.trail_threshold.value:
            lock_equity = max(current_profit * self.lock_ratio.value, self.trail_threshold.value * 0.5)
            return -(lock_equity / lev)
        return self.stoploss

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(self.LEVERAGE)
