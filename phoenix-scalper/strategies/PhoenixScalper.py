import talib.abstract as ta
from datetime import datetime
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, merge_informative_pair
from pandas import DataFrame
from functools import reduce
import numpy as np
import logging
import sys
import os
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


class PhoenixScalper(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = True
    can_short = True
    position_adjustment_enable = False

    minimal_roi = {"0": 10.0}

    stoploss = -0.12
    use_custom_stoploss = False

    trailing_stop = False

    buy_params = {
        "ema_fast": 7,
        "ema_slow": 21,
        "rsi_period": 5,
        "rsi_oversold": 38,
        "adx_threshold": 15,
        "volume_factor": 1.0,
        "short_lookback": 14,
        "short_volume_mult": 1.0,
        "short_adx_mult": 1.0,
        "short_rsi_threshold": 47,
        "sl_min": 0.003,
        "sl_max": 0.005,
        "grace_period": 10,
    }

    sell_params = {
        "rsi_overbought": 74,
        "hmm_default_target": 0.421,
        "hmm_range_target": 0.415,
        "hmm_bull_target": 0.732,
    }

    ema_fast = IntParameter(5, 10, default=7, space="buy")
    ema_slow = IntParameter(12, 22, default=15, space="buy")
    rsi_period = IntParameter(5, 10, default=7, space="buy")
    rsi_oversold = IntParameter(25, 40, default=30, space="buy")
    rsi_overbought = IntParameter(68, 78, default=72, space="sell")
    adx_threshold = IntParameter(15, 25, default=20, space="buy")
    volume_factor = DecimalParameter(1.0, 2.0, default=1.2, space="buy")

    atr_sl_mult = DecimalParameter(0.5, 1.2, default=0.7, space="buy")
    short_lookback = IntParameter(7, 15, default=10, space="buy")
    short_volume_mult = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    short_adx_mult = DecimalParameter(1.0, 1.5, default=1.3, space="buy")
    short_rsi_threshold = IntParameter(35, 55, default=45, space="buy")
    sl_min = DecimalParameter(0.0015, 0.0035, default=0.0025, space="buy")
    sl_max = DecimalParameter(0.0035, 0.0060, default=0.0050, space="buy")
    grace_period = IntParameter(3, 10, default=5, space="buy")
    hmm_default_target = DecimalParameter(0.35, 0.80, default=0.55, space="sell")
    hmm_range_target = DecimalParameter(0.25, 0.55, default=0.35, space="sell")
    hmm_bull_target = DecimalParameter(0.50, 1.20, default=0.80, space="sell")

    _hmm_cache = {}
    _kalman_cache = {}
    _hmm_update_count = 0

    protections = []

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        result = [(p, '15m') for p in pairs]
        result.append(("BTC/USDT:USDT", "1d"))
        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for period in range(5, 11):
            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        for period in list(range(5, 23)) + [50, 200]:
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

        pair_key = metadata.get('pair', 'default')
        do_heavy = (self._hmm_update_count % 12 == 0)

        try:
            if do_heavy or pair_key not in self._kalman_cache:
                from ml.kalman_filter import compute_kalman_features
                kf_features = compute_kalman_features(
                    dataframe['close'].values,
                    dataframe['atr_pct'].values,
                    dataframe['volume_ratio'].values
                )
                self._kalman_cache[pair_key] = kf_features
            else:
                kf_features = self._kalman_cache[pair_key]
            idx_len = len(dataframe)
            for col, values in kf_features.items():
                if len(values) != idx_len:
                    values = np.resize(values, idx_len)
                dataframe[col] = values
        except Exception as e:
            logger.warning(f"Kalman: {e}")
            if pair_key in self._kalman_cache:
                kf_features = self._kalman_cache[pair_key]
                idx_len = len(dataframe)
                for col, values in kf_features.items():
                    if len(values) != idx_len:
                        values = np.resize(values, idx_len)
                    dataframe[col] = values
            else:
                for col in ['kf_price','kf_trend','kf_prediction','kf_confidence',
                           'kf_direction','kf_innovation','kf_S','kf_price_momentum',
                           'kf_trend_acceleration','kf_prediction_error','kf_regime_score',
                           'kf_vol_of_trend','kf_atr_ratio']:
                    if col not in dataframe.columns:
                        dataframe[col] = 0.0

        try:
            if do_heavy or pair_key not in self._hmm_cache:
                from ml.hmm_regime import compute_hmm_features
                returns = np.log(dataframe['close'] / dataframe['close'].shift(1))
                vol = returns.rolling(10).std()
                volume_change = dataframe['volume_ratio'].pct_change()
                hmm_features = compute_hmm_features(returns.fillna(0), vol.fillna(0), volume_change.fillna(0))
                self._hmm_cache[pair_key] = hmm_features
            else:
                hmm_features = self._hmm_cache[pair_key]
            idx_len = len(dataframe)
            for col, values in hmm_features.items():
                if len(values) != idx_len:
                    values = np.resize(values, idx_len)
                dataframe[col] = values
        except Exception as e:
            logger.warning(f"HMM: {e}")
            if pair_key in self._hmm_cache:
                hmm_features = self._hmm_cache[pair_key]
                idx_len = len(dataframe)
                for col, values in hmm_features.items():
                    if len(values) != idx_len:
                        values = np.resize(values, idx_len)
                    dataframe[col] = values
            else:
                for col, val in {
                    'hmm_regime': 1, 'hmm_p_bull': 0.5, 'hmm_p_range': 0.3,
                    'hmm_p_bear': 0.2, 'hmm_regime_stability': 0.5,
                    'hmm_transition_risk': 0.3, 'hmm_vol_regime': 1.0,
                    'hmm_trend_strength': 1.0,
                }.items():
                    if col not in dataframe.columns:
                        dataframe[col] = val

        dataframe['hmm_target'] = self._compute_hmm_target(dataframe)
        dataframe['hmm_sl_pct'] = self._compute_hmm_sl(dataframe)

        self._hmm_update_count += 1

        return dataframe

    def _compute_hmm_target(self, dataframe: DataFrame) -> np.ndarray:
        n = len(dataframe)
        targets = np.full(n, self.hmm_default_target.value)
        bull_mask = (dataframe['hmm_p_bull'] > 0.7) & (dataframe['hmm_trend_strength'] > 1.2)
        range_mask = (dataframe['hmm_p_range'] > 0.5) | (
            (dataframe['hmm_p_bull'] < 0.3) & (dataframe['hmm_p_bear'] < 0.3)
        )
        stable_mask = dataframe['hmm_regime_stability'] < 0.3
        targets = np.where(bull_mask & stable_mask, self.hmm_bull_target.value, targets)
        targets = np.where(range_mask & stable_mask, self.hmm_range_target.value, targets)
        return targets

    def _compute_hmm_sl(self, dataframe: DataFrame) -> np.ndarray:
        atr_pct = dataframe['atr_pct'].values
        vol_regime = dataframe['hmm_vol_regime'].values
        sl_mult = self.atr_sl_mult.value * (0.8 + 0.4 * (vol_regime - 1.0))
        sl_pct = atr_pct * sl_mult
        sl_pct = np.clip(sl_pct, self.sl_min.value, self.sl_max.value)
        return sl_pct

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi = f"rsi_{self.rsi_period.value}"

        conditions_pullback = [
            dataframe["low"] <= dataframe[f"ema_{self.ema_slow.value}"] * 1.005,
            dataframe["close"] > dataframe[f"ema_{self.ema_slow.value}"],
            dataframe["close"] > dataframe["open"],
            dataframe[rsi] > 35,
            dataframe[rsi] < 55,
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["adx"] > self.adx_threshold.value,
            dataframe["plus_di"] > dataframe["minus_di"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_pullback),
            ["enter_long", "enter_tag"]
        ] = (1, "hmm_pullback")

        conditions_rsi = [
            dataframe[rsi] > 50,
            dataframe["close"] > dataframe["open"],
            dataframe["close"] > dataframe[f"ema_{self.ema_fast.value}"],
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["adx"] > self.adx_threshold.value,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_rsi),
            ["enter_long", "enter_tag"]
        ] = (1, "rsi_momentum")

        conditions_breakout = [
            dataframe["close"] > dataframe["high"].rolling(5).max().shift(1),
            dataframe["volume_ratio"] > self.volume_factor.value * 1.5,
            dataframe["adx"] > self.adx_threshold.value * 1.2,
            dataframe["close"] > dataframe[f"ema_{self.ema_slow.value}"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_breakout),
            ["enter_long", "enter_tag"]
        ] = (1, "momentum_breakout")

        conditions_kalman = [
            dataframe["kf_trend"] > 0,
            dataframe["kf_confidence"] > 0.6,
            dataframe["close"] > dataframe["kf_prediction"],
            dataframe["kf_trend_acceleration"] > 0,
            dataframe["close"] > dataframe[f"ema_{self.ema_fast.value}"],
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_kalman),
            ["enter_long", "enter_tag"]
        ] = (1, "kalman_cont")

        conditions_short_breakdown = [
            dataframe["close"] < dataframe["low"].rolling(self.short_lookback.value).min().shift(1),
            dataframe["volume_ratio"] > self.volume_factor.value * self.short_volume_mult.value,
            dataframe["adx"] > self.adx_threshold.value * self.short_adx_mult.value,
            dataframe["close"] < dataframe[f"ema_{self.ema_slow.value}"],
            dataframe["close"] < dataframe["open"],
            dataframe["close"] < dataframe["bb_lower"],
            dataframe["plus_di"] < dataframe["minus_di"],
            dataframe[f"rsi_{self.rsi_period.value}"] < self.short_rsi_threshold.value,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_short_breakdown),
            ["enter_short", "enter_tag"]
        ] = (1, "short_breakdown")

        conditions_short_rally = [
            dataframe["high"] >= dataframe[f"ema_{self.ema_slow.value}"] * 0.995,
            dataframe["close"] < dataframe[f"ema_{self.ema_slow.value}"],
            dataframe["close"] < dataframe["open"],
            dataframe[rsi] > 55,
            dataframe[rsi] < 75,
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["adx"] > self.adx_threshold.value,
            dataframe["plus_di"] < dataframe["minus_di"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_short_rally),
            ["enter_short", "enter_tag"]
        ] = (1, "short_rally_fail")

        conditions_short_bear = [
            dataframe["close"] < dataframe["open"],
            dataframe["close"] < dataframe["bb_lower"],
            dataframe["volume_ratio"] > self.volume_factor.value * 1.5,
            dataframe["adx"] > self.adx_threshold.value * 1.1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_short_bear),
            ["enter_short", "enter_tag"]
        ] = (1, "short_bear_momentum")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if current_profit >= 0.10:
            return "mc1_tp_10pct"
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 3600
        if elapsed > 1:
            return "max_hold_1h"
        return None

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        if after_fill:
            return -0.99
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 60
        if elapsed < 5:
            return -0.0040
        return -0.0024

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        if side == 'short':
            return 50.0
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and len(dataframe) > 0:
            last = dataframe.iloc[-1]
            hmm_p_bear = last.get('hmm_p_bear', 0.33)
            if hmm_p_bear > 0.3:
                return 50.0
        return 100.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str | None, side: str, **kwargs) -> float:
        wallet = self.wallets.get_total_stake_amount()
        kelly_75_pct = 0.177
        return max(min_stake, min(wallet * kelly_75_pct, max_stake))

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        return True

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        profit_pct = ((rate - trade.open_rate) / trade.open_rate) * 100 * trade.leverage
        duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
        direction = "SHORT" if trade.is_short else "LONG"

        result_line = f"+{profit_pct:.2f}%" if profit_pct > 0 else f"{profit_pct:.2f}%"
        dur_str = f"{int(duration_min)}m" if duration_min < 60 else f"{duration_min/60:.1f}h"

        msg = (
            f"*PHOENIX SCALPER - CLOSED*\n"
            f"{'='*25}\n"
            f"*{pair}* | {direction} | {trade.leverage}x\n"
            f"*Result:* *{result_line}*\n"
            f"*Duration:* {dur_str}\n"
            f"*Reason:* {exit_reason}\n"
            f"{'='*25}\n"
            f"_Scalper AI_"
        )
        self.dp.send_msg(msg, always_send=True)
        return True
