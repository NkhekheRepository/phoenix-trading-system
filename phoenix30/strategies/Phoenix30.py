import talib.abstract as ta
from datetime import datetime
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, merge_informative_pair
from pandas import DataFrame
from functools import reduce
import numpy as np
import logging
import joblib
import os
import sys
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "models", "winrate_model.pkl")


class Phoenix30(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 200
    process_only_new_candles = True
    can_short = False
    position_adjustment_enable = False

    minimal_roi = {"0": 0.27}

    stoploss = -0.99
    use_custom_stoploss = False

    trailing_stop = False

    buy_params = {
        "adx_threshold": 20,
        "ema_fast": 13,
        "ema_slow": 25,
        "rsi_bounce": 35,
        "rsi_period": 14,
        "rsi_pullback_high": 55,
        "rsi_pullback_low": 39,
        "volume_factor": 1.086,
        "ml_threshold": 0.30,
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 2.0,
        "atr_trail_activate": 1.5,
        "atr_trail_dist": 0.5,
    }

    sell_params = {
        "rsi_exit": 78,
    }

    ema_fast = IntParameter(5, 15, default=9, space="buy")
    ema_slow = IntParameter(15, 30, default=21, space="buy")
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_pullback_low = IntParameter(30, 48, default=40, space="buy")
    rsi_pullback_high = IntParameter(52, 65, default=58, space="buy")
    rsi_bounce = IntParameter(25, 35, default=30, space="buy")
    rsi_exit = IntParameter(72, 85, default=78, space="sell")
    adx_threshold = IntParameter(20, 35, default=25, space="buy")
    volume_factor = DecimalParameter(1.0, 2.5, default=1.3, space="buy")
    ml_threshold = DecimalParameter(0.3, 0.9, default=0.30, space="buy")

    protections = [
        {"method": "CooldownPeriod", "stop_duration": 20},
        {"method": "StoplossGuard", "lookback_period": 720, "trade_limit": 3,
         "stop_duration": 60, "only_per_pair": False},
        {"method": "MaxDrawdown", "lookback_period": 1440, "max_allowed_drawdown": 0.10,
         "stop_duration": 300, "trade_limit": 5},
        {"method": "LowProfitPairs", "lookback_period": 7, "min_profit_percentage": 0.02,
         "max_loss_percentage": -0.03, "refresh_period": 1440},
    ]

    _model_meta = None

    def __init__(self, config=None):
        super().__init__(config)
        self._regime_bull = True
        self._last_regime_check = None
        self._hmm_cache = {}
        self._kalman_cache = {}
        self._hmm_update_count = 0

        model_path = MODEL_PATH
        if os.path.exists(model_path):
            try:
                self._model_meta = joblib.load(model_path)
                logger.info(f"ML model loaded ({len(self._model_meta['feature_cols'])} features, threshold={self._model_meta['threshold']:.2f})")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
                self._model_meta = None

        try:
            from ml.regime_adaptive import RegimeAdaptiveExit
            self._regime_adaptive_exit = RegimeAdaptiveExit()
        except Exception as e:
            logger.warning(f"Failed to initialize regime-adaptive exit: {e}")
            self._regime_adaptive_exit = None

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        result = [(p, '1h') for p in pairs] + [(p, '4h') for p in pairs]
        result.append(("BTC/USDT:USDT", "1d"))
        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f"ema_{self.ema_slow.value}"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        rsi_period = self.rsi_period.value
        dataframe[f"rsi_{rsi_period}"] = ta.RSI(dataframe, timeperiod=rsi_period)
        if rsi_period != 14:
            dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        else:
            dataframe["rsi_14"] = dataframe[f"rsi_{rsi_period}"]

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_middle"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (dataframe["bb_middle"] + 1e-10)
        dataframe["bb_width_sma"] = ta.SMA(dataframe["bb_width"], timeperiod=50)
        dataframe["bb_position"] = (dataframe["close"] - dataframe["bb_lower"]) / (dataframe["bb_upper"] - dataframe["bb_lower"] + 1e-10)

        dataframe["volume_ema"] = ta.EMA(dataframe["volume"], timeperiod=20)
        dataframe["volume_ratio"] = dataframe["volume"] / (dataframe["volume_ema"] + 1e-10)

        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_ema"] = ta.EMA(dataframe["obv"], timeperiod=20)

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / (dataframe["close"] + 1e-10)

        dataframe["is_bull"] = (
            (dataframe["close"] > dataframe["ema_200"]) &
            (dataframe["ema_50"] > dataframe["ema_200"])
        ).astype(int)

        dataframe["is_bear"] = (
            (dataframe["close"] < dataframe["ema_200"]) &
            (dataframe["ema_50"] < dataframe["ema_200"])
        ).astype(int)

        ema_slow_key = f"ema_{self.ema_slow.value}"
        if ema_slow_key in dataframe.columns:
            dataframe["pullback_to_ema"] = (
                (dataframe["low"] <= dataframe[ema_slow_key] * 1.02) &
                (dataframe["close"] > dataframe[ema_slow_key]) &
                (dataframe["close"] > dataframe["open"])
            ).astype(int)
        else:
            dataframe["pullback_to_ema"] = 0

        if self.dp:
            df_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            if len(df_1h) > 0:
                df_1h['ema_50'] = ta.EMA(df_1h, timeperiod=50)
                df_1h['ema_200'] = ta.EMA(df_1h, timeperiod=200)
                df_1h['rsi_14'] = ta.RSI(df_1h, timeperiod=14)
                df_1h['adx'] = ta.ADX(df_1h, timeperiod=14)
                df_1h['is_bull'] = (
                    (df_1h['close'] > df_1h['ema_200']) &
                    (df_1h['ema_50'] > df_1h['ema_200'])
                ).astype(int)
                dataframe = merge_informative_pair(
                    dataframe,
                    df_1h[['date', 'ema_50', 'ema_200', 'rsi_14', 'adx', 'is_bull']],
                    self.timeframe, '1h', ffill=True
                )
            else:
                dataframe['ema_50_1h'] = 0
                dataframe['ema_200_1h'] = 0
                dataframe['rsi_14_1h'] = 50
                dataframe['adx_1h'] = 0
                dataframe['is_bull_1h'] = 0

            df_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')
            if len(df_4h) > 0:
                df_4h['ema_200'] = ta.EMA(df_4h, timeperiod=200)
                df_4h['ema_50'] = ta.EMA(df_4h, timeperiod=50)
                df_4h['is_bull'] = (
                    (df_4h['close'] > df_4h['ema_200']) &
                    (df_4h['ema_50'] > df_4h['ema_200'])
                ).astype(int)
                dataframe = merge_informative_pair(
                    dataframe,
                    df_4h[['date', 'ema_200', 'ema_50', 'is_bull']],
                    self.timeframe, '4h', ffill=True
                )
            else:
                dataframe['ema_200_4h'] = 0
                dataframe['is_bull_4h'] = 0

        for col, default in [
            ('is_bull_1h', 1), ('rsi_14_1h', 50), ('adx_1h', 20),
            ('ema_200_4h', 0), ('is_bull_4h', 1),
        ]:
            if col not in dataframe.columns:
                dataframe[col] = default

        pair_key = metadata.get('pair', 'default')
        now = _time.time()
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
            logger.warning(f"Failed to compute Kalman features: {e}")
            if pair_key in self._kalman_cache:
                kf_features = self._kalman_cache[pair_key]
                idx_len = len(dataframe)
                for col, values in kf_features.items():
                    if len(values) != idx_len:
                        values = np.resize(values, idx_len)
                    dataframe[col] = values
            else:
                for col in ['kf_price', 'kf_trend', 'kf_prediction', 'kf_confidence',
                           'kf_direction', 'kf_innovation', 'kf_S', 'kf_price_momentum',
                           'kf_trend_acceleration', 'kf_prediction_error', 'kf_regime_score',
                           'kf_vol_of_trend', 'kf_atr_ratio']:
                    if col not in dataframe.columns:
                        dataframe[col] = 0.0

        try:
            if do_heavy or pair_key not in self._hmm_cache:
                from ml.hmm_regime import compute_hmm_features
                returns = np.log(dataframe['close'] / dataframe['close'].shift(1))
                vol = returns.rolling(20).std()
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
            logger.warning(f"Failed to compute HMM features: {e}")
            if pair_key in self._hmm_cache:
                hmm_features = self._hmm_cache[pair_key]
                idx_len = len(dataframe)
                for col, values in hmm_features.items():
                    if len(values) != idx_len:
                        values = np.resize(values, idx_len)
                    dataframe[col] = values
            else:
                hmm_defaults = {
                    'hmm_regime': 0, 'hmm_p_bull': 0.7, 'hmm_p_range': 0.2,
                    'hmm_p_bear': 0.1, 'hmm_regime_stability': 0.5,
                    'hmm_transition_risk': 0.3, 'hmm_vol_regime': 1.0,
                    'hmm_trend_strength': 1.0,
                }
                for col, val in hmm_defaults.items():
                    if col not in dataframe.columns:
                        dataframe[col] = val

        self._hmm_update_count += 1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi = f"rsi_{self.rsi_period.value}"

        conditions_pullback = [
            dataframe["is_bull"] == 1,
            dataframe["pullback_to_ema"] == 1,
            dataframe[rsi] > self.rsi_pullback_low.value,
            dataframe[rsi] < self.rsi_pullback_high.value,
            dataframe["adx"] > self.adx_threshold.value,
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["plus_di"] > dataframe["minus_di"],
            dataframe["macdhist"] > dataframe["macdhist"].shift(1),
            dataframe["obv"] > dataframe["obv_ema"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_pullback),
            ["enter_long", "enter_tag"]
        ] = (1, "trend_pullback")

        conditions_rsi = [
            dataframe["close"] > dataframe["ema_200"],
            dataframe[rsi].shift(1) < self.rsi_bounce.value,
            dataframe[rsi] > self.rsi_bounce.value,
            dataframe["close"] > dataframe["bb_lower"],
            dataframe["close"] > dataframe["open"],
            dataframe["adx"] > self.adx_threshold.value,
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["obv"] > dataframe["obv_ema"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_rsi),
            ["enter_long", "enter_tag"]
        ] = (1, "rsi_bounce")

        conditions_macd = [
            (dataframe["macdhist"] > 0) &
            (dataframe["macdhist"].shift(1) <= 0),
            dataframe["close"] > dataframe["ema_50"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe[rsi] > 40,
            dataframe[rsi] < 60,
            dataframe["adx"] > self.adx_threshold.value,
            dataframe["volume_ratio"] > self.volume_factor.value,
            dataframe["plus_di"] > dataframe["minus_di"],
            dataframe["obv"] > dataframe["obv_ema"],
            dataframe["volume"] > 0,
            dataframe["is_bull"] == 1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_macd),
            ["enter_long", "enter_tag"]
        ] = (1, "macd_reversal")

        conditions_kalman_breakout = [
            dataframe["close"] > dataframe["kf_prediction"],
            dataframe["kf_trend"] > 0,
            dataframe["kf_trend_acceleration"] > 0,
            dataframe["close"] > dataframe["bb_upper"] * 0.995,
            dataframe["volume_ratio"] > self.volume_factor.value * 2.0,
            dataframe["adx"] > self.adx_threshold.value * 1.25,
            dataframe["kf_confidence"] > 0.5,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_kalman_breakout),
            ["enter_long", "enter_tag"]
        ] = (1, "kalman_breakout")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return 30.0

    def _compute_ml_features(self, pair: str, rate: float, last_row, current_time=None) -> list:
        if self._model_meta is None:
            return None

        open_val = last_row.get("open", rate)
        high_val = last_row.get("high", rate)
        low_val = last_row.get("low", rate)
        close_val = last_row.get("close", rate)

        candle_range = (high_val - low_val) / (close_val + 1e-10)
        body = abs(close_val - open_val) / (high_val - low_val + 1e-10)
        close_open_ratio = close_val / (open_val + 1e-10)
        bullish_candle = 1 if close_val > open_val else 0
        close_gt_open_pct = (close_val - open_val) / (open_val + 1e-10)

        now = current_time or datetime.now()
        vals = {
            "weekday": now.weekday(),
            "hour": now.hour,
            "trade_duration": 0,
            "leverage": 30.0,
            "open_rate": rate,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "rsi_14": last_row.get("rsi_14", 50),
            "adx": last_row.get("adx", 20),
            "plus_di": last_row.get("plus_di", 20),
            "minus_di": last_row.get("minus_di", 20),
            "macd": last_row.get("macd", 0),
            "macdsignal": last_row.get("macdsignal", 0),
            "macdhist": last_row.get("macdhist", 0),
            "bb_upper": last_row.get("bb_upper", 0),
            "bb_middle": last_row.get("bb_middle", 0),
            "bb_lower": last_row.get("bb_lower", 0),
            "bb_width": last_row.get("bb_width", 0),
            "bb_position": last_row.get("bb_position", 0.5),
            "volume_ema": last_row.get("volume_ema", 0),
            "volume_ratio": last_row.get("volume_ratio", 1.0),
            "obv": last_row.get("obv", 0),
            "obv_ema": last_row.get("obv_ema", 0),
            "atr": last_row.get("atr", 0),
            "atr_pct": last_row.get("atr", 0) / (close_val + 1e-10),
            "ema_50": last_row.get("ema_50", 0),
            "ema_200": last_row.get("ema_200", 0),
            "is_bull_5m": last_row.get("is_bull", 0),
            "candle_range": candle_range,
            "candle_body": body,
            "close_open_ratio": close_open_ratio,
            "is_bull_1h": last_row.get("is_bull_1h", 1),
            "rsi_14_1h": last_row.get("rsi_14_1h", 50),
            "adx_1h": last_row.get("adx_1h", 20),
            "is_bull_4h": last_row.get("is_bull_4h", 1),
            "di_spread": last_row.get("plus_di", 20) - last_row.get("minus_di", 20),
            "macd_hist_sign": 1 if last_row.get("macdhist", 0) > 0 else 0,
            "volume_spike": 1 if last_row.get("volume_ratio", 1.0) > 1.5 else 0,
            "ema_aligned": 1 if last_row.get("is_bull", 0) == 1 else 0,
            "bullish_candle": bullish_candle,
            "close_gt_open_pct": close_gt_open_pct,
            "pair_enc": 0,
            "tag_enc": 0,
            "kf_price": last_row.get("kf_price", close_val),
            "kf_trend": last_row.get("kf_trend", 0.0),
            "kf_prediction": last_row.get("kf_prediction", close_val),
            "kf_confidence": last_row.get("kf_confidence", 0.5),
            "kf_direction": last_row.get("kf_direction", 0),
            "kf_innovation": last_row.get("kf_innovation", 0.0),
            "kf_S": last_row.get("kf_S", 0.01),
            "kf_price_momentum": last_row.get("kf_price_momentum", 0.0),
            "kf_trend_acceleration": last_row.get("kf_trend_acceleration", 0.0),
            "kf_prediction_error": last_row.get("kf_prediction_error", 0.0),
            "kf_regime_score": last_row.get("kf_regime_score", 0.0),
            "kf_vol_of_trend": last_row.get("kf_vol_of_trend", 0.0),
            "kf_atr_ratio": last_row.get("kf_atr_ratio", 0.0),
            "hmm_regime": last_row.get("hmm_regime", 1),
            "hmm_p_bull": last_row.get("hmm_p_bull", 0.5),
            "hmm_p_range": last_row.get("hmm_p_range", 0.3),
            "hmm_p_bear": last_row.get("hmm_p_bear", 0.2),
            "hmm_regime_stability": last_row.get("hmm_regime_stability", 0.5),
            "hmm_transition_risk": last_row.get("hmm_transition_risk", 0.1),
            "hmm_vol_regime": last_row.get("hmm_vol_regime", 1.0),
            "hmm_trend_strength": last_row.get("hmm_trend_strength", 1.0),
        }

        features = []
        for col in self._model_meta["feature_cols"]:
            v = vals.get(col, 0)
            if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
                v = 0
            features.append(v)

        return features

    def _calc_confidence(self, last: dict) -> tuple:
        score = 0.0
        details = []
        rsi_key = f"rsi_{self.rsi_period.value}"
        rsi_val = last.get(rsi_key, 50)

        if 35 < rsi_val < 60:
            score += 1.5
            details.append("RSI healthy")

        adx_val = last.get('adx', 0)
        if adx_val > 30:
            score += 2.5
            details.append("Strong trend")
        elif adx_val > self.adx_threshold.value:
            score += 1.5
            details.append("Moderate trend")

        vol_ratio = last.get('volume_ratio', 0)
        if vol_ratio > 1.5:
            score += 2.5
            details.append("High volume")
        elif vol_ratio > 1.0:
            score += 1.5
            details.append("Normal volume")

        macd_hist = last.get('macdhist', 0)
        if macd_hist > 0:
            score += 1.5
            details.append("MACD positive")

        if last.get('obv', 0) > last.get('obv_ema', 0):
            score += 1.5
            details.append("OBV rising")

        if last.get('is_bull_1h', 0) == 1:
            score += 1.5
            details.append("1H trend aligned")

        if last.get('is_bull_4h', 0) == 1:
            score += 1.5
            details.append("4H trend aligned")

        close = last.get('close', 0)
        bb_lower = last.get('bb_lower', 0)
        bb_upper = last.get('bb_upper', 0)
        bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1
        if bb_lower > 0 and close > 0:
            bb_position = (close - bb_lower) / bb_range
            if bb_position < 0.35:
                score += 1.0
                details.append("Near BB lower")

        plus_di = last.get('plus_di', 0)
        minus_di = last.get('minus_di', 0)
        if plus_di - minus_di > 10:
            score += 1.0
            details.append("Strong DI spread")

        numeric = max(1, min(10, round(score * 10 / 17.5)))

        if numeric >= 8:
            level = "STRONG"
        elif numeric >= 6:
            level = "GOOD"
        elif numeric >= 4:
            level = "MEDIUM"
        else:
            level = "WEAK"

        bar = "|" * numeric + "-" * (10 - numeric) + f" {numeric}/10"

        return level, bar, details, numeric

    def _market_context(self, last: dict) -> str:
        bull_1h = last.get('is_bull_1h', 0)
        bull_4h = last.get('is_bull_4h', 0)
        tf_1h = "Uptrend" if bull_1h else "Downtrend"
        tf_4h = "Uptrend" if bull_4h else "Downtrend"
        return " | ".join([f"1H: {tf_1h}", f"4H: {tf_4h}"])

    def _get_market_regime(self, last: dict) -> str:
        adx_val = last.get('adx', 0)
        ema_200 = last.get('ema_200', 0)
        close = last.get('close', 0)
        is_bull = last.get('is_bull', 0)
        bb_width = last.get('bb_width', 0)
        bb_width_sma = last.get('bb_width_sma', 0)
        high_vol = bb_width > bb_width_sma * 1.5 if bb_width_sma > 0 else False
        if adx_val < 20:
            return "Ranging (High Vol)" if high_vol else "Ranging"
        elif is_bull and close > ema_200:
            return "Trending Bull"
        else:
            return "Trending Bear (High Vol)" if high_vol else "Trending Bear"

    def _check_market_regime(self) -> bool:
        if not self.dp:
            return True
        try:
            btc_daily = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1d")
            if btc_daily is not None and len(btc_daily) > 50:
                close = btc_daily['close'].values
                ema50 = ta.EMA(close, timeperiod=50)
                self._regime_bull = close[-1] > ema50[-1]
                self._last_regime_check = datetime.now()
        except Exception as e:
            logger.warning(f"Regime check failed: {e}")
        return self._regime_bull

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        if not self._check_market_regime():
            logger.info(f"REJECTED {pair}: bear market regime (BTC daily < EMA50)")
            return False
        return True

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        profit_pct = ((rate - trade.open_rate) / trade.open_rate) * 100 * trade.leverage
        duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        exit_reasons = {
            "roi": "ROI target reached",
            "stop_loss": "Stop Loss hit (ATR-based)",
            "trailing_stop_loss": "Trailing Stop",
            "exit_signal": "Exit signal",
            "rsi_overbought": "RSI overbought",
            "ema_bearish_cross": "EMA bearish crossover",
            "trend_broken": "Trend broken (below EMA200)",
            "trend_early_warning": "Trend early warning",
            "force_exit": "Force exit",
            "early_loss_cut_4h": "Early loss cut (4h, <-3%)",
            "early_loss_cut_8h": "Early loss cut (8h, <-2%)",
            "early_loss_cut_16h": "Early loss cut (16h, <0%)",
            "time_exit_24h": "Time exit (24h)",
        }
        reason_text = exit_reasons.get(exit_reason, exit_reason)

        result_line = f"+{profit_pct:.2f}%" if profit_pct > 0 else f"{profit_pct:.2f}%"

        if duration_hours < 1:
            dur_str = f"{int(duration_hours * 60)}m"
        elif duration_hours < 24:
            dur_str = f"{duration_hours:.1f}h"
        else:
            dur_str = f"{duration_hours/24:.1f}d"

        msg = (
            f"*TRADE CLOSED* {'WIN' if profit_pct > 0 else 'LOSS'}\n"
            f"{'='*25}\n"
            f"*{pair}* | LONG | {trade.leverage}x\n"
            f"{'='*25}\n\n"
            f"*Entry:* `{trade.open_rate:.2f}`\n"
            f"*Exit:* `{rate:.2f}`\n"
            f"*Result:* *{result_line}*\n"
            f"*Duration:* {dur_str}\n"
            f"*Reason:* {reason_text}\n"
            f"*Max price:* `{trade.max_rate:.2f}`\n"
            f"{'='*25}\n"
            f"_Phoenix AI_"
        )

        self.dp.send_msg(msg, always_send=True)
        return True
