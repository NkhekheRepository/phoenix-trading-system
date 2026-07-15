import talib.abstract as ta
from datetime import datetime, timezone, date
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, merge_informative_pair
from freqtrade.persistence import Trade
from pandas import DataFrame
from functools import reduce
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.monitoring import Monitor
from core.trade_intel import TradeIntelligence
from core.regime_engine import RegimeEngine
from core.risk_governor import RiskGovernor
from core.data_quality import DataValidator
from core.concept_drift import ConceptDriftDetector
from core.ml_engine import MLEngine

logger = logging.getLogger(__name__)


class PhoenixScalperV2(IStrategy):
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

    LEVERAGE = 30

    max_open_trades = 5

    MAX_CONSECUTIVE_LOSSES = 999

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
    score_threshold = IntParameter(35, 80, default=45, space="buy")
    score_high_threshold = IntParameter(45, 75, default=55, space="buy")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._consecutive_losses = 0
        self._current_trading_day = None
        self._hmm_cache = {}
        self._kalman_cache = {}
        self._hmm_update_count = 0
        self._cache_ttl = 3600

        self._start_time = datetime.now(timezone.utc)
        self._trade_intel = TradeIntelligence(on_notify=self._notify_handler)
        self._regime_engine = RegimeEngine(on_notify=self._notify_handler)
        self._risk_governor = RiskGovernor(
            max_consecutive_losses=self.MAX_CONSECUTIVE_LOSSES,
            on_notify=self._notify_handler,
        )
        self._data_validator = DataValidator(max_candle_age_minutes=10, on_notify=self._notify_handler)
        self._concept_drift = ConceptDriftDetector(
            psi_threshold=0.2, kl_threshold=0.1, wasserstein_threshold=0.5, window_size=500,
            on_notify=self._notify_handler,
        )
        self._ml_engine = MLEngine(
            trade_intel=self._trade_intel,
            concept_drift=self._concept_drift,
            on_notify=self._notify_handler,
        )
        self._ml_baseline_set = False
        self._ml_check_counter = 0
        self._drift_ref_set = False
        self._drift_ref_buffer = {}
        self._monitor = None

    def _ensure_monitor(self):
        if self._monitor is None and hasattr(self, 'dp') and self.dp is not None:
            self._monitor = Monitor(
                dp=self.dp,
                bot_name="PhoenixScalperV2",
                chat_id=self.config.get("telegram", {}).get("chat_id"),
                token=self.config.get("telegram", {}).get("token"),
            )

    def _notify_handler(self, method_name: str, **kwargs):
        self._ensure_monitor()
        if self._monitor is None:
            return
        handler = getattr(self._monitor, f"notify_{method_name}", None)
        if handler:
            try:
                handler(**kwargs)
            except Exception as e:
                logger.warning(f"Monitor notify_{method_name} failed: {e}")

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        try:
            self._ensure_monitor()
            if self._monitor is None:
                return
            open_trades = []
            try:
                all_trades = Trade.get_trades_proxy()
                for trade in all_trades:
                    if trade.is_open:
                        profit = trade.calc_profit_ratio(rate=trade.open_rate)
                        open_trades.append({
                            "pair": trade.pair,
                            "rate": trade.open_rate,
                            "profit": f"{profit * 100:.2f}%" if profit else "0.00%",
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch open trades: {e}")

            try:
                balance = self.wallets.get_total_stake_amount() if hasattr(self, 'wallets') else 0
                risk_state = self._risk_governor.update(balance, open_trades)
            except Exception:
                risk_state = self._risk_governor.get_state()

            total_trades_db = 0
            total_profit = 0.0
            win_count = 0
            loss_count = 0
            try:
                all_closed = Trade.get_trades_proxy(is_open=False)
                total_trades_db = len(all_closed)
                for t in all_closed:
                    pnl = t.close_profit_abs or 0
                    total_profit += pnl
                    if t.close_profit and t.close_profit > 0:
                        win_count += 1
                    elif t.close_profit and t.close_profit < 0:
                        loss_count += 1
            except Exception as e:
                logger.warning(f"Failed to query trades for summary: {e}")

            memory_mb = 0
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            memory_mb = int(line.split()[1]) // 1024
                            break
            except Exception:
                pass

            exchange_ok = True
            try:
                if hasattr(self, 'exchange') and hasattr(self.exchange, 'get_status'):
                    exchange_ok = self.exchange.get_status().get('status', 'connected') == 'connected'
            except Exception:
                exchange_ok = False

            date_str = current_time.strftime("%Y-%m-%d")
            self._monitor.send_daily_summary(
                date_str=date_str,
                regime=self._last_regime_str if hasattr(self, "_last_regime_str") else "N/A",
                risk_level=risk_state.level.value,
                exposure=risk_state.current_exposure,
                leverage=self.LEVERAGE,
                active_trades=open_trades,
                max_trades=self.max_open_trades,
                bot_name="PhoenixScalperV2",
                total_trades_db=total_trades_db,
                total_profit=total_profit,
                win_count=win_count,
                loss_count=loss_count,
            )

            self._monitor.send_hourly_health({
                "uptime": str(datetime.now(timezone.utc) - self._start_time).split(".")[0],
                "total_trades": total_trades_db,
                "active_trades": len(open_trades),
                "memory_mb": memory_mb,
                "exchange_ok": exchange_ok,
                "total_profit": total_profit,
                "win_count": win_count,
                "loss_count": loss_count,
                "score_threshold": self.score_threshold.value,
            })

            self._monitor.flush()
        except Exception as e:
            logger.warning(f"bot_loop_start error: {e}")

    protections = []

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        result = [(p, '15m') for p in pairs]
        result.append(("BTC/USDT:USDT", "1d"))
        return result

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
        do_heavy = (self._hmm_update_count % 60 == 0)

        if do_heavy:
            validation = self._data_validator.validate_candles(dataframe, pair_key)

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
                aligned = self._align_array(values, idx_len)
                dataframe[col] = aligned
        except Exception as e:
            logger.warning(f"Kalman: {e}")
            if pair_key in self._kalman_cache:
                kf_features = self._kalman_cache[pair_key]
                idx_len = len(dataframe)
                for col, values in kf_features.items():
                    aligned = self._align_array(values, idx_len)
                    dataframe[col] = aligned
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
                hmm_features = compute_hmm_features(
                    returns.fillna(0).values,
                    vol.fillna(0).values,
                    volume_change.fillna(0).values,
                )
                self._hmm_cache[pair_key] = hmm_features
            else:
                hmm_features = self._hmm_cache[pair_key]
            idx_len = len(dataframe)
            for col, values in hmm_features.items():
                aligned = self._align_array(values, idx_len)
                dataframe[col] = aligned
        except Exception as e:
            logger.warning(f"HMM: {e}")
            if pair_key in self._hmm_cache:
                hmm_features = self._hmm_cache[pair_key]
                idx_len = len(dataframe)
                for col, values in hmm_features.items():
                    aligned = self._align_array(values, idx_len)
                    dataframe[col] = aligned
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

        try:
            regime_result = self._regime_engine.analyze(dataframe)
            if metadata.get("pair", "").startswith("BTC/"):
                self._last_regime_str = regime_result.regime.value
        except Exception as e:
            logger.warning(f"RegimeEngine: {e}")

        if do_heavy:
            self._feed_concept_drift(dataframe)

        dataframe = self._calculate_entry_score(dataframe)

        return dataframe

    def _feed_concept_drift(self, dataframe: DataFrame):
        try:
            drift_features = ['hmm_p_bull', 'hmm_p_bear', 'hmm_regime_stability']
            for col in drift_features:
                vals = dataframe[col].dropna().values
                if len(vals) == 0:
                    continue
                if col not in self._drift_ref_buffer:
                    self._drift_ref_buffer[col] = []
                self._drift_ref_buffer[col].append(vals[-1])
                if not self._drift_ref_set and len(self._drift_ref_buffer[col]) >= 50:
                    self._concept_drift.set_reference(col, np.array(self._drift_ref_buffer[col]))
                elif self._drift_ref_set:
                    self._concept_drift.update(col, float(vals[-1]))
            if not self._drift_ref_set:
                all_ready = all(
                    len(buf) >= 50 for buf in self._drift_ref_buffer.values()
                )
                if all_ready:
                    self._drift_ref_set = True
        except Exception as e:
            logger.warning(f"ConceptDrift feed: {e}")

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
            dataframe["hmm_p_bull"] > dataframe["hmm_p_bear"],
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
            dataframe["hmm_p_bull"] > dataframe["hmm_p_bear"],
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
            dataframe["hmm_p_bull"] > dataframe["hmm_p_bear"],
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
            dataframe["hmm_p_bull"] > dataframe["hmm_p_bear"],
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
            dataframe["volume_ratio"] > self.volume_factor.value * 1.3,
            dataframe["adx"] > self.adx_threshold.value * 1.1,
            dataframe["plus_di"] < dataframe["minus_di"],
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_short_bear),
            ["enter_short", "enter_tag"]
        ] = (1, "short_bear_momentum")

        threshold = self.score_threshold.value
        dataframe.loc[
            (dataframe["enter_long"] == 1) & (dataframe["signal_score"] < threshold),
            ["enter_long", "enter_tag"]
        ] = (0, None)
        dataframe.loc[
            (dataframe["enter_short"] == 1) & (dataframe["short_score"] < threshold),
            ["enter_short", "enter_tag"]
        ] = (0, None)

        high_th = self.score_high_threshold.value
        no_entry = (dataframe["enter_long"].fillna(0) == 0) & (dataframe["enter_short"].fillna(0) == 0)
        high_long = no_entry & (dataframe["signal_score"] >= high_th) & (dataframe["signal_score"] > dataframe["short_score"])
        high_short = no_entry & (dataframe["short_score"] >= high_th) & (dataframe["short_score"] > dataframe["signal_score"])
        dataframe.loc[high_long, ["enter_long", "enter_tag"]] = (1, "score_override_long")
        dataframe.loc[high_short, ["enter_short", "enter_tag"]] = (1, "score_override_short")

        long_pass = (dataframe["enter_long"] == 1)
        if long_pass.any():
            dataframe.loc[long_pass, "enter_tag"] = (
                dataframe.loc[long_pass, "enter_tag"].astype(str) +
                " [" + dataframe.loc[long_pass, "signal_score"].astype(int).astype(str) + "]"
            )

        short_pass = (dataframe["enter_short"] == 1)
        if short_pass.any():
            dataframe.loc[short_pass, "enter_tag"] = (
                dataframe.loc[short_pass, "enter_tag"].astype(str) +
                " [" + dataframe.loc[short_pass, "short_score"].astype(int).astype(str) + "]"
            )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if current_profit >= 0.10:
            return "mc1_tp_10pct"
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 3600
        if elapsed > 3:
            return "max_hold_3h"
        return None

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        if after_fill:
            return -0.99
        lev = trade.leverage
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 60
        if elapsed < 5:
            max_eq_loss = 0.75
        elif elapsed < 30:
            max_eq_loss = 0.50
        else:
            max_eq_loss = 0.30
        time_stop = -(max_eq_loss)
        if current_profit > 0.02:
            trail_offset = 0.03
            if current_profit > 0.05:
                trail_offset = 0.02
            if current_profit > 0.10:
                trail_offset = 0.01
            trail_stop = -(current_profit - trail_offset)
            trail_stop = max(trail_stop, -(0.02))
            return max(trail_stop, time_stop)
        return time_stop

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(self.LEVERAGE)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str | None, side: str, **kwargs) -> float:
        wallet = self.wallets.get_total_stake_amount()
        kelly_75_pct = 0.10
        return max(min_stake, min(wallet * kelly_75_pct, max_stake))

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        today = current_time.date()
        if self._current_trading_day != today:
            self._current_trading_day = today
            self._consecutive_losses = 0
        if self._consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            logger.info(f"Loss breaker active ({self._consecutive_losses} consecutive), rejecting {pair} {side}")
            return False

        regime_str = self._last_regime_str if hasattr(self, "_last_regime_str") else "unknown"
        if regime_str in ("strong_bear", "low_volatility", "weak_bull"):
            logger.info(f"Regime {regime_str} blocked, rejecting {pair} {side}")
            return False

        trade_id = self._trade_intel.start_trade(
            pair=pair, side=side, leverage=self.leverage(pair, current_time, rate, 0, 0, entry_tag, side),
            entry_price=rate, entry_tag=entry_tag or "",
            market_state={}, regime=regime_str,
            regime_confidence=0.5, risk_level="normal",
        )
        return True

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        profit_pct = trade.calc_profit_ratio(rate=rate) * 100

        today = current_time.date()
        if self._current_trading_day != today:
            self._current_trading_day = today
            self._consecutive_losses = 0

        if profit_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        try:
            balance = self.wallets.get_total_stake_amount() if hasattr(self, 'wallets') else 0
            self._risk_governor.record_trade_result(profit_pct, balance)
        except Exception:
            pass

        failure_factors = []
        success_factors = []
        if profit_pct < 0:
            failure_factors.append(f"exit_reason:{exit_reason}")
            failure_factors.append("trade_lost")
        else:
            success_factors.append("trade_won")
            success_factors.append(f"exit_reason:{exit_reason}")

        entry_time = trade.open_date_utc if hasattr(trade, 'open_date_utc') else trade.open_date
        self._trade_intel.close_trade(
            trade_id=str(trade.id),
            exit_price=rate,
            profit_pct=profit_pct,
            exit_reason=exit_reason,
            failure_factors=failure_factors,
            success_factors=success_factors,
            pair=trade.pair,
            side="short" if trade.is_short else "long",
            entry_tag=trade.enter_tag or "",
            leverage=trade.leverage,
            entry_price=trade.open_rate,
            regime=self._last_regime_str if hasattr(self, "_last_regime_str") else "unknown",
            entry_time=entry_time,
        )

        if not self._ml_baseline_set:
            total = self._trade_intel.get_trade_count()
            if total >= 20:
                try:
                    winners = self._trade_intel.analyze_winning_patterns(n_trades=100)
                    losers = self._trade_intel.analyze_losing_patterns(n_trades=100)
                    n_w = winners.get("total_winners", 0)
                    n_l = losers.get("total_losers", 0)
                    if n_w + n_l >= 20:
                        self._ml_engine.set_baseline({
                            "win_rate": n_w / (n_w + n_l) if (n_w + n_l) > 0 else 0.5,
                            "avg_loss_pct": abs(losers.get("avg_loss_pct", 0)),
                        })
                        self._ml_baseline_set = True
                except Exception as e:
                    logger.warning(f"ML baseline: {e}")

        if self._ml_baseline_set:
            self._ml_check_counter += 1
            if self._ml_check_counter % 10 == 0:
                try:
                    triggers = self._ml_engine.check_retrain_triggers()
                except Exception as e:
                    logger.warning(f"ML check: {e}")

        return True
