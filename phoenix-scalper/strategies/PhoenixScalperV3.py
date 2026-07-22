import talib.abstract as ta
from datetime import datetime, timezone
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
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
from core.telegram_ev import register_ev_command

logger = logging.getLogger(__name__)


class PhoenixScalperV3(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = False
    can_short = True
    position_adjustment_enable = False

    minimal_roi = {}

    stoploss = -0.12
    use_custom_stoploss = True

    trailing_stop = False

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
    score_threshold = IntParameter(35, 80, default=55, space="buy")
    score_high_threshold = IntParameter(45, 75, default=60, space="buy")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._consecutive_losses = 0
        self._current_trading_day = None
        self._hmm_last_train = {}
        self._kalman_last_train = {}
        self._hmm_cache = {}
        self._kalman_cache = {}

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
        self._drift_mode = "normal"

    def _ensure_monitor(self):
        if self._monitor is None and hasattr(self, 'dp') and self.dp is not None:
            self._monitor = Monitor(
                dp=self.dp,
                bot_name="PhoenixScalperV3",
                chat_id=self.config.get("telegram", {}).get("chat_id"),
                token=self.config.get("telegram", {}).get("token"),
            )
            self._ensure_ev_command()

    def _ensure_ev_command(self):
        if not hasattr(self, '_ev_registered') or not self._ev_registered:
            try:
                rpc = self.dp._DataProvider__rpc
                for mod in rpc.registered_modules:
                    if hasattr(mod, '_app'):
                        register_ev_command(mod)
                        self._ev_registered = True
                        break
            except Exception as e:
                logger.warning(f"telegram_ev: could not register /ev: {e}")

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
        self._ensure_ev_command()
        try:
            regime_str = self._last_regime_str if hasattr(self, "_last_regime_str") else "unknown"
            regime_max = {"strong_bear": 10, "weak_bear": 10, "low_volatility": 7, "weak_bull": 3, "strong_bull": 0}
            new_max = regime_max.get(regime_str, 5)

            drift_summary = {}
            try:
                drift_summary = self._concept_drift.get_drift_summary()
            except Exception:
                pass
            max_psi = max((s.get("current_psi", 0) for s in drift_summary.values()), default=0)
            if max_psi > 2.0:
                self._drift_mode = "critical"
                drift_factor = {"strong_bear": 3, "weak_bear": 3, "low_volatility": 3, "weak_bull": 3, "strong_bull": 0}
                new_max = drift_factor.get(regime_str, 3)
                self.score_threshold.value = 60
                self.score_high_threshold.value = 65
            elif max_psi > 0.5:
                self._drift_mode = "warning"
                drift_factor = {"strong_bear": 7, "weak_bear": 7, "low_volatility": 5, "weak_bull": 3, "strong_bull": 0}
                new_max = drift_factor.get(regime_str, 5)
                self.score_threshold.value = 58
                self.score_high_threshold.value = 62
            else:
                self._drift_mode = "normal"
                self.score_threshold.value = 55
                self.score_high_threshold.value = 60

            if new_max != self.max_open_trades:
                self.max_open_trades = new_max
                logger.info(f"Regime {regime_str} drift={self._drift_mode}: max_open_trades -> {new_max}")

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
                leverage=10,
                active_trades=open_trades,
                max_trades=self.max_open_trades,
                bot_name="PhoenixScalperV3",
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
                "drift_mode": self._drift_mode,
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

        if pair_key not in self._hmm_cache:
            self._data_validator.validate_candles(dataframe, pair_key)

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

        current_time = datetime.now()

        try:
            last_kalman = self._kalman_last_train.get(pair_key, datetime.min)
            kalman_stale = (current_time - last_kalman).total_seconds() > 86400

            if kalman_stale or pair_key not in self._kalman_cache:
                from ml.kalman_filter import compute_kalman_features
                kf_features = compute_kalman_features(
                    dataframe['close'].values,
                    dataframe['atr_pct'].values,
                    dataframe['volume_ratio'].values
                )
                self._kalman_cache[pair_key] = kf_features
                self._kalman_last_train[pair_key] = current_time
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
            last_hmm = self._hmm_last_train.get(pair_key, datetime.min)
            hmm_stale = (current_time - last_hmm).total_seconds() > 86400

            if hmm_stale or pair_key not in self._hmm_cache:
                from ml.hmm_regime_v3 import compute_hmm_features
                returns = np.log(dataframe['close'] / dataframe['close'].shift(1))
                vol = returns.rolling(10).std()
                volume_change = dataframe['volume_ratio'].pct_change()
                hmm_features = compute_hmm_features(returns.fillna(0), vol.fillna(0), volume_change.fillna(0))
                self._hmm_cache[pair_key] = hmm_features
                self._hmm_last_train[pair_key] = current_time
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

        try:
            regime_result = self._regime_engine.analyze(dataframe)
            self._last_regime_str = regime_result.regime.value
        except Exception as e:
            logger.warning(f"RegimeEngine: {e}")

        self._feed_concept_drift(dataframe)

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
            dataframe["hmm_p_bear"] > dataframe["hmm_p_bull"],
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
            dataframe["hmm_p_bear"] > dataframe["hmm_p_bull"],
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
            dataframe["hmm_p_bear"] > dataframe["hmm_p_bull"],
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
            return "max_hold_3h"
        return None

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        if after_fill:
            return -0.99
        regime_str = self._last_regime_str if hasattr(self, "_last_regime_str") else "unknown"
        if trade.get_custom_data('regime_at_entry') is None:
            trade.set_custom_data('regime_at_entry', regime_str)
        lev = trade.leverage
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 60
        if elapsed < 5:
            max_eq_loss = 0.50
        elif elapsed < 30:
            max_eq_loss = 0.30
        else:
            max_eq_loss = 0.20
        time_stop = -(max_eq_loss / lev)
        if current_profit > 0.02:
            trail_offset = 0.03 / lev
            if current_profit > 0.05:
                trail_offset = 0.02 / lev
            if current_profit > 0.10:
                trail_offset = 0.01 / lev
            trail_stop = -(current_profit - trail_offset)
            trail_stop = max(trail_stop, -(0.02 / lev))
            return max(trail_stop, time_stop)
        return time_stop

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return 10.0

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

        import re as _re
        score_m = _re.search(r'\[(\d+)\]', entry_tag or "")
        if score_m:
            sc = int(score_m.group(1))
            if sc > 59:
                logger.info(f"Score ceiling: {sc} > 59, rejecting {pair} {side}")
                return False

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        regime_str = "unknown"
        if dataframe is not None and len(dataframe) > 0:
            last = dataframe.iloc[-1]
            hmm_p_bull = last.get('hmm_p_bull', 0.5)
            hmm_p_bear = last.get('hmm_p_bear', 0.2)
            if side == 'long' and hmm_p_bear > hmm_p_bull:
                logger.info(f"Regime filter rejected long {pair}: bear={hmm_p_bear:.2f} > bull={hmm_p_bull:.2f}")
                return False
            if side == 'short' and hmm_p_bull > hmm_p_bear:
                logger.info(f"Regime filter rejected short {pair}: bull={hmm_p_bull:.2f} > bear={hmm_p_bear:.2f}")
                return False
            regime_str = self._last_regime_str if hasattr(self, "_last_regime_str") else "unknown"
            if regime_str not in ("low_volatility", "weak_bull"):
                logger.info(f"Regime {regime_str} not allowed (v3 allowlist), rejecting {pair} {side}")
                return False

        trade_id = self._trade_intel.start_trade(
            pair=pair, side=side, leverage=10,
            entry_price=rate, entry_tag=entry_tag or "",
            market_state={}, regime=regime_str,
            regime_confidence=0.5, risk_level="normal",
        )
        return True

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        profit_pct = trade.calc_profit_ratio(rate=rate) * 100

        regime_str = self._last_regime_str if hasattr(self, "_last_regime_str") else "unknown"
        trade.set_custom_data('regime_at_exit', regime_str)
        trade.set_custom_data('exit_reason', exit_reason)

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
