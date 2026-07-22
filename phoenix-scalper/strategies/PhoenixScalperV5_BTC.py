import talib.abstract as ta
from datetime import datetime, timezone
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
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
from core.telegram_ev import register_ev_command

logger = logging.getLogger(__name__)


class PhoenixScalperV5_BTC(IStrategy):
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
    max_open_trades = 3

    MAX_CONSECUTIVE_LOSSES = 5

    buy_params = {
        "vol_spike_mult": 0.5,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "adx_min": 20,
        "obv_rsi_min": 50,
        "obv_proximity": 0.998,
    }

    sell_params = {
        "tp_target": 0.071,
        "bleed_loss": 0.033,
        "bleed_time": 289,
        "max_hold_min": 303,
        "trail_threshold": 0.078,
        "lock_ratio": 0.359,
    }

    vol_spike_mult = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    rsi_oversold = IntParameter(20, 35, default=30, space="buy")
    rsi_overbought = IntParameter(65, 80, default=70, space="buy")
    adx_min = IntParameter(15, 30, default=20, space="buy")
    obv_rsi_min = IntParameter(45, 60, default=50, space="buy")
    obv_proximity = DecimalParameter(0.990, 1.005, default=0.998, space="buy")

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
        self._drift_ref_set = False
        self._drift_ref_buffer = {}
        self._monitor = None
        self._drift_mode = "normal"
        self._daily_trade_counts = {}

    def _ensure_monitor(self):
        if self._monitor is None and hasattr(self, 'dp') and self.dp is not None:
            self._monitor = Monitor(
                dp=self.dp,
                bot_name="PhoenixScalperV4",
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
                regime="BTC_ONLY",
                risk_level=risk_state.level.value,
                exposure=risk_state.current_exposure,
                leverage=self.LEVERAGE,
                active_trades=open_trades,
                max_trades=self.max_open_trades,
                bot_name="PhoenixScalperV4",
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
        return [("BTC/USDT:USDT", "15m"), ("BTC/USDT:USDT", "1d")]

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
            self._data_validator.validate_candles(dataframe, pair_key)

        dataframe["rsi_5"] = ta.RSI(dataframe, timeperiod=5)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        for period in [7, 21, 50, 200]:
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=17, signalperiod=5)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

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
                    predict_horizon=1,
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
            for col, val in {
                'hmm_regime': 1, 'hmm_p_bull': 0.5, 'hmm_p_range': 0.3,
                'hmm_p_bear': 0.2, 'hmm_regime_stability': 0.5,
                'hmm_transition_risk': 0.3, 'hmm_vol_regime': 1.0,
                'hmm_trend_strength': 1.0,
            }.items():
                if col not in dataframe.columns:
                    dataframe[col] = val

        self._hmm_update_count += 1

        try:
            regime_result = self._regime_engine.analyze(dataframe)
            if metadata.get("pair", "").startswith("BTC/"):
                self._last_regime_str = regime_result.regime.value
        except Exception as e:
            logger.warning(f"RegimeEngine: {e}")

        if do_heavy:
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
                all_ready = all(len(buf) >= 50 for buf in self._drift_ref_buffer.values())
                if all_ready:
                    self._drift_ref_set = True
        except Exception as e:
            logger.warning(f"ConceptDrift feed: {e}")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_short"] = 0
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        vol_spike = dataframe["volume_ratio"] >= self.vol_spike_mult.value

        long_setup_1 = (
            vol_spike &
            (dataframe["rsi_5"] < self.rsi_oversold.value) &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[long_setup_1 & (dataframe["enter_long"] == 0), ["enter_long", "enter_tag"]] = (1, "vol_spike_oversold")

        obv_bear = dataframe["obv"] < dataframe["obv_ema"]
        long_setup_2 = (
            (dataframe["macdhist"].shift(1) < 0) &
            (dataframe["macdhist"] > 0) &
            (dataframe["volume_ratio"] > 0.4) &
            (dataframe["adx"] >= self.adx_min.value) &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[long_setup_2 & (dataframe["enter_long"] == 0), ["enter_long", "enter_tag"]] = (1, "macd_flip_long")

        short_setup_1 = (
            vol_spike &
            (dataframe["rsi_5"] > self.rsi_overbought.value) &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[short_setup_1 & (dataframe["enter_short"] == 0), ["enter_short", "enter_tag"]] = (1, "vol_spike_overbought")

        short_setup_2 = (
            obv_bear &
            (dataframe["close"] >= dataframe["ema_21"] * self.obv_proximity.value) &
            (dataframe["close"] < dataframe["ema_21"]) &
            (dataframe["rsi_5"] > self.obv_rsi_min.value) &
            (dataframe["volume_ratio"] > 0.4) &
            (dataframe["volume"] > 0)
        )
        dataframe.loc[short_setup_2 & (dataframe["enter_short"] == 0), ["enter_short", "enter_tag"]] = (1, "obv_resistance_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 60
        if current_profit >= self.sell_params["tp_target"]:
            return "v5btc_tp"
        if current_profit < -self.sell_params["bleed_loss"] and elapsed > self.sell_params["bleed_time"]:
            return "v5btc_bleed_exit"
        if elapsed > self.sell_params["max_hold_min"]:
            return "v5btc_max_hold"
        return None

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        if after_fill:
            return -0.99
        trail_threshold = self.sell_params["trail_threshold"]
        lock_ratio = self.sell_params["lock_ratio"]
        if current_profit > trail_threshold:
            lock_equity = max(current_profit * lock_ratio, trail_threshold * 0.5)
            return -(lock_equity / trade.leverage)
        return self.stoploss

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return float(self.LEVERAGE)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str | None, side: str, **kwargs) -> float:
        wallet = self.wallets.get_total_stake_amount()
        stake_pct = 0.08
        return max(min_stake, min(wallet * stake_pct, max_stake))

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        today = current_time.date()
        if self._current_trading_day != today:
            self._current_trading_day = today
            self._consecutive_losses = 0
            self._daily_trade_counts = {}

        if self._consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            logger.info(f"V5BTC loss breaker active ({self._consecutive_losses}), rejecting {pair}")
            return False

        daily_key = f"{today}_{pair}"
        pair_daily_count = self._daily_trade_counts.get(daily_key, 0)
        if pair_daily_count >= 10:
            logger.info(f"V5BTC daily limit reached for {pair} ({pair_daily_count}/10)")
            return False
        self._daily_trade_counts[daily_key] = pair_daily_count + 1

        pair_cooldown_key = f"cooldown_{pair}"
        last_loss_time = getattr(self, pair_cooldown_key, None)
        if last_loss_time is not None:
            elapsed = (current_time - last_loss_time).total_seconds() / 60
            if elapsed < 30:
                logger.info(f"V5BTC cooldown active for {pair} ({elapsed:.0f}m)")
                return False

        trade_id = self._trade_intel.start_trade(
            pair=pair, side=side, leverage=self.leverage(pair, current_time, rate, 0, 0, entry_tag, side),
            entry_price=rate, entry_tag=entry_tag or "",
            market_state={}, regime="BTC_ONLY",
            regime_confidence=0.5, risk_level="normal",
        )
        return True

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        profit_pct = trade.calc_profit_ratio(rate=rate) * 100
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

        if profit_pct < 0:
            pair_cooldown_key = f"cooldown_{trade.pair}"
            setattr(self, pair_cooldown_key, current_time)

        self._trade_intel.close_trade(
            trade_id=str(trade.id),
            exit_price=rate,
            profit_pct=profit_pct,
            exit_reason=exit_reason,
            failure_factors=failure_factors,
            success_factors=success_factors,
        )

        return True
