import pytest
from core.monitoring import Monitor, MessageFormatter, Event, RateLimiter


class TestMessageFormatter:
    def test_regime_alert(self):
        msg = MessageFormatter.regime_alert("BULL", "BEAR", 0.85, "conservative", {"trend": 0.5, "momentum": 0.3, "hmm": 0.2})
        assert "REGIME SHIFT" in msg
        assert "BULL" in msg
        assert "BEAR" in msg
        assert "85%" in msg

    def test_risk_alert(self):
        msg = MessageFormatter.risk_alert("NORMAL", "PAUSED", ["5 consecutive losses"], {})
        assert "RISK" in msg.upper()
        assert "PAUSED" in msg

    def test_risk_update(self):
        msg = MessageFormatter.risk_update("NORMAL", [])
        assert "RISK STATUS" in msg.upper()
        assert "NORMAL" in msg

    def test_trade_attribution_success(self):
        msg = MessageFormatter.trade_attribution(
            "BTC/USDT", "long", 2.5, "take_profit", "entry_tag", "BULL",
            [], ["good_entry", "trend_following"], "2h"
        )
        assert "✅" in msg
        assert "BTC/USDT" in msg
        assert "2.50%" in msg

    def test_trade_attribution_loss(self):
        msg = MessageFormatter.trade_attribution(
            "ETH/USDT", "short", -1.2, "stop_loss", "entry", "BEAR",
            ["bad_timing"], [], "1h"
        )
        assert "❌" in msg
        assert "-1.20%" in msg

    def test_drift_alert(self):
        msg = MessageFormatter.drift_alert("rsi", 0.8, 0.5, 2.0, "warning", "monitor closely")
        assert "CONCEPT DRIFT" in msg
        assert "rsi" in msg
        assert "PSI" in msg

    def test_experiment_result(self):
        msg = MessageFormatter.experiment_result("exp_001", "test hypothesis", {"sharpe": 2.1}, "approved", "good results")
        assert "EXPERIMENT" in msg.upper()
        assert "exp_001" in msg
        assert "APPROVED" in msg.upper()

    def test_allocation_change(self):
        msg = MessageFormatter.allocation_change(
            [{"strategy": "trend", "weight": 0.6}], 0.2, "BULL", "NORMAL"
        )
        assert "ALLOCATION" in msg.upper()

    def test_memory_match(self):
        msg = MessageFormatter.memory_match("bear_regime", "loses 60%", 0.85)
        assert "MARKET MEMORY" in msg.upper()
        assert "bear_regime" in msg

    def test_deployment_event(self):
        msg = MessageFormatter.deployment_event("promote_shadow", "v1.2", "success", "new features")
        assert "DEPLOYMENT" in msg.upper()
        assert "v1.2" in msg

    def test_research_hypothesis(self):
        msg = MessageFormatter.research_hypothesis({
            "type": "parameter_tuning", "priority": "high",
            "observation": "Loss rate 60%", "hypothesis": "Tighten thresholds"
        })
        assert "HYPOTHESIS" in msg.upper()
        assert "Loss rate" in msg

    def test_daily_summary(self):
        msg = MessageFormatter.daily_summary({
            "date": "2026-07-08", "bot_name": "TestBot",
            "trades": 10, "wins": 6, "losses": 4, "win_rate": 0.6,
            "pnl": "$50.00", "regime": "BULL", "risk_level": "NORMAL",
            "exposure": 0.5, "leverage": 10.0,
            "best_trade": "BTC +2.5%", "worst_trade": "ETH -1.2%",
            "active_trades": [{"pair": "BTC/USDT"}], "max_trades": 5,
        })
        assert "DAILY" in msg.upper()
        assert "TestBot" in msg

    def test_bot_health(self):
        msg = MessageFormatter.bot_health({
            "bot_name": "TestBot", "uptime": "2h", "total_trades": 50,
            "active_trades": 2, "memory_mb": 256, "exchange": "binance", "exchange_ok": True,
        })
        assert "BOT HEALTH" in msg.upper()

    def test_retrain_alert(self):
        msg = MessageFormatter.retrain_alert([{"source": "degradation", "metric": "sharpe", "value": 1.2}])
        assert "RETRAIN" in msg.upper()
        assert "sharpe" in msg

    def test_retrain_scheduled(self):
        msg = MessageFormatter.retrain_scheduled("exp_001", "degradation")
        assert "RETRAIN SCHEDULED" in msg.upper()
        assert "exp_001" in msg

    def test_validation_gate_report_pass(self):
        msg = MessageFormatter.validation_gate_report("v1.5", True, "6/6")
        assert "VALIDATION" in msg.upper()
        assert "PASSED" in msg

    def test_validation_gate_report_fail(self):
        msg = MessageFormatter.validation_gate_report("v1.5", False, "3/6")
        assert "VALIDATION" in msg.upper()
        assert "FAILED" in msg

    def test_promotion_decision_promote(self):
        msg = MessageFormatter.promotion_decision("v1", "v2", True, ["Better Sharpe"])
        assert "PROMOTE" in msg

    def test_promotion_decision_reject(self):
        msg = MessageFormatter.promotion_decision("v1", "v2", False, ["Worse Sharpe"])
        assert "REJECT" in msg

    def test_rollback_alert(self):
        msg = MessageFormatter.rollback_alert("v1", "Live Sharpe 0.8 < threshold 1.0")
        assert "ROLLBACK" in msg.upper()

    def test_market_event(self):
        msg = MessageFormatter.market_event("volatility_spike", "3-sigma ATR spike", "high")
        assert "MARKET EVENT" in msg.upper()

    def test_knowledge_added(self):
        msg = MessageFormatter.knowledge_added("bear_regime", "Loses 60% in STRONG_BEAR", "quant_agent")
        assert "KNOWLEDGE" in msg.upper()

    def test_data_quality_alert(self):
        msg = MessageFormatter.data_quality_alert("BTC/USDT", "stale_data", "Last candle 18 min old")
        assert "DATA QUALITY" in msg.upper()

    def test_strategy_note(self):
        msg = MessageFormatter.strategy_note("scalping", "Increased ATR multiplier")
        assert "STRATEGY NOTE" in msg.upper()

    def test_separator(self):
        assert "━━━━━━━━━━" in MessageFormatter.SEP


class TestRateLimiter:
    def test_first_call_allowed(self):
        rl = RateLimiter()
        assert rl.can_send("test_key", 300) is True

    def test_rapid_call_blocked(self):
        rl = RateLimiter()
        rl.can_send("test_key", 300)
        assert rl.can_send("test_key", 300) is False

    def test_different_keys_independent(self):
        rl = RateLimiter()
        assert rl.can_send("key_a", 300) is True
        assert rl.can_send("key_b", 300) is True

    def test_zero_interval_always_allowed(self):
        rl = RateLimiter()
        assert rl.can_send("test_key", 0) is True
        assert rl.can_send("test_key", 0) is True


class TestEvent:
    def test_event_creation(self):
        ev = Event("test", "alert", "info", "Title", "Message body")
        assert ev.source == "test"
        assert ev.event_type == "alert"
        assert ev.severity == "info"
        assert ev.title == "Title"
        assert ev.message == "Message body"
        assert ev.data == {}
        assert ev.timestamp is not None

    def test_event_with_data(self):
        ev = Event("test", "alert", "warning", "Title", "Body", data={"key": "val"})
        assert ev.data["key"] == "val"


class TestMonitorNotifyMethods:
    def setup_method(self):
        self.monitor = Monitor()

    def _count_queue(self):
        return len(self.monitor._queue)

    def test_notify_regime_change_normal(self):
        self.monitor.notify_regime_change("BULL", "BEAR", 0.85, "conservative", {"trend": 0.5, "momentum": 0.3, "hmm": 0.2})
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "regime_engine"
        assert ev.event_type == "regime_change"
        assert "BEAR" in ev.message

    def test_notify_regime_change_rate_limited(self):
        self.monitor.notify_regime_change("BULL", "BEAR", 0.85, "conservative", {"trend": 0.5, "momentum": 0.3, "hmm": 0.2})
        self.monitor.notify_regime_change("BEAR", "BEAR", 0.90, "aggressive", {"trend": -0.2, "momentum": 0.1, "hmm": 0.8})
        assert self._count_queue() == 1
        assert self._count_queue() == 1

    def test_notify_risk_change_normal(self):
        self.monitor.notify_risk_change("NORMAL", [], {})
        assert self._count_queue() >= 1

    def test_notify_risk_change_emergency(self):
        self.monitor.notify_risk_change("EMERGENCY", ["critical"], {"drawdown": 0.15})
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.event_type in ("risk_alert", "risk_update", "risk_change")

    def test_notify_trade_attribution(self):
        self.monitor.notify_trade_attribution(
            "BTC/USDT", "long", 2.5, "take_profit", "entry_tag", "BULL",
            [], ["good_entry"], "2h"
        )
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.event_type == "trade_attribution"

    def test_notify_drift(self):
        self.monitor.notify_drift("rsi", 0.8, 0.5, 2.0, "warning", "monitor")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "concept_drift"

    def test_notify_experiment(self):
        self.monitor.notify_experiment("exp_001", "test", {"sharpe": 2.0}, "approved", "good")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "experiment_db"

    def test_notify_allocation(self):
        self.monitor.notify_allocation([], 0.2, "BULL", "NORMAL")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "strategy_allocator"

    def test_notify_memory_match(self):
        self.monitor.notify_memory_match("topic", "insight", 0.85)
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "market_memory"

    def test_notify_deployment(self):
        self.monitor.notify_deployment("promote_shadow", "v1.2", "success", "changes")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "deployment"

    def test_notify_research(self):
        self.monitor.notify_research({"type": "test", "hypothesis": "test"})
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "research"

    def test_notify_retrain_triggers(self):
        self.monitor.notify_retrain_triggers([{"source": "degradation", "metric": "sharpe", "value": 1.2}])
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "ml_engine"
        assert ev.event_type == "retrain_triggers"

    def test_notify_retrain_scheduled(self):
        self.monitor.notify_retrain_scheduled("exp_001", "degradation")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "ml_engine"
        assert ev.event_type == "retrain_scheduled"

    def test_notify_validation_complete_pass(self):
        self.monitor.notify_validation_complete("v1.5", True, "6/6")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "validation_pipeline"
        assert ev.severity == "info"

    def test_notify_validation_complete_fail(self):
        self.monitor.notify_validation_complete("v1.5", False, "3/6")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.severity == "warning"

    def test_notify_champion_challenger_eval_promote(self):
        self.monitor.notify_champion_challenger_eval("v1", "v2", True, [])
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "champion_challenger"

    def test_notify_champion_promoted(self):
        self.monitor.notify_champion_promoted("v2", "v1", ["Better Sharpe"])
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.event_type == "champion_promoted"

    def test_notify_auto_rollback(self):
        self.monitor.notify_auto_rollback("v1", 0.8, 1.0)
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.event_type == "auto_rollback"
        assert ev.severity == "warning"

    def test_notify_market_event(self):
        self.monitor.notify_market_event("volatility_spike", "3-sigma", "high")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "market_memory"
        assert ev.event_type == "market_event"

    def test_notify_knowledge_added(self):
        self.monitor.notify_knowledge_added("topic", "insight", "source")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.event_type == "knowledge_added"

    def test_notify_data_quality(self):
        self.monitor.notify_data_quality("BTC/USDT", "stale_data", "18 min old")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "data_quality"

    def test_notify_strategy_note(self):
        self.monitor.notify_strategy_note("scalping", "Increased ATR")
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.event_type == "strategy_note"

    def test_send_daily_summary(self):
        self.monitor.send_daily_summary(
            "2026-07-08", "BULL", "NORMAL", 0.5, 10.0,
            [{"pair": "BTC/USDT"}], 5, "TestBot"
        )
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "monitor"
        assert ev.event_type == "daily_summary"

    def test_send_hourly_health(self):
        self.monitor.send_hourly_health({
            "bot_name": "TestBot", "uptime": "2h", "total_trades": 50,
            "active_trades": 2, "memory_mb": 256, "exchange": "binance", "exchange_ok": True,
        })
        assert self._count_queue() == 1
        ev = self.monitor._queue[-1]
        assert ev.source == "monitor"
        assert ev.event_type == "bot_health"

    def test_digest_on_flush(self):
        self.monitor.notify_memory_match("topic", "insight", 0.85)
        self.monitor.notify_knowledge_added("topic", "insight", "src")
        self.monitor.flush()
        count = self._count_queue()
        assert count == 0

    def test_event_data_is_valid_telegram_format(self):
        self.monitor.notify_regime_change("BULL", "BEAR", 0.85, "conservative", {"trend": 0.5, "momentum": 0.3, "hmm": 0.2})
        ev = self.monitor._queue[-1]
        assert len(ev.message) > 10
        assert ev.timestamp is not None
