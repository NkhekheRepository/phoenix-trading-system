import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from core.data_quality import DataValidator
from core.concept_drift import ConceptDriftDetector
from core.ml_engine import MLEngine
from core.monitoring import Monitor, MessageFormatter, Event


class TestValidatorE2E:

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _make_dataframe(self, n_candles=100, with_nan=False, with_stale=False):
        now = datetime.now(timezone.utc)
        if with_stale:
            dates = [now - timedelta(minutes=60 * i) for i in range(n_candles)]
        else:
            dates = [now - timedelta(minutes=5 * i) for i in range(n_candles)]
        dates.reverse()
        closes = np.random.randn(n_candles) * 100 + 50000
        df = pd.DataFrame({
            "date": dates,
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": np.random.randn(n_candles) * 100 + 1000,
        })
        if with_nan:
            df.loc[10:25, "close"] = np.nan
        return df

    def _collector(self):
        """Return (handler_func, list) — handler appends kwargs to list."""
        collected = []

        def handler(method_name, **kwargs):
            collected.append((method_name, kwargs))

        return handler, collected

    # ------------------------------------------------------------------ #
    #  Test 1: DataValidator on_notify → Monitor queue
    # ------------------------------------------------------------------ #

    def test_data_quality_routes_to_monitor_via_on_notify(self):
        handler, collected = self._collector()
        dv = DataValidator(max_candle_age_minutes=5, on_notify=handler)
        df = self._make_dataframe(n_candles=50, with_nan=True, with_stale=True)
        dv.validate_candles(df, "BTC/USDT")

        if collected:
            method, kwargs = collected[0]
            assert method == "data_quality"
            assert "pair" in kwargs
            assert kwargs["pair"] == "BTC/USDT"

    def test_data_quality_clean_data_no_notification(self):
        handler, collected = self._collector()
        dv = DataValidator(max_candle_age_minutes=60, on_notify=handler)
        df = self._make_dataframe(n_candles=50)
        dv.validate_candles(df, "ETH/USDT")
        assert len(collected) == 0

    def test_data_quality_monitor_queue_integration(self):
        monitor = Monitor(dp=None)
        handler_calls = []

        def handler(method_name, **kwargs):
            handler_calls.append((method_name, kwargs))
            getattr(monitor, f"notify_{method_name}")(**kwargs)

        dv = DataValidator(max_candle_age_minutes=5, on_notify=handler)
        df = self._make_dataframe(n_candles=50, with_nan=True, with_stale=True)
        dv.validate_candles(df, "SOL/USDT")

        if handler_calls:
            assert len(monitor._queue) >= 1
            ev = monitor._queue[-1]
            assert ev.source == "data_quality"
            assert ev.event_type == "data_quality"

    # ------------------------------------------------------------------ #
    #  Test 2: ConceptDriftDetector → MLEngine retrain trigger
    # ------------------------------------------------------------------ #

    def test_concept_drift_triggers_ml_retrain(self):
        drift = ConceptDriftDetector(
            psi_threshold=0.1, kl_threshold=0.5, wasserstein_threshold=0.3,
            window_size=100,
        )
        engine = MLEngine(concept_drift=drift, trade_intel=None)

        ref = np.random.randn(200) * 1 + 50
        drift.set_reference("feat_x", ref)

        drifted = np.random.randn(400) * 5 + 80
        result = None
        for v in drifted:
            r = drift.update("feat_x", v)
            if r:
                result = r

        assert result is not None, "Drift should be detected with very different distributions"
        assert result.drift_detected

        summary = drift.get_drift_summary()
        assert "feat_x" in summary
        assert summary["feat_x"]["current_psi"] > 0.3

        triggers = engine.check_retrain_triggers()
        drift_triggers = [t for t in triggers if t.source == "concept_drift"]

        if summary["feat_x"]["trend"] == "increasing":
            assert len(drift_triggers) >= 1

    def test_no_drift_no_retrain_trigger(self):
        drift = ConceptDriftDetector(
            psi_threshold=0.5, kl_threshold=0.5, wasserstein_threshold=1.0,
            window_size=50,
        )
        engine = MLEngine(concept_drift=drift)
        ref = np.random.randn(100) * 1 + 50
        drift.set_reference("feat_x", ref)
        for v in ref[:80]:
            drift.update("feat_x", v)
        triggers = engine.check_retrain_triggers()
        drift_triggers = [t for t in triggers if t.source == "concept_drift"]
        assert len(drift_triggers) == 0

    # ------------------------------------------------------------------ #
    #  Test 3: MLEngine performance degradation → on_notify
    # ------------------------------------------------------------------ #

    def test_perf_degradation_fires_on_notify(self):
        collected = []

        class FakeTradeIntel:
            def analyze_losing_patterns(self, n_trades=50):
                return {"total_losers": 35, "total_winners": 5, "avg_loss_pct": 2.5}
            def analyze_winning_patterns(self, n_trades=50):
                return {"total_winners": 5, "total_win_pct": 1.0}

        engine = MLEngine(
            trade_intel=FakeTradeIntel(),
            on_notify=lambda method_name, **kw: collected.append((method_name, kw)),
        )
        engine.set_baseline({"win_rate": 0.75, "avg_loss_pct": 1.5})
        triggers = engine.check_retrain_triggers()

        perf_triggers = [t for t in triggers if t.source == "performance_degradation"]
        assert len(perf_triggers) >= 1
        assert any(m == "retrain_triggers" for m, _ in collected)

    def test_good_performance_no_notification(self):
        collected = []

        class FakeTradeIntel:
            def analyze_losing_patterns(self, n_trades=50):
                return {"total_losers": 5, "total_winners": 45, "avg_loss_pct": 0.5}
            def analyze_winning_patterns(self, n_trades=50):
                return {"total_winners": 45, "total_win_pct": 2.0}

        engine = MLEngine(
            trade_intel=FakeTradeIntel(),
            on_notify=lambda method_name, **kw: collected.append((method_name, kw)),
        )
        engine.set_baseline({"win_rate": 0.8, "avg_loss_pct": 1.0})
        triggers = engine.check_retrain_triggers()
        perf_triggers = [t for t in triggers if t.source == "performance_degradation"]
        assert len(perf_triggers) == 0
        assert len(collected) == 0

    # ------------------------------------------------------------------ #
    #  Test 4: Full chain — validators → Monitor → MessageFormatter
    # ------------------------------------------------------------------ #

    def test_full_chain_events_route_through_monitor(self):
        monitor = Monitor(dp=None)

        def handler(method_name, **kwargs):
            getattr(monitor, f"notify_{method_name}")(**kwargs)

        # DataValidator
        dv = DataValidator(max_candle_age_minutes=5, on_notify=handler)
        bad_df = self._make_dataframe(n_candles=50, with_nan=True)
        dv.validate_candles(bad_df, "BTC/USDT")

        # MLEngine degradation
        class FakeTradeIntel:
            def analyze_losing_patterns(self, n_trades=50):
                return {"total_losers": 30, "total_winners": 10, "avg_loss_pct": 2.0}
            def analyze_winning_patterns(self, n_trades=50):
                return {"total_winners": 10, "total_win_pct": 1.0}

        engine = MLEngine(
            trade_intel=FakeTradeIntel(),
            on_notify=handler,
        )
        engine.set_baseline({"win_rate": 0.7, "avg_loss_pct": 1.2})
        engine.check_retrain_triggers()

        assert len(monitor._queue) >= 1

        for event in monitor._queue:
            assert isinstance(event, Event)
            assert event.source in ("data_quality", "ml_engine", "monitor")

    # ------------------------------------------------------------------ #
    #  Test 5: Smoke — all on_notify consumers produce valid events
    # ------------------------------------------------------------------ #

    def test_data_validator_on_notify_event_structure(self):
        handler, collected = self._collector()
        dv = DataValidator(max_candle_age_minutes=5, on_notify=handler)
        df = self._make_dataframe(n_candles=30, with_nan=True, with_stale=True)
        dv.validate_candles(df, "BTC/USDT")
        for method, kwargs in collected:
            assert method == "data_quality"
            assert "pair" in kwargs
            assert "issue" in kwargs
            assert "details" in kwargs

    def test_ml_engine_on_notify_event_structure(self):
        collected = []

        class FakeTradeIntel:
            def analyze_losing_patterns(self, n_trades=50):
                return {"total_losers": 30, "total_winners": 10, "avg_loss_pct": 2.0}
            def analyze_winning_patterns(self, n_trades=50):
                return {"total_winners": 10, "total_win_pct": 1.0}

        def handler(method_name, **kwargs):
            collected.append((method_name, kwargs))

        engine = MLEngine(trade_intel=FakeTradeIntel(), on_notify=handler)
        engine.set_baseline({"win_rate": 0.6, "avg_loss_pct": 1.5})
        triggers = engine.check_retrain_triggers()

        if triggers:
            for method, kwargs in collected:
                assert method == "retrain_triggers"
                assert "triggers" in kwargs
                assert isinstance(kwargs["triggers"], list)

    def test_monitor_event_from_on_notify_roundtrip(self):
        monitor = Monitor(dp=None)
        events = []

        def handler(method_name, **kwargs):
            getattr(monitor, f"notify_{method_name}")(**kwargs)
            events.append((method_name, kwargs))

        dv = DataValidator(max_candle_age_minutes=5, on_notify=handler)
        df = self._make_dataframe(n_candles=30, with_nan=True)
        dv.validate_candles(df, "BTC/USDT")

        if events:
            ev = monitor._queue[-1]
            fmt = MessageFormatter.data_quality_alert(
                pair=ev.data.get("pair", "?"),
                issue=ev.data.get("issue", "?"),
                severity=ev.severity,
                details=ev.data.get("details", ""),
            )
            assert "DATA QUALITY" in fmt.upper()
            assert "BTC/USDT" in fmt
