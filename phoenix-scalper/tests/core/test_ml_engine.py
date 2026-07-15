import pytest
import numpy as np
from core.ml_engine import MLEngine, RetrainTrigger, FeatureRecommendation, EnsembleWeights


class TestMLEngine:
    def test_no_triggers_without_data(self):
        engine = MLEngine()
        triggers = engine.check_retrain_triggers()
        assert len(triggers) == 0

    def test_performance_degradation_detected(self):
        class FakeTradeIntel:
            def analyze_losing_patterns(self, n_trades=50):
                return {"total_losers": 30, "total_winners": 10, "avg_loss_pct": 2.0}
            def analyze_winning_patterns(self, n_trades=50):
                return {"total_winners": 10, "total_win_pct": 1.5}

        engine = MLEngine(trade_intel=FakeTradeIntel())
        engine.set_baseline({"win_rate": 0.6, "avg_loss_pct": 1.5})
        triggers = engine.check_retrain_triggers()
        perf_triggers = [t for t in triggers if t.source == "performance_degradation"]
        assert len(perf_triggers) >= 1
        assert perf_triggers[0].metric == "win_rate"

    def test_no_degradation_with_good_performance(self):
        class FakeTradeIntel:
            def analyze_losing_patterns(self, n_trades=50):
                return {"total_losers": 5, "total_winners": 45, "avg_loss_pct": 1.0}
            def analyze_winning_patterns(self, n_trades=50):
                return {"total_winners": 45, "total_win_pct": 2.0}

        engine = MLEngine(trade_intel=FakeTradeIntel())
        engine.set_baseline({"win_rate": 0.8, "avg_loss_pct": 1.5})
        triggers = engine.check_retrain_triggers()
        perf_triggers = [t for t in triggers if t.source == "performance_degradation"]
        assert len(perf_triggers) == 0

    def test_feature_recommendation_removes_low_importance(self):
        engine = MLEngine()
        importances = {"feature_a": 0.5, "feature_b": 0.3, "feature_c": 0.02, "feature_d": 0.01}
        rec = engine.recommend_feature_changes(importances)
        assert "feature_a" in rec.keep
        assert "feature_c" in rec.remove
        assert len(rec.keep) >= 2
        assert len(rec.remove) >= 1

    def test_ensemble_weights_adjust_by_performance(self):
        engine = MLEngine()
        current = {"trend": 0.5, "scalping": 0.3, "mean_reversion": 0.2}
        perf = {
            "trend": {"sharpe": 2.0, "win_rate": 0.7, "trades": 50},
            "scalping": {"sharpe": 0.5, "win_rate": 0.4, "trades": 30},
            "mean_reversion": {"sharpe": 1.0, "win_rate": 0.55, "trades": 20},
        }
        result = engine.adjust_ensemble_weights(current, perf)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.weights["trend"] > result.weights["scalping"]

    def test_ensemble_weights_no_perf_data(self):
        engine = MLEngine()
        current = {"trend": 0.5, "scalping": 0.3, "mean_reversion": 0.2}
        result = engine.adjust_ensemble_weights(current, {})
        for k in current:
            assert k in result.weights
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_schedule_retrain_creates_experiment(self):
        class FakeExperimentDB:
            def __init__(self):
                self.experiments = {}
            def create_experiment(self, dataset_version, model_version, parameters, training_period, validation_period):
                exp_id = "test_exp_123"
                self.experiments[exp_id] = {"params": parameters}
                return exp_id

        engine = MLEngine(experiment_db=FakeExperimentDB())
        trigger = RetrainTrigger(
            source="test", severity="info", metric="test_metric",
            value=0.5, threshold=0.3, description="Test trigger",
        )
        exp_id = engine.schedule_retrain(trigger)
        assert exp_id is not None
        assert exp_id == "test_exp_123"

    def test_feature_stability(self):
        engine = MLEngine()
        for i in range(10):
            engine._feature_importance_history.setdefault("feat_x", []).append(0.5 + np.random.normal(0, 0.02))
        stability = engine.get_feature_stability("feat_x")
        assert "mean" in stability
        assert "std" in stability
        assert stability["stable"]
