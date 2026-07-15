import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrainTrigger:
    source: str
    severity: str
    metric: str
    value: float
    threshold: float
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeatureRecommendation:
    keep: List[str]
    remove: List[str]
    add: List[str]
    importance_scores: Dict[str, float]


@dataclass
class EnsembleWeights:
    weights: Dict[str, float]
    reason: str


class MLEngine:
    def __init__(self, trade_intel=None, concept_drift=None, experiment_db=None, market_memory=None, on_notify=None):
        self.trade_intel = trade_intel
        self.concept_drift = concept_drift
        self.experiment_db = experiment_db
        self.market_memory = market_memory
        self.on_notify = on_notify
        self._last_check: Optional[datetime] = None
        self._trigger_history: List[RetrainTrigger] = []
        self._baseline_performance: Dict[str, float] = {}
        self._feature_importance_history: Dict[str, List[float]] = {}

    def set_baseline(self, metrics: Dict[str, float]):
        self._baseline_performance = metrics.copy()

    def check_retrain_triggers(self, trade_data: Optional[List[Dict]] = None) -> List[RetrainTrigger]:
        triggers = []

        perf = self.performance_degradation(trade_data)
        if perf:
            triggers.append(perf)

        drift = self.concept_drift_detected()
        if drift:
            triggers.append(drift)

        regime = self.regime_change_significant()
        if regime:
            triggers.append(regime)

        validated = self.new_validated_data()
        if validated:
            triggers.append(validated)

        self._trigger_history.extend(triggers)
        if len(self._trigger_history) > 200:
            self._trigger_history = self._trigger_history[-200:]

        if triggers and self.on_notify:
            self.on_notify("retrain_triggers", triggers=[t.__dict__ for t in triggers])

        return triggers

    def performance_degradation(self, trade_data: Optional[List[Dict]] = None) -> Optional[RetrainTrigger]:
        if not self.trade_intel or not self._baseline_performance:
            return None

        try:
            losers = self.trade_intel.analyze_losing_patterns(n_trades=50)
            winners = self.trade_intel.analyze_winning_patterns(n_trades=50)

            total_losers = losers.get("total_losers", 0)
            total_winners = winners.get("total_winners", 0)
            recent_total = total_losers + total_winners

            if recent_total < 10:
                return None

            recent_win_rate = total_winners / recent_total
            baseline_win_rate = self._baseline_performance.get("win_rate", recent_win_rate)

            if baseline_win_rate > 0 and recent_win_rate < baseline_win_rate * 0.85:
                return RetrainTrigger(
                    source="performance_degradation",
                    severity="warning",
                    metric="win_rate",
                    value=recent_win_rate,
                    threshold=baseline_win_rate * 0.85,
                    description=f"Win rate dropped {recent_win_rate:.1%} vs baseline {baseline_win_rate:.1%}",
                )

            if losers.get("avg_loss_pct", 0) > self._baseline_performance.get("avg_loss_pct", 0) * 1.3:
                return RetrainTrigger(
                    source="performance_degradation",
                    severity="info",
                    metric="avg_loss",
                    value=losers["avg_loss_pct"],
                    threshold=self._baseline_performance.get("avg_loss_pct", 0) * 1.3,
                    description="Average loss size increased significantly",
                )
        except Exception as e:
            logger.warning(f"Performance check failed: {e}")

        return None

    def concept_drift_detected(self) -> Optional[RetrainTrigger]:
        if not self.concept_drift:
            return None

        try:
            summary = self.concept_drift.get_drift_summary()
            if not summary:
                return None

            critical_drifts = []
            for feature, stats in summary.items():
                if stats.get("current_psi", 0) > 0.3 and stats.get("trend") == "increasing":
                    critical_drifts.append(feature)

            if critical_drifts:
                return RetrainTrigger(
                    source="concept_drift",
                    severity="warning",
                    metric="psi_trend",
                    value=summary.get(critical_drifts[0], {}).get("current_psi", 0),
                    threshold=0.3,
                    description=f"Concept drift in {', '.join(critical_drifts[:3])}",
                )
        except Exception as e:
            logger.warning(f"Drift check failed: {e}")

        return None

    def regime_change_significant(self) -> Optional[RetrainTrigger]:
        if not self.market_memory:
            return None

        try:
            summary = self.market_memory.get_summary()
            regimes = summary.get("regimes_tracked", [])
            if len(regimes) >= 3:
                return RetrainTrigger(
                    source="regime_change",
                    severity="info",
                    metric="regime_count",
                    value=len(regimes),
                    threshold=3,
                    description=f"Multiple regimes encountered ({len(regimes)}): {', '.join(regimes[:5])}",
                )
        except Exception as e:
            logger.warning(f"Regime check failed: {e}")

        return None

    def new_validated_data(self) -> Optional[RetrainTrigger]:
        if not self.experiment_db:
            return None

        try:
            stats = self.experiment_db.get_statistics()
            approved = stats.get("approved", 0)

            recent = self.experiment_db.get_recent_experiments(n=5)
            for exp in recent:
                if exp.decision == "approved" and exp.status == "completed":
                    return RetrainTrigger(
                        source="validated_experiment",
                        severity="info",
                        metric="approved_experiments",
                        value=approved,
                        threshold=1,
                        description=f"New validated experiment: {exp.experiment_id[:12]}",
                    )
        except Exception as e:
            logger.warning(f"Experiment check failed: {e}")

        return None

    def recommend_feature_changes(self, model_feature_importance: Optional[Dict[str, float]] = None) -> FeatureRecommendation:
        importances = model_feature_importance or {}
        if not importances:
            return FeatureRecommendation(keep=[], remove=[], add=[], importance_scores={})

        sorted_features = sorted(importances.items(), key=lambda x: -x[1])
        if not sorted_features:
            return FeatureRecommendation(keep=[], remove=[], add=[], importance_scores={})

        max_imp = max(importances.values())
        threshold = max_imp * 0.05

        keep = []
        remove = []
        for feat, imp in importances.items():
            self._feature_importance_history.setdefault(feat, []).append(imp)
            if len(self._feature_importance_history[feat]) > 50:
                self._feature_importance_history[feat] = self._feature_importance_history[feat][-50:]

            if imp >= threshold:
                keep.append(feat)
            else:
                remove.append(feat)

        return FeatureRecommendation(
            keep=keep,
            remove=remove,
            add=[],
            importance_scores=importances,
        )

    def adjust_ensemble_weights(self, current_weights: Dict[str, float],
                                strategy_performance: Dict[str, Dict]) -> EnsembleWeights:
        if not strategy_performance:
            return EnsembleWeights(weights=current_weights.copy(), reason="no_performance_data")

        adjusted = {}
        total_score = 0.0
        scores = {}

        for strategy, weight in current_weights.items():
            perf = strategy_performance.get(strategy, {})
            sharpe = perf.get("sharpe", 0)
            win_rate = perf.get("win_rate", 0.5)
            trades = perf.get("trades", 0)

            if trades < 5:
                scores[strategy] = weight
            else:
                score = max(0, sharpe * 0.6 + win_rate * 0.4)
                scores[strategy] = max(score, 0.01)
            total_score += scores[strategy]

        if total_score > 0:
            for strategy in current_weights:
                adjusted[strategy] = scores[strategy] / total_score
        else:
            adjusted = current_weights.copy()

        total = sum(adjusted.values())
        if abs(total - 1.0) > 0.001:
            for k in adjusted:
                adjusted[k] /= total

        return EnsembleWeights(
            weights=adjusted,
            reason="adjusted_by_sharpe_winrate" if strategy_performance else "no_change",
        )

    def schedule_retrain(self, trigger: RetrainTrigger, params: Optional[Dict] = None) -> Optional[str]:
        if not self.experiment_db:
            logger.warning("No experiment_db, cannot schedule retrain")
            return None

        exp_id = self.experiment_db.create_experiment(
            dataset_version="live",
            model_version=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            parameters={
                "trigger_source": trigger.source,
                "trigger_metric": trigger.metric,
                "trigger_value": trigger.value,
                "trigger_threshold": trigger.threshold,
                "trigger_description": trigger.description,
                "auto_params": params or {},
            },
            training_period="auto",
            validation_period="auto",
        )

        logger.info(f"Scheduled retrain experiment {exp_id} triggered by {trigger.source}")

        if self.on_notify:
            self.on_notify("retrain_scheduled", experiment_id=exp_id, trigger=trigger.source)

        return exp_id

    def get_trigger_history(self, n: int = 50) -> List[RetrainTrigger]:
        return self._trigger_history[-n:] if self._trigger_history else []

    def get_feature_stability(self, feature: str) -> Dict:
        history = self._feature_importance_history.get(feature, [])
        if not history:
            return {"stable": True, "mean": 0, "std": 0, "trend": "unknown"}
        import numpy as np
        return {
            "stable": np.std(history) < 0.1,
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "trend": "increasing" if len(history) > 2 and history[-1] > np.mean(history[:-1]) else "stable",
        }
