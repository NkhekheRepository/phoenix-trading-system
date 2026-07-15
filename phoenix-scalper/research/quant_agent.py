import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class QuantResearchAgent:
    def __init__(self, trade_intel, experiment_db, market_memory, validation_pipeline=None, ml_engine=None, on_notify=None):
        self._trade_intel = trade_intel
        self._experiment_db = experiment_db
        self._market_memory = market_memory
        self._validation_pipeline = validation_pipeline
        self._ml_engine = ml_engine
        self.on_notify = on_notify
        self._hypotheses = []

    def analyze_losing_patterns(self, n_trades: int = 50) -> Dict:
        analysis = self._trade_intel.analyze_losing_patterns(n_trades)
        self._market_memory.record_event("analysis", f"Loss pattern analysis on {n_trades} trades")
        return analysis

    def generate_hypotheses(self, analysis: Dict) -> List[Dict]:
        hypotheses = []
        factors = analysis.get("failure_factors", {})
        worst_regimes = analysis.get("worst_regimes", {})
        loss_rate = analysis.get("loss_rate", 0)

        if loss_rate > 0.5:
            hypotheses.append({
                "type": "parameter_tuning",
                "observation": f"Loss rate {loss_rate:.1%} exceeds 50%",
                "hypothesis": "Entry conditions too loose, tighten signal thresholds",
                "suggested_experiment": "Increase ADX threshold by 20%, reduce volume_factor by 10%",
                "priority": "high",
            })

        for factor, count in factors.items():
            if count >= 3:
                hypotheses.append({
                    "type": "feature_weakness",
                    "observation": f"Failure factor '{factor}' appears {count} times",
                    "hypothesis": f"Conditions related to {factor} need adjustment",
                    "suggested_experiment": f"Test alternate {factor} parameter ranges",
                    "priority": "medium",
                })

        for regime, count in worst_regimes.items():
            if count >= 3:
                hypotheses.append({
                    "type": "regime_weakness",
                    "observation": f"High loss count {count} in regime '{regime}'",
                    "hypothesis": f"Strategy underperforms in {regime} regime",
                    "suggested_experiment": f"Add regime-specific exit or reduce position size in {regime}",
                    "priority": "high",
                })

        self._hypotheses.extend(hypotheses)

        if self.on_notify and hypotheses:
            for h in hypotheses[:3]:
                self.on_notify("research", hypothesis=h)

        return hypotheses

    def design_experiment(self, hypothesis: Dict) -> Optional[str]:
        exp_id = self._experiment_db.create_experiment(
            dataset_version="current",
            model_version="current",
            parameters=hypothesis.get("suggested_experiment", {}),
            training_period="last_30d",
            validation_period="last_7d",
        )
        logger.info(f"Created experiment {exp_id} for hypothesis: {hypothesis.get('hypothesis')}")

        if self._validation_pipeline:
            try:
                report = self._validation_pipeline.validate_strategy(
                    strategy_version=exp_id,
                    strategy_params=hypothesis,
                    historical_data=None,
                )
                logger.info(f"Validation for {exp_id}: {report.gates_passed}/{report.gates_total} passed")
            except Exception as e:
                logger.warning(f"Validation pipeline failed for {exp_id}: {e}")

        return exp_id

    def evaluate_result(self, experiment_id: str, results: Dict) -> Dict:
        self._experiment_db.record_results(experiment_id, results)

        sharpe = results.get("sharpe", 0)
        win_rate = results.get("win_rate", 0)
        max_dd = results.get("max_drawdown", 1)

        if sharpe > 2.0 and win_rate > 0.5 and max_dd < 0.15:
            decision = "approved"
            reason = f"Sharpe {sharpe:.2f}, WR {win_rate:.1%}, DD {max_dd:.1%} meet targets"
            self._market_memory.remember_successful_condition(
                experiment_id, "all", results
            )
        elif sharpe < 0.5 or max_dd > 0.25:
            decision = "rejected"
            reason = f"Failed validation: Sharpe {sharpe:.2f}, DD {max_dd:.1%}"
            self._market_memory.remember_failed_condition(
                experiment_id, "all", results
            )
        else:
            decision = "pending"
            reason = "Inconclusive, needs more data"

        self._experiment_db.conclude_experiment(experiment_id, decision, reason)

        if self._ml_engine and decision == "approved":
            self._ml_engine.schedule_retrain(
                trigger=self._ml_engine.RetrainTrigger(
                    source="quant_agent",
                    severity="info",
                    metric="experiment_approved",
                    value=sharpe,
                    threshold=2.0,
                    description=f"Experiment {experiment_id} approved: {reason}",
                ) if hasattr(self._ml_engine, 'RetrainTrigger') else None,
            )

        return {"decision": decision, "reason": reason, "experiment_id": experiment_id}
