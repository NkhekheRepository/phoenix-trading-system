import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    experiment_id: str
    dataset_version: str
    model_version: str
    parameters: Dict[str, Any]
    training_period: str
    validation_period: str
    results: Dict[str, float]
    decision: str
    reason: str
    timestamp: str
    status: str = "completed"


class ExperimentDB:
    def __init__(self, storage_path: str = "data/experiments", on_notify=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_path / "index.json"
        self._experiments: Dict[str, Experiment] = self._load_index()
        self.on_notify = on_notify

    def _load_index(self) -> Dict:
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    data = json.load(f)
                    return {k: Experiment(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load experiment index: {e}")
        return {}

    def _save_index(self):
        with open(self._index_file, "w") as f:
            data = {k: asdict(v) for k, v in self._experiments.items()}
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        dataset_version: str,
        model_version: str,
        parameters: Dict,
        training_period: str,
        validation_period: str,
    ) -> str:
        raw = f"{dataset_version}{model_version}{json.dumps(parameters, sort_keys=True)}{datetime.now().timestamp()}"
        exp_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        experiment = Experiment(
            experiment_id=exp_id,
            dataset_version=dataset_version,
            model_version=model_version,
            parameters=parameters,
            training_period=training_period,
            validation_period=validation_period,
            results={},
            decision="pending",
            reason="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        self._experiments[exp_id] = experiment
        self._save_index()
        return exp_id

    def record_results(self, experiment_id: str, results: Dict[str, float]):
        exp = self._experiments.get(experiment_id)
        if not exp:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        exp.results = results
        self._save_index()

    def conclude_experiment(self, experiment_id: str, decision: str, reason: str):
        exp = self._experiments.get(experiment_id)
        if not exp:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        exp.decision = decision
        exp.reason = reason
        exp.status = "completed"
        self._save_index()

        if self.on_notify:
            self.on_notify("experiment", experiment_id=experiment_id,
                           hypothesis=exp.parameters.get("hypothesis", ""),
                           results=exp.results, decision=decision, reason=reason)

        exp_path = self.storage_path / f"{experiment_id}.json"
        with open(exp_path, "w") as f:
            json.dump(asdict(exp), f, indent=2)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def get_recent_experiments(self, n: int = 20) -> List[Experiment]:
        sorted_exp = sorted(
            self._experiments.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_exp[:n]

    def get_best_experiments(self, metric: str = "sharpe", n: int = 5) -> List[Experiment]:
        with_results = [
            e for e in self._experiments.values()
            if e.results.get(metric) is not None
        ]
        sorted_exp = sorted(with_results, key=lambda e: e.results[metric], reverse=True)
        return sorted_exp[:n]

    def get_statistics(self) -> Dict:
        total = len(self._experiments)
        approved = sum(1 for e in self._experiments.values() if e.decision == "approved")
        rejected = sum(1 for e in self._experiments.values() if e.decision == "rejected")
        return {
            "total_experiments": total,
            "approved": approved,
            "rejected": rejected,
            "pending": total - approved - rejected,
            "approval_rate": approved / total if total > 0 else 0,
        }
