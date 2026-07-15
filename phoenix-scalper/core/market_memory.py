import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MarketMemory:
    def __init__(self, storage_path: str = "data/market_memory", on_notify=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._memory_file = self.storage_path / "memory.json"
        self._memory: Dict[str, Any] = self._load()
        self.on_notify = on_notify

    def _load(self) -> Dict:
        if self._memory_file.exists():
            try:
                with open(self._memory_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load market memory: {e}")
        return {
            "failed_conditions": [],
            "successful_conditions": [],
            "regime_performance": {},
            "strategy_notes": {},
            "historical_events": [],
            "knowledge_base": [],
        }

    def _save(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self._memory_file, "w") as f:
            json.dump(self._memory, f, indent=2)

    def remember_failed_condition(self, condition: str, regime: str, context: Dict):
        self._memory["failed_conditions"].append({
            "condition": condition,
            "regime": regime,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._memory["failed_conditions"]) > 500:
            self._memory["failed_conditions"] = self._memory["failed_conditions"][-500:]
        self._save()

    def remember_successful_condition(self, condition: str, regime: str, context: Dict):
        self._memory["successful_conditions"].append({
            "condition": condition,
            "regime": regime,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._memory["successful_conditions"]) > 500:
            self._memory["successful_conditions"] = self._memory["successful_conditions"][-500:]
        self._save()

    def update_regime_performance(self, regime: str, profit_pct: float, win: bool):
        if regime not in self._memory["regime_performance"]:
            self._memory["regime_performance"][regime] = {
                "trades": 0, "wins": 0, "losses": 0,
                "total_profit": 0.0, "avg_profit": 0.0,
            }
        rp = self._memory["regime_performance"][regime]
        rp["trades"] += 1
        rp["wins"] += 1 if win else 0
        rp["losses"] += 0 if win else 1
        rp["total_profit"] += profit_pct
        rp["avg_profit"] = rp["total_profit"] / rp["trades"]
        self._save()

    def add_strategy_note(self, strategy: str, note: str):
        if strategy not in self._memory["strategy_notes"]:
            self._memory["strategy_notes"][strategy] = []
        self._memory["strategy_notes"][strategy].append({
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        if self.on_notify:
            self.on_notify("strategy_note", strategy=strategy, note=note)

    def record_event(self, event_type: str, description: str, impact: str = "unknown"):
        self._memory["historical_events"].append({
            "type": event_type,
            "description": description,
            "impact": impact,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._memory["historical_events"]) > 200:
            self._memory["historical_events"] = self._memory["historical_events"][-200:]
        self._save()
        if self.on_notify:
            self.on_notify("market_event", event_type=event_type, description=description, impact=impact)

    def add_knowledge(self, topic: str, insight: str, source: str = "analysis"):
        self._memory["knowledge_base"].append({
            "topic": topic,
            "insight": insight,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        if self.on_notify:
            self.on_notify("knowledge_added", topic=topic, insight=insight, source=source)

    def query(self, condition: str, regime: str = None) -> Dict:
        results = {"failure_count": 0, "success_count": 0, "notes": []}

        for entry in self._memory["failed_conditions"]:
            if condition.lower() in entry["condition"].lower():
                if regime is None or entry["regime"] == regime:
                    results["failure_count"] += 1

        for entry in self._memory["successful_conditions"]:
            if condition.lower() in entry["condition"].lower():
                if regime is None or entry["regime"] == regime:
                    results["success_count"] += 1

        if regime and regime in self._memory["regime_performance"]:
            rp = self._memory["regime_performance"][regime]
            results["win_rate"] = rp["wins"] / rp["trades"] if rp["trades"] > 0 else 0
            results["avg_profit"] = rp["avg_profit"]
            results["total_trades"] = rp["trades"]

        if self.on_notify and regime and results.get("win_rate") is not None:
            win_rate = results.get("win_rate", 0)
            total = results.get("total_trades", 0)
            if total >= 5 and (win_rate > 0.7 or win_rate < 0.3):
                self.on_notify(
                    "memory_match",
                    topic=f"Regime {regime} {condition}",
                    insight=f"Win rate {win_rate:.0%} over {total} trades",
                    confidence=min(0.5 + abs(win_rate - 0.5) * 2, 0.95),
                )

        return results

    def get_summary(self) -> Dict:
        return {
            "failed_conditions": len(self._memory["failed_conditions"]),
            "successful_conditions": len(self._memory["successful_conditions"]),
            "regimes_tracked": list(self._memory["regime_performance"].keys()),
            "regime_trades": {r: v["trades"] for r, v in self._memory["regime_performance"].items()},
            "events": len(self._memory["historical_events"]),
            "knowledge_items": len(self._memory["knowledge_base"]),
        }
