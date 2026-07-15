import logging
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Allocation:
    strategy: str
    weight: float
    reason: str


@dataclass
class AllocationResult:
    allocations: List[Allocation]
    cash_reserve: float
    regime: str
    risk_level: str


class StrategyAllocator:
    def __init__(self, on_notify=None):
        self._strategy_performance: Dict[str, Dict] = {}
        self.on_notify = on_notify

    def allocate(self, regime: str, risk_level: str, confidence: float,
                 strategy_performance: Dict[str, float] = None) -> AllocationResult:
        base_allocation = self._get_base_allocation(regime)

        if strategy_performance:
            base_allocation = self._adjust_for_performance(base_allocation, strategy_performance)

        base_allocation = self._adjust_for_risk(base_allocation, risk_level)

        cash_reserve = self._get_cash_reserve(risk_level)

        allocations = []
        for strategy, weight in base_allocation.items():
            allocations.append(Allocation(
                strategy=strategy,
                weight=weight,
                reason=f"Regime: {regime}, Risk: {risk_level}",
            ))

        result = AllocationResult(
            allocations=allocations,
            cash_reserve=cash_reserve,
            regime=regime,
            risk_level=risk_level,
        )

        if self.on_notify:
            allocs_data = [{"strategy": a.strategy, "weight": a.weight} for a in allocations]
            self.on_notify("allocation", allocations=allocs_data,
                           cash_reserve=cash_reserve, regime=regime, risk=risk_level)

        return result

    def _get_base_allocation(self, regime: str) -> Dict[str, float]:
        allocations = {
            "strong_bull": {"trend_follow": 0.50, "breakout": 0.30, "scalping": 0.15, "cash": 0.05},
            "weak_bull": {"trend_follow": 0.30, "scalping": 0.30, "mean_reversion": 0.20, "cash": 0.20},
            "sideways": {"mean_reversion": 0.35, "scalping": 0.30, "trend_follow": 0.10, "cash": 0.25},
            "weak_bear": {"defensive": 0.30, "scalping": 0.20, "mean_reversion": 0.15, "cash": 0.35},
            "strong_bear": {"defensive": 0.40, "cash": 0.50, "scalping": 0.10},
            "high_volatility": {"defensive": 0.50, "cash": 0.40, "scalping": 0.10},
            "low_volatility": {"scalping": 0.40, "trend_follow": 0.30, "mean_reversion": 0.20, "cash": 0.10},
        }
        return allocations.get(regime, {"scalping": 0.50, "cash": 0.50})

    def _adjust_for_performance(self, allocation: Dict[str, float],
                                 performance: Dict[str, float]) -> Dict[str, float]:
        if not performance:
            return allocation

        total_weight = sum(allocation.values())
        if total_weight <= 0:
            return allocation

        for strategy, weight in list(allocation.items()):
            perf = performance.get(strategy, 0)
            if perf < -5:
                reduced = weight * 0.5
                allocation[strategy] = max(reduced, 0.05)
                excess = weight - allocation[strategy]
                allocation["cash"] = allocation.get("cash", 0) + excess

        total = sum(allocation.values())
        if total > 0:
            for k in allocation:
                allocation[k] /= total

        return allocation

    def _adjust_for_risk(self, allocation: Dict[str, float], risk_level: str) -> Dict[str, float]:
        risk_reduction = {
            "normal": 1.0,
            "caution": 0.75,
            "reduced": 0.50,
            "paused": 0.10,
            "emergency": 0.0,
        }
        mult = risk_reduction.get(risk_level, 1.0)

        if mult >= 1.0:
            return allocation

        for strategy in list(allocation.keys()):
            if strategy != "cash":
                allocation[strategy] *= mult

        cash = 1.0 - sum(allocation.values())
        allocation["cash"] = max(0, cash)

        return allocation

    def _get_cash_reserve(self, risk_level: str) -> float:
        reserves = {
            "normal": 0.05,
            "caution": 0.20,
            "reduced": 0.50,
            "paused": 0.90,
            "emergency": 1.0,
        }
        return reserves.get(risk_level, 0.05)
