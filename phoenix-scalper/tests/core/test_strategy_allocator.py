import pytest
from core.strategy_allocator import StrategyAllocator


class TestStrategyAllocator:
    def setup_method(self):
        self.allocator = StrategyAllocator()

    def test_allocation_plus_cash_sum_to_one(self):
        result = self.allocator.allocate("strong_bull", "normal", 0.9)
        total = sum(a.weight for a in result.allocations)
        assert abs(total - 1.0) < 0.01
        assert result.cash_reserve >= 0

    def test_strong_bull_allocates_trend(self):
        result = self.allocator.allocate("strong_bull", "normal", 0.9)
        strategies = {a.strategy: a.weight for a in result.allocations}
        assert strategies.get("trend_follow", 0) > 0

    def test_strong_bear_allocates_defensive(self):
        result = self.allocator.allocate("strong_bear", "normal", 0.9)
        strategies = {a.strategy: a.weight for a in result.allocations}
        assert strategies.get("defensive", 0) > 0

    def test_emergency_risk_sets_high_cash(self):
        result = self.allocator.allocate("strong_bull", "emergency", 0.9)
        assert result.cash_reserve >= 0.9

    def test_unknown_regime_defaults(self):
        result = self.allocator.allocate("unknown_regime", "normal", 0.5)
        assert len(result.allocations) > 0

    def test_performance_adjustment(self):
        perf = {"trend_follow": -10, "scalping": 5}
        result = self.allocator.allocate("strong_bull", "normal", 0.9, perf)
        strategies = {a.strategy: a.weight for a in result.allocations}
        assert strategies.get("cash", 0) > 0.05
