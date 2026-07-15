import pytest
from core.risk_governor import RiskGovernor, RiskLevel


class TestRiskGovernor:
    def setup_method(self):
        self.governor = RiskGovernor(
            max_daily_drawdown=0.05,
            max_weekly_drawdown=0.10,
            max_consecutive_losses=5,
        )

    def test_normal_state(self):
        state = self.governor.update(10000, [])
        assert state.level == RiskLevel.NORMAL
        assert state.recommended_leverage_mult == 1.0

    def test_consecutive_losses_triggers_reduced(self):
        for _ in range(5):
            self.governor.record_trade_result(-2.0, 10000)
        state = self.governor.update(10000, [])
        assert state.level in (RiskLevel.REDUCED, RiskLevel.PAUSED)

    def test_should_allow_entry_normal(self):
        self.governor.update(10000, [])
        assert self.governor.should_allow_entry() is True

    def test_should_deny_entry_emergency(self):
        self.governor.max_consecutive_losses = 2
        for _ in range(10):
            self.governor.record_trade_result(-5.0, 10000)
        state = self.governor.update(8000, [])
        if state.level in (RiskLevel.PAUSED, RiskLevel.EMERGENCY):
            assert self.governor.should_allow_entry() is False

    def test_exposure_limit(self):
        trades = [{"stake_amount": 9000, "leverage": 10}]
        state = self.governor.update(10000, trades)
        assert state.current_exposure >= 0.8

    def test_emergency_close(self):
        self.governor.max_consecutive_losses = 2
        for _ in range(10):
            self.governor.record_trade_result(-10.0, 10000)
        state = self.governor.update(5000, [])
        if state.level == RiskLevel.EMERGENCY:
            assert self.governor.should_emergency_close() is True
