import logging
from datetime import datetime, date, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    REDUCED = "reduced"
    PAUSED = "paused"
    EMERGENCY = "emergency"


@dataclass
class RiskState:
    level: RiskLevel = RiskLevel.NORMAL
    daily_drawdown: float = 0.0
    weekly_drawdown: float = 0.0
    consecutive_losses: int = 0
    current_exposure: float = 0.0
    max_exposure: float = 0.0
    current_leverage: float = 0.0
    max_leverage: float = 0.0
    trade_count_today: int = 0
    daily_loss: float = 0.0
    triggers: List[str] = field(default_factory=list)
    recommended_leverage_mult: float = 1.0
    recommended_max_trades: int = 5
    recommended_stake_mult: float = 1.0


class RiskGovernor:
    def __init__(
        self,
        max_daily_drawdown: float = 0.05,
        max_weekly_drawdown: float = 0.10,
        max_consecutive_losses: int = 5,
        max_daily_loss: float = 0.08,
        max_exposure_pct: float = 0.80,
        max_leverage_global: float = 50.0,
        on_notify=None,
    ):
        self.max_daily_drawdown = max_daily_drawdown
        self.max_weekly_drawdown = max_weekly_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss = max_daily_loss
        self.max_exposure_pct = max_exposure_pct
        self.max_leverage_global = max_leverage_global
        self.on_notify = on_notify

        self._state = RiskState()
        self._last_level: Optional[str] = None
        self._start_of_day_balance: Optional[float] = None
        self._start_of_week_balance: Optional[float] = None
        self._current_date: Optional[date] = None
        self._current_week: Optional[int] = None
        self._consecutive_losses_streak: int = 0

    def update(self, current_balance: float, open_trades: List[Dict]) -> RiskState:
        today = datetime.now(timezone.utc).date()
        current_week = today.isocalendar()[1]

        if self._current_date != today:
            self._start_of_day_balance = current_balance
            self._current_date = today
            self._state.trade_count_today = 0
            self._state.daily_loss = 0.0

        if self._current_week != current_week:
            self._start_of_week_balance = current_balance
            self._current_week = current_week

        if self._start_of_day_balance and self._start_of_day_balance > 0:
            self._state.daily_drawdown = max(0, (self._start_of_day_balance - current_balance) / self._start_of_day_balance)

        if self._start_of_week_balance and self._start_of_week_balance > 0:
            self._state.weekly_drawdown = max(0, (self._start_of_week_balance - current_balance) / self._start_of_week_balance)

        self._state.consecutive_losses = self._consecutive_losses_streak

        total_exposure = sum(t.get("stake_amount", 0) for t in open_trades)
        self._state.current_exposure = total_exposure / current_balance if current_balance > 0 else 0

        leverages = [t.get("leverage", 1) for t in open_trades]
        self._state.current_leverage = max(leverages) if leverages else 0

        self._state.triggers = []
        self._evaluate_risk()

        new_level = self._state.level.value
        if self._last_level is not None and self._last_level != new_level and self.on_notify:
            impact = {
                "leverage_mult": self._state.recommended_leverage_mult,
                "max_trades": self._state.recommended_max_trades,
                "stake_mult": self._state.recommended_stake_mult,
            }
            self.on_notify("risk_change", level=new_level, triggers=self._state.triggers, impact=impact)
        self._last_level = new_level

        return self._state

    def _evaluate_risk(self):
        triggers = []

        if self._state.daily_drawdown >= self.max_daily_drawdown:
            triggers.append(f"Daily drawdown {self._state.daily_drawdown:.1%} >= {self.max_daily_drawdown:.1%}")

        if self._state.weekly_drawdown >= self.max_weekly_drawdown:
            triggers.append(f"Weekly drawdown {self._state.weekly_drawdown:.1%} >= {self.max_weekly_drawdown:.1%}")

        if self._state.consecutive_losses >= self.max_consecutive_losses:
            triggers.append(f"Consecutive losses {self._state.consecutive_losses} >= {self.max_consecutive_losses}")

        if self._state.daily_loss >= self.max_daily_loss:
            triggers.append(f"Daily loss {self._state.daily_loss:.1%} >= {self.max_daily_loss:.1%}")

        if self._state.current_exposure >= self.max_exposure_pct:
            triggers.append(f"Exposure {self._state.current_exposure:.1%} >= {self.max_exposure_pct:.1%}")

        if self._state.current_leverage >= self.max_leverage_global * 0.9:
            triggers.append(f"Leverage {self._state.current_leverage:.0f}x near limit {self.max_leverage_global:.0f}x")

        self._state.triggers = triggers

        if len(triggers) >= 3:
            self._state.level = RiskLevel.EMERGENCY
            self._state.recommended_leverage_mult = 0.0
            self._state.recommended_max_trades = 0
            self._state.recommended_stake_mult = 0.0
        elif len(triggers) >= 2:
            self._state.level = RiskLevel.PAUSED
            self._state.recommended_leverage_mult = 0.25
            self._state.recommended_max_trades = 1
            self._state.recommended_stake_mult = 0.25
        elif len(triggers) >= 1:
            self._state.level = RiskLevel.REDUCED
            self._state.recommended_leverage_mult = 0.5
            self._state.recommended_max_trades = 2
            self._state.recommended_stake_mult = 0.5
        else:
            self._state.level = RiskLevel.NORMAL
            self._state.recommended_leverage_mult = 1.0
            self._state.recommended_max_trades = 5
            self._state.recommended_stake_mult = 1.0

    def record_trade_result(self, profit_pct: float, current_balance: float):
        self._state.trade_count_today += 1

        if profit_pct < 0:
            self._consecutive_losses_streak += 1
            self._state.daily_loss += abs(profit_pct) / 100
        else:
            self._consecutive_losses_streak = 0

    def should_allow_entry(self) -> bool:
        return self._state.level not in (RiskLevel.PAUSED, RiskLevel.EMERGENCY)

    def should_emergency_close(self) -> bool:
        return self._state.level == RiskLevel.EMERGENCY

    def get_state(self) -> RiskState:
        return self._state
