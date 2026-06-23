import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RegimeAdaptiveExit:
    def get_regime_specific_exit(self, regime: str, profit_pct: float, hold_minutes: float, kf_state: dict) -> Optional[str]:
        if "Bear" in regime:
            return "regime_bear_exit"

        if "Ranging" in regime:
            if profit_pct > 0.005:
                return "take_profit_range"
            if hold_minutes > 60 and profit_pct < 0:
                return "time_exit_range"

        if "Bull" in regime:
            if hold_minutes > 120 and profit_pct < 0.002:
                return "stall_exit"
            if kf_state.get('kf_direction', 0) == -1 and kf_state.get('kf_confidence', 0.5) > 0.7:
                return "kalman_reversal_exit"

        kf_dir = kf_state.get('kf_direction', 0)
        kf_conf = kf_state.get('kf_confidence', 0.5)
        if kf_dir == -1 and kf_conf > 0.8 and profit_pct < 0:
            return "kalman_high_conf_reversal"

        return None
