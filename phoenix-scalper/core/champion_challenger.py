import logging
import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    version: str
    timestamp: str
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_drawdown: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float


@dataclass
class PromotionDecision:
    promote: bool
    champion: str
    challenger: str
    reasons: List[str]
    blocking_reasons: List[str]
    sharpe_test: Dict
    drawdown_test: Dict
    expectancy_test: Dict
    min_trades_check: Dict
    stability_check: Dict


class ChampionChallenger:
    def __init__(self, deployment_manager=None, experiment_db=None, market_memory=None, on_notify=None):
        self.deployment_manager = deployment_manager
        self.experiment_db = experiment_db
        self.market_memory = market_memory
        self.on_notify = on_notify
        self._champion: Optional[PerformanceSnapshot] = None
        self._challengers: Dict[str, PerformanceSnapshot] = {}

    def set_champion(self, version: str, trades: List[Dict]) -> PerformanceSnapshot:
        stats = self._compute_stats(version, trades)
        self._champion = stats
        logger.info(f"Champion set: {version} (trades={stats.total_trades}, sharpe={stats.sharpe_ratio:.2f})")
        return stats

    def register_challenger(self, version: str, trades: List[Dict],
                             validation_report: Optional[Dict] = None) -> PerformanceSnapshot:
        stats = self._compute_stats(version, trades)

        if validation_report and not validation_report.get("passed", False):
            logger.warning(f"Challenger {version} failed validation, registering anyway for tracking")
            stats._validation_failed = True

        self._challengers[version] = stats
        logger.info(f"Challenger registered: {version} (trades={stats.total_trades}, sharpe={stats.sharpe_ratio:.2f})")
        return stats

    def evaluate(self) -> PromotionDecision:
        if not self._champion:
            return PromotionDecision(
                promote=False, champion="", challenger="",
                reasons=["no_champion_defined"],
                blocking_reasons=["No champion set. Use set_champion() first."],
                sharpe_test={}, drawdown_test={}, expectancy_test={},
                min_trades_check={}, stability_check={},
            )

        if not self._challengers:
            return PromotionDecision(
                promote=False, champion=self._champion.version, challenger="",
                reasons=["no_challenger_defined"],
                blocking_reasons=["No challenger registered. Use register_challenger() first."],
                sharpe_test={}, drawdown_test={}, expectancy_test={},
                min_trades_check={}, stability_check={},
            )

        best_challenger = self._select_best_challenger()
        if not best_challenger:
            return PromotionDecision(
                promote=False, champion=self._champion.version, challenger="",
                reasons=["no_valid_challenger"],
                blocking_reasons=["No challenger with sufficient data"],
                sharpe_test={}, drawdown_test={}, expectancy_test={},
                min_trades_check={}, stability_check={},
            )

        candidate = self._challengers[best_challenger]

        reasons = []
        blocking = []

        sharpe_test = self._compare_sharpe(self._champion, candidate)
        if sharpe_test.get("passed", False):
            reasons.append(f"Sharpe {candidate.sharpe_ratio:.2f} >= champion {self._champion.sharpe_ratio:.2f}")
        else:
            blocking.append(f"Sharpe {candidate.sharpe_ratio:.2f} < champion {self._champion.sharpe_ratio:.2f}")

        dd_test = self._compare_drawdown(self._champion, candidate)
        if dd_test.get("passed", False):
            reasons.append(f"Max DD {candidate.max_drawdown:.1%} <= champion {self._champion.max_drawdown:.1%} * 1.1")
        else:
            blocking.append(f"Max DD {candidate.max_drawdown:.1%} > champion {self._champion.max_drawdown:.1%} * 1.1")

        exp_test = self._compare_expectancy(self._champion, candidate)
        if exp_test.get("passed", False):
            reasons.append(f"Expectancy {candidate.expectancy:.4f} >= champion {self._champion.expectancy:.4f}")
        else:
            blocking.append(f"Expectancy {candidate.expectancy:.4f} < champion {self._champion.expectancy:.4f}")

        min_trades_check = self._require_min_trades(candidate, min_n=30)
        if min_trades_check.get("passed", False):
            reasons.append(f"Minimum {candidate.total_trades} trades >= 30")
        else:
            blocking.append(f"Only {candidate.total_trades} trades, need >= 30")

        stability_check = self._check_stability(candidate)
        if stability_check.get("passed", False):
            reasons.append(f"Sharpe stability score {stability_check.get('stability_score', 0):.2f} >= 0.5")
        else:
            blocking.append(f"Sharpe stability {stability_check.get('stability_score', 0):.2f} < 0.5")

        promote = len(blocking) == 0

        decision = PromotionDecision(
            promote=promote,
            champion=self._champion.version,
            challenger=candidate.version,
            reasons=reasons,
            blocking_reasons=blocking,
            sharpe_test=sharpe_test,
            drawdown_test=dd_test,
            expectancy_test=exp_test,
            min_trades_check=min_trades_check,
            stability_check=stability_check,
        )

        if self.on_notify:
            self.on_notify("champion_challenger_eval", promote=promote,
                           champion=decision.champion, challenger=decision.challenger)

        return decision

    def promote(self, decision: PromotionDecision) -> bool:
        if not decision.promote:
            logger.warning("Cannot promote: decision rejected")
            return False

        if not self.deployment_manager:
            logger.warning("Cannot promote: no deployment_manager")
            return False

        version = decision.challenger
        self.deployment_manager.promote_to_shadow(version)
        self.deployment_manager.promote_to_canary(version)
        self.deployment_manager.promote_to_full(version)

        self._champion = self._challengers.get(version)
        self._challengers.pop(version, None)

        if self.market_memory:
            self.market_memory.add_knowledge(
                topic=f"champion_promotion_{version}",
                insight=f"Promoted over {decision.champion}. {len(decision.reasons)} criteria met.",
                source="champion_challenger",
            )

        if self.on_notify:
            self.on_notify("champion_promoted", version=version,
                           previous=decision.champion, reasons=decision.reasons)

        logger.info(f"Challenger {version} promoted to champion")
        return True

    def rollback_if_needed(self, current_trades: List[Dict],
                            baseline_buffer: float = 0.2) -> Optional[str]:
        if not self._champion:
            return None

        current = self._compute_stats("current", current_trades)
        if current.total_trades < 10:
            return None

        champion_sharpe = self._champion.sharpe_ratio
        current_sharpe = current.sharpe_ratio

        threshold = champion_sharpe * (1 - baseline_buffer)

        if current_sharpe < threshold and current_sharpe < 0:
            rollback_version = None
            if self.deployment_manager:
                rollback_version = self.deployment_manager.rollback()

            if self.market_memory:
                self.market_memory.record_event(
                    event_type="auto_rollback",
                    description=f"Sharpe {current_sharpe:.2f} below threshold {threshold:.2f}",
                    impact=f"Rolled back to {rollback_version or 'unknown'}",
                )

            if self.on_notify:
                self.on_notify("auto_rollback", version=rollback_version,
                               sharpe=current_sharpe, threshold=threshold)

            return rollback_version

        return None

    def get_status(self) -> Dict:
        champion_info = None
        if self._champion:
            champion_info = {
                "version": self._champion.version,
                "trades": self._champion.total_trades,
                "sharpe": self._champion.sharpe_ratio,
                "win_rate": self._champion.win_rate,
                "max_drawdown": self._champion.max_drawdown,
                "profit_factor": self._champion.profit_factor,
            }

        challenger_info = []
        for ver, stats in self._challengers.items():
            challenger_info.append({
                "version": stats.version,
                "trades": stats.total_trades,
                "sharpe": stats.sharpe_ratio,
                "win_rate": stats.win_rate,
                "max_drawdown": stats.max_drawdown,
            })

        return {
            "champion": champion_info,
            "challengers": challenger_info,
            "n_challengers": len(self._challengers),
        }

    def _compute_stats(self, version: str, trades: List[Dict]) -> PerformanceSnapshot:
        if not trades:
            return PerformanceSnapshot(
                version=version, timestamp=datetime.now(timezone.utc).isoformat(),
                total_trades=0, win_rate=0, sharpe_ratio=0, max_drawdown=0,
                avg_drawdown=0, profit_factor=1, expectancy=0, avg_win=0, avg_loss=0,
            )

        profits = [t.get("profit_pct", 0) for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        win_rate = len(wins) / len(profits) if profits else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 1 if wins else 0.5

        mean_profit = float(np.mean(profits)) if len(profits) > 0 else 0
        std_profit = float(np.std(profits)) if len(profits) > 1 else 0.01
        sharpe = mean_profit / (std_profit + 1e-10) * math.sqrt(96)
        sharpe = max(min(sharpe, 10), -10)

        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-10)
        max_dd = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0
        avg_dd = float(abs(np.mean(drawdowns))) if len(drawdowns) > 0 else 0

        return PerformanceSnapshot(
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_trades=len(profits),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

    def _select_best_challenger(self) -> Optional[str]:
        valid = []
        for ver, stats in self._challengers.items():
            if stats.total_trades >= 10:
                valid.append((ver, stats.sharpe_ratio))

        if not valid:
            return None

        valid.sort(key=lambda x: -x[1])
        return valid[0][0]

    def _compare_sharpe(self, champion: PerformanceSnapshot, challenger: PerformanceSnapshot) -> Dict:
        passed = challenger.sharpe_ratio >= champion.sharpe_ratio
        return {
            "passed": passed,
            "champion_sharpe": champion.sharpe_ratio,
            "challenger_sharpe": challenger.sharpe_ratio,
            "difference": challenger.sharpe_ratio - champion.sharpe_ratio,
        }

    def _compare_drawdown(self, champion: PerformanceSnapshot, challenger: PerformanceSnapshot) -> Dict:
        threshold = champion.max_drawdown * 1.1
        passed = challenger.max_drawdown <= threshold
        return {
            "passed": passed,
            "champion_max_dd": champion.max_drawdown,
            "challenger_max_dd": challenger.max_drawdown,
            "threshold": threshold,
        }

    def _compare_expectancy(self, champion: PerformanceSnapshot, challenger: PerformanceSnapshot) -> Dict:
        passed = challenger.expectancy >= champion.expectancy
        return {
            "passed": passed,
            "champion_expectancy": champion.expectancy,
            "challenger_expectancy": challenger.expectancy,
        }

    def _require_min_trades(self, challenger: PerformanceSnapshot, min_n: int = 30) -> Dict:
        passed = challenger.total_trades >= min_n
        return {
            "passed": passed,
            "trades": challenger.total_trades,
            "required": min_n,
        }

    def _check_stability(self, challenger: PerformanceSnapshot) -> Dict:
        stability = min(challenger.sharpe_ratio / 2, 1.0) if challenger.sharpe_ratio > 0 else 0
        stability = max(stability, 0)
        return {
            "passed": stability >= 0.5,
            "stability_score": stability,
            "sharpe": challenger.sharpe_ratio,
        }
