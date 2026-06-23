import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    profit_pct: float
    win: bool
    duration_hours: float
    entry_price: float
    exit_price: float
    regime: int
    kf_direction: int
    kf_confidence: float


class MonteCarloValidator:
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate_trade_sequences(self, trade_results: List[TradeResult]) -> Dict:
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations...")

        profits = np.array([t.profit_pct for t in trade_results])
        wins = np.array([t.win for t in trade_results])
        durations = np.array([t.duration_hours for t in trade_results])
        regimes = np.array([t.regime for t in trade_results])
        kf_dirs = np.array([t.kf_direction for t in trade_results])
        kf_confs = np.array([t.kf_confidence for t in trade_results])

        results = {
            'max_dd': [], 'sharpe': [], 'calmar': [], 'final_equity': [],
            'ruin_prob': [], 'win_streak_max': [], 'loss_streak_max': [],
            'win_rate': [], 'avg_win': [], 'avg_loss': [], 'profit_factor': [],
        }

        for sim in range(self.n_simulations):
            indices = np.random.permutation(len(profits))
            shuffled_profits = profits[indices]
            shuffled_wins = wins[indices]

            equity = [1000.0]
            for profit in shuffled_profits:
                equity.append(equity[-1] * (1 + profit / 100))
            equity = np.array(equity)

            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dd = dd.max()

            returns_arr = np.diff(equity) / equity[:-1]
            if np.std(returns_arr) > 0:
                sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252 * 288)
            else:
                sharpe = 0.0

            annual_return = (equity[-1] / equity[0]) ** (365 / len(trade_results) * 288) - 1
            calmar = annual_return / (max_dd + 1e-10)

            final_equity = equity[-1]
            ruin_prob = 1 if min(equity) < 500 else 0

            win_streak_max = self._max_streak(shuffled_wins, True)
            loss_streak_max = self._max_streak(shuffled_wins, False)

            win_rate = np.mean(shuffled_wins)
            avg_win = np.mean(shuffled_profits[shuffled_wins]) if np.any(shuffled_wins) else 0
            avg_loss = np.mean(shuffled_profits[~shuffled_wins]) if np.any(~shuffled_wins) else 0

            total_profit = np.sum(shuffled_profits[shuffled_wins])
            total_loss = abs(np.sum(shuffled_profits[~shuffled_wins]))
            profit_factor = total_profit / (total_loss + 1e-10)

            results['max_dd'].append(max_dd)
            results['sharpe'].append(sharpe)
            results['calmar'].append(calmar)
            results['final_equity'].append(final_equity)
            results['ruin_prob'].append(ruin_prob)
            results['win_streak_max'].append(win_streak_max)
            results['loss_streak_max'].append(loss_streak_max)
            results['win_rate'].append(win_rate)
            results['avg_win'].append(avg_win)
            results['avg_loss'].append(avg_loss)
            results['profit_factor'].append(profit_factor)

        return results

    def _max_streak(self, array: np.ndarray, target: bool) -> int:
        max_streak = 0
        current_streak = 0
        for val in array:
            if val == target:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def validate_targets(self, results: Dict) -> Dict:
        validation = {}

        sharpe_p10 = np.percentile(results['sharpe'], 10)
        validation['sharpe_p10'] = sharpe_p10
        validation['meets_sharpe_target'] = sharpe_p10 > 2.0

        calmar_p50 = np.percentile(results['calmar'], 50)
        validation['calmar_p50'] = calmar_p50
        validation['meets_calmar_target'] = calmar_p50 > 2.0

        max_dd_p95 = np.percentile(results['max_dd'], 95)
        validation['max_dd_p95'] = max_dd_p95
        validation['meets_dd_target'] = max_dd_p95 < 0.15

        ruin_prob = np.mean(results['ruin_prob'])
        validation['ruin_prob'] = ruin_prob
        validation['meets_ruin_target'] = ruin_prob < 0.1

        win_rate_p50 = np.percentile(results['win_rate'], 50)
        validation['win_rate_p50'] = win_rate_p50
        validation['meets_winrate_target'] = win_rate_p50 > 0.86

        pf_p50 = np.percentile(results['profit_factor'], 50)
        validation['profit_factor_p50'] = pf_p50
        validation['meets_pf_target'] = pf_p50 > 1.5

        validation['overall_success'] = all([
            validation['meets_sharpe_target'], validation['meets_calmar_target'],
            validation['meets_dd_target'], validation['meets_ruin_target'],
            validation['meets_winrate_target'], validation['meets_pf_target'],
        ])

        return validation

    def generate_report(self, results: Dict, validation: Dict) -> str:
        report = []
        report.append("=" * 60)
        report.append("MONTE CARLO VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Simulations: {self.n_simulations:,}")
        report.append("")

        report.append("KEY METRICS (percentiles):")
        report.append(f"  Sharpe Ratio: {np.percentile(results['sharpe'], 50):.2f} (p10: {np.percentile(results['sharpe'], 10):.2f})")
        report.append(f"  Calmar Ratio: {np.percentile(results['calmar'], 50):.2f}")
        report.append(f"  Max Drawdown: {np.percentile(results['max_dd'], 50):.1%} (p95: {np.percentile(results['max_dd'], 95):.1%})")
        report.append(f"  Win Rate: {np.percentile(results['win_rate'], 50):.1%}")
        report.append(f"  Profit Factor: {np.percentile(results['profit_factor'], 50):.2f}")
        report.append(f"  Ruin Probability: {np.mean(results['ruin_prob']):.1%}")
        report.append("")

        report.append("RISK METRICS (95th percentile):")
        report.append(f"  Max Drawdown: {np.percentile(results['max_dd'], 95):.1%}")
        report.append(f"  Max Win Streak: {np.percentile(results['win_streak_max'], 95)}")
        report.append(f"  Max Loss Streak: {np.percentile(results['loss_streak_max'], 95)}")
        report.append("")

        report.append("TARGET VALIDATION:")
        report.append(f"  Sharpe > 2.0 (p10): {'PASS' if validation['meets_sharpe_target'] else 'FAIL'} ({np.percentile(results['sharpe'], 10):.2f})")
        report.append(f"  Calmar > 2.0 (p50): {'PASS' if validation['meets_calmar_target'] else 'FAIL'} ({np.percentile(results['calmar'], 50):.2f})")
        report.append(f"  Max DD < 15% (p95): {'PASS' if validation['meets_dd_target'] else 'FAIL'} ({np.percentile(results['max_dd'], 95):.1%})")
        report.append(f"  Ruin < 10%: {'PASS' if validation['meets_ruin_target'] else 'FAIL'} ({np.mean(results['ruin_prob']):.1%})")
        report.append(f"  Win Rate > 86% (p50): {'PASS' if validation['meets_winrate_target'] else 'FAIL'} ({np.percentile(results['win_rate'], 50):.1%})")
        report.append(f"  Profit Factor > 1.5 (p50): {'PASS' if validation['meets_pf_target'] else 'FAIL'} ({np.percentile(results['profit_factor'], 50):.2f})")
        report.append("")

        report.append("OVERALL ASSESSMENT:")
        if validation['overall_success']:
            report.append("  ALL TARGETS MET - Strategy is ready for production!")
        else:
            report.append("  SOME TARGETS NOT MET - Strategy needs refinement")
            unmet = []
            if not validation['meets_sharpe_target']: unmet.append("Sharpe > 2.0")
            if not validation['meets_calmar_target']: unmet.append("Calmar > 2.0")
            if not validation['meets_dd_target']: unmet.append("Max DD < 15%")
            if not validation['meets_ruin_target']: unmet.append("Ruin < 10%")
            if not validation['meets_winrate_target']: unmet.append("Win Rate > 86%")
            if not validation['meets_pf_target']: unmet.append("Profit Factor > 1.5")
            report.append(f"  Unmet targets: {', '.join(unmet)}")

        report.append("=" * 60)
        return "\n".join(report)


def create_sample_trade_results(n_samples: int = 1000) -> List[TradeResult]:
    np.random.seed(42)
    trades = []
    for i in range(n_samples):
        if np.random.random() < 0.388:
            profit_pct = np.random.beta(2, 5) * 15
            win = True
        else:
            profit_pct = -np.random.beta(2, 3) * 8
            win = False
        regime = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        if regime == 0:
            kf_direction = np.random.choice([-1, 0, 1], p=[0.1, 0.2, 0.7])
        elif regime == 1:
            kf_direction = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        else:
            kf_direction = np.random.choice([-1, 0, 1], p=[0.6, 0.2, 0.2])
        kf_confidence = np.random.beta(2, 2)
        duration_hours = np.random.gamma(shape=2, scale=20)
        trade = TradeResult(
            profit_pct=profit_pct, win=win, duration_hours=duration_hours,
            entry_price=10000.0, exit_price=10000.0 * (1 + profit_pct/100),
            regime=regime, kf_direction=kf_direction, kf_confidence=kf_confidence,
        )
        trades.append(trade)
    return trades


def main():
    logger.info("Starting Monte Carlo validation...")
    trade_results = create_sample_trade_results(1000)
    validator = MonteCarloValidator(n_simulations=10000)
    results = validator.simulate_trade_sequences(trade_results)
    validation = validator.validate_targets(results)
    report = validator.generate_report(results, validation)
    print(report)


if __name__ == "__main__":
    main()
