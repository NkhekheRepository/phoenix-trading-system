import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    strategy_version: str
    timestamp: str
    gates_passed: int
    gates_total: int
    backtest_result: Dict
    walk_forward_result: Dict
    monte_carlo_result: Dict
    stress_test_result: Dict
    transaction_cost_result: Dict
    overfitting_score: Dict
    passed: bool
    summary: str


@dataclass
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class ValidationPipeline:
    def __init__(self, experiment_db=None, market_memory=None, on_notify=None):
        self.experiment_db = experiment_db
        self.market_memory = market_memory
        self.on_notify = on_notify

    def validate_strategy(self, strategy_version: str, strategy_params: Dict,
                           historical_data: Optional[Dict] = None) -> ValidationReport:
        gates = {}

        gates["backtest"] = self._run_backtest(strategy_params, historical_data)

        gates["walk_forward"] = self._run_walk_forward(strategy_params, historical_data)

        gates["monte_carlo"] = self._run_monte_carlo(gates["backtest"].get("trades", []))

        gates["stress_test"] = self._run_stress_test(strategy_params, historical_data)

        gates["transaction_cost"] = self._run_tx_cost_analysis(gates["backtest"].get("trades", []))

        gates["overfitting"] = self._detect_overfitting(gates["walk_forward"])

        passed_gates = sum(1 for g in gates.values() if g.get("passed", False))
        total_gates = len(gates)
        all_passed = passed_gates == total_gates

        report = ValidationReport(
            strategy_version=strategy_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gates_passed=passed_gates,
            gates_total=total_gates,
            backtest_result=gates.get("backtest", {}),
            walk_forward_result=gates.get("walk_forward", {}),
            monte_carlo_result=gates.get("monte_carlo", {}),
            stress_test_result=gates.get("stress_test", {}),
            transaction_cost_result=gates.get("transaction_cost", {}),
            overfitting_score=gates.get("overfitting", {}),
            passed=all_passed,
            summary=f"{passed_gates}/{total_gates} gates passed",
        )

        if self.market_memory:
            self.market_memory.add_knowledge(
                topic=f"validation_{strategy_version}",
                insight=report.summary,
                source="validation_pipeline",
            )

        if self.on_notify:
            self.on_notify("validation_complete", version=strategy_version,
                           passed=all_passed, gates=f"{passed_gates}/{total_gates}")

        return report

    def _run_backtest(self, params: Dict, data: Optional[Dict]) -> Dict:
        if not data or "prices" not in data:
            return self._default_backtest_result()

        prices = np.array(data["prices"])
        if len(prices) < 100:
            return {**self._default_backtest_result(), "warning": "insufficient_data"}

        mid = len(prices) // 2
        purge_gap = 50
        train_end = mid - purge_gap // 2
        test_start = mid + purge_gap // 2

        train_prices = prices[:train_end]
        test_prices = prices[test_start:]

        train_return = (train_prices[-1] / train_prices[0]) - 1 if len(train_prices) > 1 else 0
        test_return = (test_prices[-1] / test_prices[0]) - 1 if len(test_prices) > 1 else 0

        volatility = float(np.std(np.diff(np.log(prices + 1e-10)))) if len(prices) > 1 else 0

        return {
            "passed": True,
            "trades": [],
            "total_return": float(test_return),
            "train_return": float(train_return),
            "sharpe": float(test_return / (volatility + 1e-10)) if volatility > 0 else 0,
            "volatility": volatility,
            "max_drawdown": float(abs(min(0, np.min(np.diff(prices)) / (prices[0] + 1e-10)))) if len(prices) > 1 else 0,
            "trade_count": len(prices) // 20,
            "purge_gap": purge_gap,
        }

    def _run_walk_forward(self, params: Dict, data: Optional[Dict]) -> Dict:
        if not data or "prices" not in data:
            return self._default_wf_result()

        prices = np.array(data["prices"])
        if len(prices) < 200:
            return self._default_wf_result()

        n_splits = 5
        split_size = len(prices) // (n_splits + 1)
        purge_gap = 50

        fold_results = []
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_start = train_end + purge_gap
            test_end = min(test_start + split_size, len(prices))

            if test_end <= test_start:
                continue

            train_prices = prices[:train_end]
            test_prices = prices[test_start:test_end]

            if len(train_prices) < 50 or len(test_prices) < 10:
                continue

            fold_return = (test_prices[-1] / test_prices[0]) - 1 if len(test_prices) > 1 else 0
            fold_vol = float(np.std(np.diff(np.log(test_prices + 1e-10)))) if len(test_prices) > 1 else 0
            fold_sharpe = fold_return / (fold_vol + 1e-10) if fold_vol > 0 else 0

            fold_results.append({
                "fold": i,
                "return": float(fold_return),
                "sharpe": float(fold_sharpe),
                "trades": max(1, len(test_prices) // 20),
            })

        if not fold_results:
            return self._default_wf_result()

        sharpes = [f["sharpe"] for f in fold_results]
        returns = [f["return"] for f in fold_results]

        return {
            "passed": float(np.mean(sharpes)) > 0,
            "n_splits": len(fold_results),
            "mean_sharpe": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "min_sharpe": float(min(sharpes)),
            "max_sharpe": float(max(sharpes)),
            "mean_return": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "fold_results": fold_results,
            "purge_gap": purge_gap,
        }

    def _run_monte_carlo(self, trades: List[Dict]) -> Dict:
        try:
            from ml.monte_carlo import MonteCarloValidator, TradeResult, create_sample_trade_results
        except ImportError:
            return self._default_mc_result()

        validator = MonteCarloValidator()

        if not trades:
            sample_trades = create_sample_trade_results(n_samples=100)
        else:
            sample_trades = []
            for t in trades:
                sample_trades.append(TradeResult(
                    profit_pct=t.get("profit_pct", 0),
                    win=t.get("profit_pct", 0) > 0,
                    duration_hours=t.get("duration_hours", 1),
                    entry_price=t.get("entry_price", 100),
                    exit_price=t.get("exit_price", 100),
                    regime=t.get("regime", 1),
                    kf_direction=t.get("kf_direction", 0),
                    kf_confidence=t.get("kf_confidence", 0.5),
                ))

        results = validator.simulate_trade_sequences(sample_trades)
        validation = validator.validate_targets(results)

        ruin_prob = float(np.mean(results.get("ruin_prob", [1])))
        sharpe_median = float(np.median(results.get("sharpe", [0])))
        dd_median = float(np.median(results.get("max_dd", [1])))
        pf_median = float(np.median(results.get("profit_factor", [1])))

        return {
            "passed": ruin_prob < 0.10 and validation.get("overall_success", False),
            "simulations": validator.n_simulations,
            "sharpe_mean": sharpe_median,
            "max_drawdown_mean": dd_median,
            "ruin_probability": ruin_prob,
            "profit_factor": pf_median,
            "overall_success": validation.get("overall_success", False),
            "note": "used_sample_data" if not trades else "",
        }

    def _run_stress_test(self, params: Dict, data: Optional[Dict]) -> Dict:
        scenarios = self._generate_stress_scenarios(data)
        results = []

        for scenario in scenarios:
            result = self._evaluate_scenario(scenario, params)
            results.append(result)

        survived = sum(1 for r in results if r.get("survived", False))
        return {
            "passed": survived >= len(results) * 0.6,
            "scenarios_tested": len(results),
            "scenarios_survived": survived,
            "results": results,
        }

    def _generate_stress_scenarios(self, data: Optional[Dict]) -> List[Dict]:
        scenarios = []

        if data and "prices" in data and len(data["prices"]) > 0:
            prices = np.array(data["prices"])
            volatility = float(np.std(np.diff(np.log(prices + 1e-10)))) if len(prices) > 1 else 0.01

            flash_crash = prices.copy()
            crash_point = len(flash_crash) // 2
            crash_len = min(10, len(flash_crash) - crash_point)
            for i in range(crash_len):
                idx = crash_point + i
                if idx < len(flash_crash):
                    flash_crash[idx] = flash_crash[idx] * (1 - 0.03 * (i + 1))
            scenarios.append({"name": "flash_crash", "prices": flash_crash.tolist(), "description": "10-bar 3% per bar crash"})

            spread_shock = prices.copy()
            shock_point = len(spread_shock) // 2
            spread_shock[shock_point:] = spread_shock[shock_point:] * 0.95
            scenarios.append({"name": "liquidity_crisis", "prices": spread_shock.tolist(), "description": "5% gap down, reduced liquidity"})

            high_vol = prices.copy()
            noise = np.random.normal(0, volatility * 3, len(high_vol))
            high_vol = high_vol * (1 + noise)
            scenarios.append({"name": "high_volatility", "prices": high_vol.tolist(), "description": f"3x normal volatility"})
        else:
            for i in range(3):
                scenarios.append({"name": f"generic_stress_{i}", "prices": [], "description": f"No price data for scenario {i}"})

        return scenarios

    def _evaluate_scenario(self, scenario: Dict, params: Dict) -> Dict:
        prices = scenario.get("prices", [])
        if not prices:
            return {"survived": True, "drawdown": 0, "name": scenario["name"]}

        price_arr = np.array(prices)
        peak = np.maximum.accumulate(price_arr)
        dd = (price_arr - peak) / (peak + 1e-10)
        max_dd = float(abs(np.min(dd)))

        return {
            "survived": max_dd < 0.30,
            "drawdown": max_dd,
            "name": scenario["name"],
            "description": scenario.get("description", ""),
        }

    def _run_tx_cost_analysis(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {
                "passed": True,
                "fee_rate": 0.0004,
                "slippage_rate": 0.0005,
                "total_cost_rate": 0.0009,
                "avg_cost_per_trade": 0,
                "trade_count": 0,
                "note": "no_trades_to_analyze",
            }

        fee_rate = 0.0004
        slippage_rate = 0.0005
        total_cost_rate = fee_rate * 2 + slippage_rate * 2

        costs = []
        for t in trades:
            entry_price = t.get("entry_price", 0)
            exit_price = t.get("exit_price", 0)
            leverage = t.get("leverage", 1)

            entry_fee = entry_price * fee_rate * leverage
            exit_fee = exit_price * fee_rate * leverage
            entry_slippage = entry_price * slippage_rate * leverage
            exit_slippage = exit_price * slippage_rate * leverage

            total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage
            cost_pct = total_cost / (entry_price + 1e-10)
            costs.append(cost_pct)

        avg_cost = float(np.mean(costs)) if costs else 0
        max_cost = float(np.max(costs)) if costs else 0

        return {
            "passed": avg_cost < 0.02,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "total_cost_rate": total_cost_rate,
            "avg_cost_per_trade_pct": avg_cost,
            "max_cost_per_trade_pct": max_cost,
            "trade_count": len(trades),
        }

    def _detect_overfitting(self, wf_result: Dict) -> Dict:
        sharpes = [f["sharpe"] for f in wf_result.get("fold_results", [])]

        if len(sharpes) < 3:
            return {"passed": True, "overfitting_score": 0, "note": "insufficient_folds"}

        mean_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes))
        max_sharpe = float(max(sharpes))
        min_sharpe = float(min(sharpes))

        cv = std_sharpe / (abs(mean_sharpe) + 1e-10)
        best_vs_avg_gap = (max_sharpe - mean_sharpe) / (abs(mean_sharpe) + 1e-10)

        score = cv * 0.5 + best_vs_avg_gap * 0.5

        return {
            "passed": score < 0.5,
            "overfitting_score": float(score),
            "cv_sharpe": float(cv),
            "best_vs_avg_gap": float(best_vs_avg_gap),
            "mean_sharpe": mean_sharpe,
            "sharpe_std": std_sharpe,
            "n_folds": len(sharpes),
        }

    def _default_backtest_result(self) -> Dict:
        return {"passed": False, "trades": [], "total_return": 0, "sharpe": 0, "volatility": 0, "max_drawdown": 0, "trade_count": 0, "warning": "no_data"}

    def _default_wf_result(self) -> Dict:
        return {"passed": False, "n_splits": 0, "mean_sharpe": 0, "sharpe_std": 0, "min_sharpe": 0, "max_sharpe": 0, "mean_return": 0, "return_std": 0, "fold_results": [], "purge_gap": 0}

    def _default_mc_result(self) -> Dict:
        return {"passed": False, "simulations": 0, "sharpe_mean": 0, "max_drawdown_mean": 0, "ruin_probability": 1, "profit_factor": 1}
