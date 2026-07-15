import pytest
import numpy as np
from core.validation_pipeline import ValidationPipeline


class TestValidationPipeline:
    def test_default_report_when_no_data(self):
        pipeline = ValidationPipeline()
        report = pipeline.validate_strategy("test_v1", {}, None)
        assert not report.passed
        assert report.gates_total == 6

    def test_backtest_with_data(self):
        pipeline = ValidationPipeline()
        data = {"prices": np.cumprod(1 + np.random.normal(0.001, 0.02, 500)).tolist()}
        result = pipeline._run_backtest({}, data)
        assert "sharpe" in result
        assert "trade_count" in result
        assert result["trade_count"] > 0

    def test_walk_forward_splits_correctly(self):
        pipeline = ValidationPipeline()
        data = {"prices": np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)).tolist()}
        result = pipeline._run_walk_forward({}, data)
        assert result["n_splits"] >= 3
        assert "mean_sharpe" in result
        assert "sharpe_std" in result

    def test_walk_forward_fails_with_insufficient_data(self):
        pipeline = ValidationPipeline()
        data = {"prices": [100] * 50}
        result = pipeline._run_walk_forward({}, data)
        assert result["n_splits"] == 0
        assert not result["passed"]

    def test_stress_test_generates_scenarios(self):
        pipeline = ValidationPipeline()
        data = {"prices": np.cumprod(1 + np.random.normal(0.001, 0.02, 300)).tolist()}
        result = pipeline._run_stress_test({}, data)
        assert result["scenarios_tested"] >= 3
        assert "scenarios_survived" in result

    def test_tx_cost_analysis_with_trades(self):
        pipeline = ValidationPipeline()
        trades = [
            {"entry_price": 100, "exit_price": 105, "leverage": 10, "profit_pct": 5},
            {"entry_price": 50, "exit_price": 48, "leverage": 5, "profit_pct": -2},
        ]
        result = pipeline._run_tx_cost_analysis(trades)
        assert result["trade_count"] == 2
        assert result["avg_cost_per_trade_pct"] > 0
        assert result["fee_rate"] == 0.0004

    def test_overfitting_detection_with_multiple_folds(self):
        pipeline = ValidationPipeline()
        wf_result = {
            "fold_results": [
                {"sharpe": 2.0, "return": 0.1, "fold": 0},
                {"sharpe": 1.8, "return": 0.09, "fold": 1},
                {"sharpe": 0.2, "return": 0.01, "fold": 2},
                {"sharpe": 1.9, "return": 0.095, "fold": 3},
                {"sharpe": 0.1, "return": 0.005, "fold": 4},
            ]
        }
        result = pipeline._detect_overfitting(wf_result)
        assert "overfitting_score" in result
        assert "mean_sharpe" in result
        assert result["n_folds"] == 5

    def test_overfitting_low_variance_passes(self):
        pipeline = ValidationPipeline()
        wf_result = {
            "fold_results": [
                {"sharpe": 2.0, "return": 0.1, "fold": 0},
                {"sharpe": 1.9, "return": 0.095, "fold": 1},
                {"sharpe": 2.1, "return": 0.105, "fold": 2},
            ]
        }
        result = pipeline._detect_overfitting(wf_result)
        assert result["passed"]
