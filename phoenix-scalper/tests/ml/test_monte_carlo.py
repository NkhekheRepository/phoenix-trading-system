import pytest
import numpy as np
from ml.monte_carlo import MonteCarloValidator, TradeResult, create_sample_trade_results


class TestMonteCarlo:
    def setup_method(self):
        self.validator = MonteCarloValidator(n_simulations=100)

    def test_simulate_trade_sequences(self):
        trades = create_sample_trade_results(100)
        results = self.validator.simulate_trade_sequences(trades)
        assert "sharpe" in results
        assert "max_dd" in results
        assert "win_rate" in results
        assert len(results["sharpe"]) == 100

    def test_validate_targets(self):
        trades = create_sample_trade_results(100)
        results = self.validator.simulate_trade_sequences(trades)
        validation = self.validator.validate_targets(results)
        assert "overall_success" in validation
        assert "meets_sharpe_target" in validation
        assert "meets_dd_target" in validation

    def test_generate_report(self):
        trades = create_sample_trade_results(100)
        results = self.validator.simulate_trade_sequences(trades)
        validation = self.validator.validate_targets(results)
        report = self.validator.generate_report(results, validation)
        assert "MONTE CARLO" in report
        assert "OVERALL" in report
