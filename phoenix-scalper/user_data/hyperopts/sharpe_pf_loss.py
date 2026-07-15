from datetime import datetime
from pandas import DataFrame
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_sharpe_daily import SharpeHyperOptLossDaily
from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeAndProfitFactorLoss(IHyperOptLoss):
    """
    Custom loss that targets Sharpe > 4.0 AND Profit Factor > 4.0.

    Loss = -Sharpe + max(0, 4.0 - PF) * 2 + max(0, 4.0 - Sharpe) * 0.5
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs,
    ) -> float:
        resample_freq = "1D"
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        annual_risk_free_rate = 0.0
        risk_free_rate = annual_risk_free_rate / days_in_year

        results.loc[:, "profit_ratio_after_slippage"] = (
            results["profit_ratio"] - slippage_per_trade_ratio
        )

        daily_profit = results.resample(resample_freq, on="close_date").agg(
            {"profit_ratio_after_slippage": sum}
        )
        daily_profit = daily_profit["profit_ratio_after_slippage"]

        if len(daily_profit) < 2:
            return 1000.0

        avg_daily = daily_profit.mean()
        std_daily = daily_profit.std()

        if std_daily == 0:
            return 1000.0

        daily_sharpe = (avg_daily - risk_free_rate) / std_daily
        sharpe = daily_sharpe * (days_in_year / (resample_freq == "1D")) ** 0.5

        total_profit = results.loc[results["profit_ratio"] > 0, "profit_ratio"].sum()
        total_loss = abs(
            results.loc[results["profit_ratio"] < 0, "profit_ratio"].sum()
        )

        pf = total_profit / total_loss if total_loss != 0 else 999.0

        loss = -sharpe
        if pf < 4.0:
            loss += (4.0 - pf) * 2.0
        if sharpe < 4.0:
            loss += (4.0 - sharpe) * 0.5

        return loss
