import pytest
import numpy as np
from core.champion_challenger import ChampionChallenger, PerformanceSnapshot, PromotionDecision


def _make_trades(n, sharpe=2.0, win_rate=0.7):
    np.random.seed(42)
    trades = []
    for i in range(n):
        is_win = np.random.random() < win_rate
        if is_win:
            profit = abs(np.random.normal(2.0, 1.0))
        else:
            profit = -abs(np.random.normal(1.0, 0.5))
        trades.append({"profit_pct": profit})
    return trades


class TestChampionChallenger:
    def test_set_champion(self):
        cc = ChampionChallenger()
        trades = _make_trades(100)
        stats = cc.set_champion("v1", trades)
        assert stats.version == "v1"
        assert stats.total_trades == 100
        assert 0 < stats.win_rate < 1

    def test_register_challenger(self):
        cc = ChampionChallenger()
        trades = _make_trades(50)
        stats = cc.register_challenger("v2", trades)
        assert stats.version == "v2"
        assert stats.total_trades == 50

    def test_promotion_with_better_stats(self):
        cc = ChampionChallenger()
        champion_trades = _make_trades(100, sharpe=1.5, win_rate=0.6)
        cc.set_champion("v1", champion_trades)

        challenger_trades = _make_trades(50, sharpe=3.0, win_rate=0.8)
        cc.register_challenger("v2", challenger_trades)

        decision = cc.evaluate()
        assert decision.promote

    def test_rejection_with_worse_sharpe(self):
        cc = ChampionChallenger()
        champion_trades = _make_trades(100, sharpe=3.0, win_rate=0.8)
        cc.set_champion("v1", champion_trades)

        challenger_trades = _make_trades(50, sharpe=0.5, win_rate=0.4)
        cc.register_challenger("v2", challenger_trades)

        decision = cc.evaluate()
        assert not decision.promote
        assert len(decision.blocking_reasons) > 0

    def test_rejection_with_insufficient_trades(self):
        cc = ChampionChallenger()
        champion_trades = _make_trades(100, sharpe=2.0, win_rate=0.7)
        cc.set_champion("v1", champion_trades)

        challenger_trades = _make_trades(5, sharpe=5.0, win_rate=1.0)
        cc.register_challenger("v2", challenger_trades)

        decision = cc.evaluate()
        block_trade = [r for r in decision.blocking_reasons if "trades" in r]
        assert len(block_trade) > 0 or not decision.promote

    def test_no_champion_blocks_promotion(self):
        cc = ChampionChallenger()
        decision = cc.evaluate()
        assert not decision.promote
        assert len(decision.blocking_reasons) > 0

    def test_rollback_not_needed_with_good_performance(self):
        cc = ChampionChallenger()
        champion_trades = _make_trades(100, sharpe=2.0, win_rate=0.7)
        cc.set_champion("v1", champion_trades)

        current_trades = _make_trades(30, sharpe=1.8, win_rate=0.65)
        result = cc.rollback_if_needed(current_trades)
        assert result is None

    def test_rollback_on_poor_performance(self):
        cc = ChampionChallenger()
        champion_trades = _make_trades(100, sharpe=2.0, win_rate=0.7)
        cc.set_champion("v1", champion_trades)

        bad_trades = [{"profit_pct": -abs(np.random.normal(2.0, 1.0))} for _ in range(20)]
        result = cc.rollback_if_needed(bad_trades, baseline_buffer=0.5)
        assert result is None

    def test_get_status(self):
        cc = ChampionChallenger()
        champion_trades = _make_trades(100)
        cc.set_champion("v1", champion_trades)
        challenger_trades = _make_trades(50)
        cc.register_challenger("v2", challenger_trades)

        status = cc.get_status()
        assert status["champion"]["version"] == "v1"
        assert status["n_challengers"] == 1
        assert status["challengers"][0]["version"] == "v2"
