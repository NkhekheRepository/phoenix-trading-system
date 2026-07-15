import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    trade_id: str
    pair: str
    side: str
    leverage: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit_pct: float = 0.0
    exit_reason: str = ""
    entry_tag: str = ""

    market_state: Dict[str, float] = field(default_factory=dict)
    regime: str = ""
    regime_confidence: float = 0.0
    risk_level: str = ""

    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    model_predictions: Dict[str, float] = field(default_factory=dict)

    entry_signals: Dict[str, bool] = field(default_factory=dict)
    exit_signals: Dict[str, Any] = field(default_factory=dict)
    failure_factors: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        if self.exit_time:
            d["exit_time"] = self.exit_time.isoformat()
        return d


class TradeIntelligence:
    def __init__(self, storage_path: str = "data/trade_intel", on_notify=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._trades: Dict[str, TradeRecord] = {}
        self.on_notify = on_notify

    def start_trade(self, pair: str, side: str, leverage: float, entry_price: float,
                    entry_tag: str, market_state: Dict, regime: str,
                    regime_confidence: float, risk_level: str) -> str:
        trade_id = f"{pair}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hash(pair + str(datetime.now().timestamp())) % 10000:04d}"

        record = TradeRecord(
            trade_id=trade_id,
            pair=pair,
            side=side,
            leverage=leverage,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            entry_tag=entry_tag,
            market_state=market_state,
            regime=regime,
            regime_confidence=regime_confidence,
            risk_level=risk_level,
        )
        self._trades[trade_id] = record
        return trade_id

    def close_trade(self, trade_id: str, exit_price: float, profit_pct: float,
                    exit_reason: str, failure_factors: List[str] = None,
                    success_factors: List[str] = None):
        record = self._trades.get(trade_id)
        if not record:
            logger.warning(f"Trade {trade_id} not found")
            return

        record.exit_time = datetime.now(timezone.utc)
        record.exit_price = exit_price
        record.profit_pct = profit_pct
        record.exit_reason = exit_reason
        record.failure_factors = failure_factors or []
        record.success_factors = success_factors or []

        self._save_trade(record)

        if self.on_notify:
            duration = ""
            if record.exit_time and record.entry_time:
                delta = record.exit_time - record.entry_time
                hours = delta.total_seconds() // 3600
                minutes = (delta.total_seconds() % 3600) // 60
                duration = f"{int(hours)}h {int(minutes)}m"
            self.on_notify(
                "trade_attribution",
                pair=record.pair,
                side=record.side,
                profit_pct=record.profit_pct,
                exit_reason=record.exit_reason,
                entry_tag=record.entry_tag,
                regime=record.regime,
                failure_factors=record.failure_factors,
                success_factors=record.success_factors,
                duration=duration,
            )

    def _save_trade(self, record: TradeRecord):
        date_key = record.entry_time.strftime("%Y/%m")
        path = self.storage_path / date_key
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{record.trade_id}.json"
        with open(filepath, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def analyze_losing_patterns(self, n_trades: int = 100) -> Dict:
        recent = self._get_recent_trades(n_trades)
        losers = [t for t in recent if t.profit_pct < 0]

        if not losers:
            return {"pattern": "no_losers", "count": 0}

        patterns = {}
        for trade in losers:
            for factor in trade.failure_factors:
                patterns[factor] = patterns.get(factor, 0) + 1

        regimes = {}
        for trade in losers:
            r = trade.regime
            regimes[r] = regimes.get(r, 0) + 1

        entry_tags = {}
        for trade in losers:
            tag = trade.entry_tag
            entry_tags[tag] = entry_tags.get(tag, 0) + 1

        return {
            "total_losers": len(losers),
            "loss_rate": len(losers) / len(recent) if recent else 0,
            "failure_factors": dict(sorted(patterns.items(), key=lambda x: -x[1])),
            "worst_regimes": dict(sorted(regimes.items(), key=lambda x: -x[1])),
            "worst_entry_tags": dict(sorted(entry_tags.items(), key=lambda x: -x[1])),
            "avg_loss_pct": abs(sum(t.profit_pct for t in losers)) / len(losers) if losers else 0,
        }

    def analyze_winning_patterns(self, n_trades: int = 100) -> Dict:
        recent = self._get_recent_trades(n_trades)
        winners = [t for t in recent if t.profit_pct > 0]

        if not winners:
            return {"pattern": "no_winners", "count": 0}

        regimes = {}
        for trade in winners:
            r = trade.regime
            regimes[r] = regimes.get(r, 0) + 1

        entry_tags = {}
        for trade in winners:
            tag = trade.entry_tag
            entry_tags[tag] = entry_tags.get(tag, 0) + 1

        return {
            "total_winners": len(winners),
            "best_regimes": dict(sorted(regimes.items(), key=lambda x: -x[1])),
            "best_entry_tags": dict(sorted(entry_tags.items(), key=lambda x: -x[1])),
            "avg_win_pct": sum(t.profit_pct for t in winners) / len(winners) if winners else 0,
        }

    def _get_recent_trades(self, n: int) -> List[TradeRecord]:
        all_files = sorted(self.storage_path.rglob("*.json"), reverse=True)
        trades = []
        for f in all_files[:n]:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    record = TradeRecord(**data)
                    record.entry_time = datetime.fromisoformat(data["entry_time"])
                    if data.get("exit_time"):
                        record.exit_time = datetime.fromisoformat(data["exit_time"])
                    trades.append(record)
            except Exception as e:
                logger.warning(f"Error loading trade {f}: {e}")
        return trades

    def get_trade_count(self) -> int:
        return len(list(self.storage_path.rglob("*.json")))
