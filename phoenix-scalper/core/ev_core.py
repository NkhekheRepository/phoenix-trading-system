import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCORE_BANDS = [
    ("lt55", (0, 55)),
    ("55-59", (55, 60)),
    ("60-64", (60, 65)),
    ("65plus", (65, 999)),
]

REGIME_ALLOWLIST = ("low_volatility", "weak_bull")
MAX_HOLD_HOURS = 1.0


@dataclass
class EVResult:
    bot_name: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_pnl_abs: float = 0.0
    avg_trade_pnl: float = 0.0
    ev_per_trade: float = 0.0
    win_rate: float = 0.0
    by_score_band: dict[str, dict] = field(default_factory=dict)
    by_exit_reason: dict[str, dict] = field(default_factory=dict)
    by_regime: dict[str, dict] = field(default_factory=dict)
    blocked_count: int = 0
    open_trades: int = 0
    current_regime: str = "unknown"
    regime_mode: str = "unknown"
    profit_factor: float = 0.0
    score_threshold: int = 55
    score_high_threshold: int = 60


def _parse_score(enter_tag: str | None) -> int | None:
    if not enter_tag:
        return None
    m = re.search(r'\[(\d+)\]', enter_tag)
    if m:
        return int(m.group(1))
    return None


def _score_band(score: int | None) -> str:
    if score is None:
        return "noscore"
    for name, (lo, hi) in SCORE_BANDS:
        if lo <= score < hi:
            return name
    return "noscore"


def compute_ev(db_path: str | Path,
               bot_name: str = "",
               current_regime: str = "unknown",
               regime_mode: str = "unknown",
               score_threshold: int = 55,
               score_high_threshold: int = 60) -> EVResult:
    result = EVResult(
        bot_name=bot_name,
        current_regime=current_regime,
        regime_mode=regime_mode,
        score_threshold=score_threshold,
        score_high_threshold=score_high_threshold,
    )
    try:
        con = sqlite3.connect(str(db_path))
    except sqlite3.Error as e:
        logger.error(f"ev_core: cannot open {db_path}: {e}")
        return result

    try:
        rows = con.execute(
            "SELECT enter_tag, close_profit, close_profit_abs, "
            "exit_reason, is_open FROM trades"
        ).fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"ev_core: query failed: {e}")
        con.close()
        return result

    total = 0.0
    total_abs = 0.0
    wins = 0
    losses = 0
    opens = 0
    by_score: dict[str, list[float]] = {}
    by_exit: dict[str, list[float]] = {}
    profit_abs_sum = 0.0
    loss_abs_sum = 0.0

    for enter_tag, close_profit, close_profit_abs, exit_reason, is_open in rows:
        score = _parse_score(enter_tag)
        band = _score_band(score)

        cp = close_profit or 0.0
        cpa = close_profit_abs or 0.0

        if is_open:
            opens += 1
            continue

        total += cp
        total_abs += cpa
        if cp > 0:
            wins += 1
            profit_abs_sum += cpa
        elif cp < 0:
            losses += 1
            loss_abs_sum += abs(cpa)

        by_score.setdefault(band, []).append(cpa)
        by_exit.setdefault(exit_reason or "unknown", []).append(cpa)

    con.close()

    closed = wins + losses
    result.total_trades = closed
    result.open_trades = opens
    result.wins = wins
    result.losses = losses
    result.total_pnl = total
    result.total_pnl_abs = total_abs
    result.ev_per_trade = total_abs / closed if closed else 0.0
    result.avg_trade_pnl = total / closed if closed else 0.0
    result.win_rate = wins / closed if closed else 0.0
    result.profit_factor = profit_abs_sum / loss_abs_sum if loss_abs_sum > 0 else float('inf') if profit_abs_sum > 0 else 0.0

    for band, vals in sorted(by_score.items()):
        result.by_score_band[band] = {
            "n": len(vals),
            "sum": round(sum(vals), 4),
            "avg": round(sum(vals) / len(vals), 4),
        }
    for er, vals in sorted(by_exit.items()):
        result.by_exit_reason[er] = {
            "n": len(vals),
            "sum": round(sum(vals), 4),
            "avg": round(sum(vals) / len(vals), 4),
        }

    return result


def format_ev_table(result: EVResult, show_bands: bool = True) -> str:
    lines = []
    sep = "━" * 34

    lines.append(f"📊 *{result.bot_name} EV Report*")
    lines.append(f"Regime: {result.current_regime} ({result.regime_mode})")
    lines.append(f"Thresholds: score≥{result.score_threshold} · high≥{result.score_high_threshold}")
    lines.append(sep)

    lines.append(
        f"Trades: {result.total_trades} closed · {result.open_trades} open · "
        f"{result.blocked_count} blocked"
    )
    if result.total_trades:
        lines.append(
            f"Win: {result.wins}/{result.total_trades} "
            f"({result.win_rate * 100:.0f}%) · "
            f"PF: {result.profit_factor:.2f}"
        )
        lines.append(
            f"EV: *${result.ev_per_trade:+.2f}/trade* · "
            f"Cum: ${result.total_pnl_abs:+.2f}"
        )
    lines.append(sep)

    if show_bands and result.by_score_band:
        lines.append("By Score Band:")
        for band, d in result.by_score_band.items():
            lines.append(
                f"  {band:8s} n={d['n']:3d} "
                f"avg=${d['avg']:+.2f} sum=${d['sum']:+.2f}"
            )

    if result.by_exit_reason:
        lines.append("By Exit:")
        for er, d in result.by_exit_reason.items():
            lines.append(
                f"  {er or 'open':20s} n={d['n']:3d} "
                f"avg=${d['avg']:+.2f} sum=${d['sum']:+.2f}"
            )

    lines.append(sep)
    return "\n".join(lines)
