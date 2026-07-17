#!/usr/bin/env python3
"""
Gated backtest simulator — replays historical backup DBs through the new
gating rules (score ceiling, regime allowlist, 1h max hold) and compares
old (unfiltered) vs gated performance.

Usage:
  python tools/gated_backtest.py
  python tools/gated_backtest.py --v2-only
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev_core import (
    compute_ev,
    format_ev_table,
    _parse_score,
    _score_band,
    SCORE_BANDS,
    REGIME_ALLOWLIST,
    MAX_HOLD_HOURS,
    EVResult,
)

BACKUPS = {
    "V2": Path("/home/nkhekhe/phoenix-db-backups/v2_trades_20260717_0356.sqlite"),
    "V3": Path("/home/nkhekhe/phoenix-db-backups/v3_trades_20260717_0356.sqlite"),
}

CONTAINER_NAMES = {
    "V2": "phoenix-scalper-bot",
    "V3": "b6eb803b036c_phoenix-scalper-v3-bot",
}


def rebuild_regime_timeline(container_name: str) -> list[tuple[datetime, str]]:
    """Fetch container logs via docker and return sorted [(timestamp, regime), ...].
    
    Parses lines like:
        2026-07-17 04:42:16,404 - PhoenixScalperV2 - INFO - Regime weak_bear drift=normal: ...
    """
    import subprocess
    timeline = []
    pattern = re.compile(
        r'Regime\s+(\w+)\s+drift='
    )

    try:
        out = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True, text=True, timeout=30,
        )
        log_text = out.stdout + out.stderr
    except Exception as e:
        print(f"  [warn] could not fetch logs for {container_name}: {e}")
        return []

    for line in log_text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        ts_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
        if not ts_match:
            continue
        try:
            ts = datetime.fromisoformat(ts_match.group(1))
        except ValueError:
            continue
        timeline.append((ts, m.group(1)))

    timeline.sort(key=lambda x: x[0])
    return timeline


def get_regime_at(timeline: list[tuple[datetime, str]], dt: datetime) -> str:
    """Return the regime active at timestamp dt based on the timeline."""
    regime = "unknown"
    for ts, r in timeline:
        if ts <= dt:
            regime = r
        else:
            break
    return regime


def run_backtest_v2(backup_path: Path):
    """V2 backtest: score ceiling + regime allowlist + 1h cap."""
    print(f"\n{'='*60}")
    print("  V2 GATED BACKTEST")
    print(f"{'='*60}")

    timeline = rebuild_regime_timeline(CONTAINER_NAMES["V2"])
    print(f"  Regime events parsed: {len(timeline)}")

    con = sqlite3.connect(str(backup_path))
    rows = con.execute(
        "SELECT enter_tag, close_profit, close_profit_abs, "
        "exit_reason, is_open, open_date, close_date FROM trades"
    ).fetchall()
    con.close()

    total_old = len(rows)
    total_pnl_old = sum(r[2] or 0 for r in rows)
    total_abs_old = sum(r[2] or 0 for r in rows)
    wins_old = sum(1 for r in rows if r[1] and r[1] > 0)

    gated = []
    capped = 0
    blocked_regime = 0
    blocked_score = 0

    for et, cp, cpa, er, is_open, opendt, closedt in rows:
        cpa = cpa or 0.0
        cp = cp or 0.0

        if is_open:
            continue

        open_ts = datetime.fromisoformat(opendt) if isinstance(opendt, str) else opendt

        # 1. Score ceiling: only 55-59
        score = _parse_score(et)
        band = _score_band(score)
        if band != "55-59":
            blocked_score += 1
            continue

        # 2. Regime allowlist
        regime = get_regime_at(timeline, open_ts)
        if regime not in REGIME_ALLOWLIST:
            blocked_regime += 1
            continue

        # 3. 1h max hold (flag if was timed out)
        if er == "max_hold_3h":
            capped += 1

        gated.append((cp, cpa, er or "unknown", et))

    n_gated = len(gated)
    pnl_gated = sum(r[1] for r in gated)
    wins_gated = sum(1 for r in gated if r[0] > 0)

    wr_old = f"{wins_old/total_old*100:.1f}%" if total_old else "N/A"
    wr_gated = f"{wins_gated/n_gated*100:.1f}%" if n_gated else "N/A"

    print(f"\n  Old (unfiltered): {total_old} trades, ${total_abs_old:+.2f} total, WR {wr_old}")
    print(f"  Gated:            {n_gated} trades, ${pnl_gated:+.2f} total, WR {wr_gated}")
    print(f"  Blocked by score: {blocked_score}")
    print(f"  Blocked by regime:{blocked_regime}")
    print(f"  1h-capped saves:  {capped} trades (previously blown by max_hold_3h)")
    notional_trades_blocked = total_old - n_gated
    print(f"  Trades prevented: {notional_trades_blocked} ({blocked_score}+{blocked_regime})")

    if n_gated:
        old_ev = total_abs_old / total_old if total_old else 0
        print(f"\n  Gated EV: ${pnl_gated/n_gated:+.2f}/trade (was ${old_ev:+.2f})")
    print(f"{'='*60}")


def run_backtest_v3(backup_path: Path):
    """V3 backtest: regime allowlist only (no score filter)."""
    print(f"\n{'='*60}")
    print("  V3 GATED BACKTEST (regime allowlist only)")
    print(f"{'='*60}")

    timeline = rebuild_regime_timeline(CONTAINER_NAMES["V3"])
    print(f"  Regime events parsed: {len(timeline)}")

    con = sqlite3.connect(str(backup_path))
    rows = con.execute(
        "SELECT enter_tag, close_profit, close_profit_abs, "
        "exit_reason, is_open, open_date, close_date FROM trades"
    ).fetchall()
    con.close()

    total_old = len(rows)
    total_abs_old = sum(r[2] or 0 for r in rows)
    wins_old = sum(1 for r in rows if r[1] and r[1] > 0)

    gated = []
    regime_breakdown = defaultdict(list)

    for et, cp, cpa, er, is_open, opendt, closedt in rows:
        cpa = cpa or 0.0
        cp = cp or 0.0

        if is_open:
            continue

        open_ts = datetime.fromisoformat(opendt) if isinstance(opendt, str) else opendt
        regime = get_regime_at(timeline, open_ts)
        regime_breakdown[regime].append(cpa)

        if regime in REGIME_ALLOWLIST:
            gated.append((cp, cpa, er or "unknown", et))

    n_gated = len(gated)
    pnl_gated = sum(r[1] for r in gated)
    wins_gated = sum(1 for r in gated if r[0] > 0)

    wr_old = f"{wins_old/total_old*100:.1f}%" if total_old else "N/A"

    print(f"\n  Old (unfiltered): {total_old} trades, ${total_abs_old:+.2f} total, WR {wr_old}")

    # Regime breakdown
    print("\n  Historical EV by regime:")
    for regime in sorted(regime_breakdown.keys()):
        vals = regime_breakdown[regime]
        ev = sum(vals) / len(vals)
        print(f"    {regime:15s} n={len(vals):3d} ev=${ev:+.2f} sum=${sum(vals):+.2f}")

    if n_gated:
        old_ev = total_abs_old / total_old if total_old else 0
        print(f"\n  Gated (allowlist only):")
        print(f"    {n_gated} trades, ${pnl_gated:.2f} total, WR {wins_gated/n_gated*100:.1f}%")
        print(f"    EV: ${pnl_gated/n_gated:+.2f}/trade (was ${old_ev:+.2f})")
    else:
        print("\n  Gated: 0 trades survived the allowlist filter.")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Gated backtest simulator")
    parser.add_argument("--v2-only", action="store_true", help="V2 only")
    parser.add_argument("--v3-only", action="store_true", help="V3 only")
    args = parser.parse_args()

    if not args.v3_only:
        if BACKUPS["V2"].exists():
            run_backtest_v2(BACKUPS["V2"])
        else:
            print(f"V2 backup not found: {BACKUPS['V2']}")

    if not args.v2_only:
        if BACKUPS["V3"].exists():
            run_backtest_v3(BACKUPS["V3"])
        else:
            print(f"V3 backup not found: {BACKUPS['V3']}")


if __name__ == "__main__":
    main()
