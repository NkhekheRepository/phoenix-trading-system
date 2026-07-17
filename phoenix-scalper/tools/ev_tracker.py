#!/usr/bin/env python3
"""
Live EV tracker — polls both scalper containers and reports gated EV metrics.

Usage:
  python tools/ev_tracker.py                     # single-shot
  python tools/ev_tracker.py --watch 300          # loop every 300s
  python tools/ev_tracker.py --watch 300 --json   # append to ev_log.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev_core import compute_ev, format_ev_table, EVResult

BOTS = {
    "V2": {
        "container": "phoenix-scalper-bot",
        "db_path": "/freqtrade/user_data/tradesv3.sqlite",
        "name": "PhoenixScalperV2",
    },
    "V3": {
        "container": "b6eb803b036c_phoenix-scalper-v3-bot",
        "db_path": "/freqtrade/user_data/tradesv3.sqlite",
        "name": "PhoenixScalperV3",
    },
}


def fetch_db(container: str, remote_path: str) -> Path | None:
    dest = Path(tempfile.gettempdir()) / f"ev_tracker_{container}.sqlite"
    try:
        subprocess.run(
            ["docker", "cp", f"{container}:{remote_path}", str(dest)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return dest if dest.exists() else None
    except subprocess.CalledProcessError:
        return None


def count_blocked(container: str) -> int:
    try:
        out = subprocess.run(
            ["docker", "logs", container, "--tail", "2000"],
            capture_output=True, text=True, timeout=15,
        )
        lines = (out.stdout + out.stderr).splitlines()
        return len([l for l in lines if "not allowed" in l.lower()])
    except Exception:
        return 0


def get_regime_from_logs(container: str) -> tuple[str, str]:
    try:
        out = subprocess.run(
            ["docker", "logs", container, "--tail", "200"],
            capture_output=True, text=True, timeout=15,
        )
        lines = (out.stdout + out.stderr).splitlines()
        current = "unknown"
        for line in reversed(lines):
            m = re.search(r'Regime\s+(\w+)\s+drift=(\w+)', line)
            if m:
                current = m.group(1)
                mode = m.group(2)
                return current, mode
        return "unknown", "unknown"
    except Exception:
        return "unknown", "unknown"


def compute_bot_ev(bot_key: str, cfg: dict) -> EVResult:
    db_path = fetch_db(cfg["container"], cfg["db_path"])
    regime, mode = get_regime_from_logs(cfg["container"])
    blocked = count_blocked(cfg["container"])

    result = EVResult(bot_name=cfg["name"], current_regime=regime, regime_mode=mode, blocked_count=blocked)

    if not db_path:
        return result

    try:
        from core.ev_core import compute_ev as _ce
        result = _ce(
            db_path=db_path,
            bot_name=cfg["name"],
            current_regime=regime,
            regime_mode=mode,
        )
        result.blocked_count = blocked
    except Exception as e:
        result.blocked_count = blocked

    try:
        os.unlink(db_path)
    except OSError:
        pass

    return result


def json_serialize(result: EVResult) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "bot": result.bot_name,
        "regime": result.current_regime,
        "regime_mode": result.regime_mode,
        "total_trades": result.total_trades,
        "open_trades": result.open_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": round(result.win_rate, 4),
        "ev_per_trade": round(result.ev_per_trade, 4),
        "total_pnl_abs": round(result.total_pnl_abs, 4),
        "profit_factor": round(result.profit_factor, 4),
        "blocked_count": result.blocked_count,
        "by_score_band": result.by_score_band,
        "by_exit_reason": result.by_exit_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Live EV tracker for phoenix scalper bots")
    parser.add_argument("--watch", type=int, default=0, help="Poll interval in seconds (0 = single-shot)")
    parser.add_argument("--json", action="store_true", help="Append as JSON to ev_log.json")
    parser.add_argument("--log", type=str, default="ev_log.json", help="JSON log file path")
    args = parser.parse_args()

    log_path = Path(args.log) if args.json else None

    while True:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"\n{'='*50}")
        print(f"  EV Snapshot — {ts} UTC")
        print(f"{'='*50}")

        entries = []
        for key, cfg in BOTS.items():
            result = compute_bot_ev(key, cfg)
            print(format_ev_table(result))
            if log_path:
                entries.append(json_serialize(result))

        if log_path and entries:
            existing = []
            if log_path.exists():
                try:
                    existing = json.loads(log_path.read_text())
                except (json.JSONDecodeError, Exception):
                    existing = []
            existing.extend(entries)
            log_path.write_text(json.dumps(existing, indent=2))

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
