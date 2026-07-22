#!/usr/bin/env python3
"""
Phoenix Scalper Observer — unified live dashboard for all trading bots.

Usage:
  python tools/observer.py                  # single-shot
  python tools/observer.py --watch 60       # refresh every 60s
  python tools/observer.py --json           # JSON output
  python tools/observer.py --scores         # show score distribution
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


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
    "V4": {
        "container": "phoenix-scalper-v4-bot",
        "db_path": "/freqtrade/user_data/tradesv3.sqlite",
        "name": "PhoenixScalperV4",
    },
}

ANSI = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}

NO_COLOR = {k: "" for k in ANSI}


class BotTelemetry:
    def __init__(self, key: str, cfg: dict):
        self.key = key
        self.cfg = cfg
        self.container = cfg["container"]

        self.alive = False
        self.state = "unknown"
        self.uptime = ""
        self.memory_mb = 0
        self.exchange_ok = True

        self.total_trades = 0
        self.open_trades = 0
        self.open_list: list[dict] = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.avg_profit = 0.0

        self.regime = "unknown"
        self.drift_mode = "normal"
        self.max_open = 5

        self.blocked_regime = 0
        self.blocked_ceiling = 0
        self.blocked_gate = 0
        self.score_counts: dict[int, int] = {}
        self.exit_reasons: dict[str, list] = {}
        self.entry_signals: dict[str, list] = {}
        self.exchange_errors = 0
        self.last_open_date = ""

    def gather(self):
        self._gather_liveness()
        self._gather_db_trades()
        self._gather_logs()

    def _docker_exec(self, cmd: list[str]) -> str:
        try:
            r = subprocess.run(
                ["docker", "exec", self.container] + cmd,
                capture_output=True, text=True, timeout=15,
            )
            return r.stdout
        except Exception:
            return ""

    def _docker_logs(self, tail: int = 2000) -> str:
        try:
            r = subprocess.run(
                ["docker", "logs", self.container, "--tail", str(tail)],
                capture_output=True, text=True, timeout=15,
            )
            return r.stderr + "\n" + r.stdout
        except Exception:
            return ""

    def _gather_liveness(self):
        out = self._docker_exec(["sh", "-c", "echo running"])
        if "running" in out:
            self.alive = True
            self.state = "RUNNING"
        out2 = self._docker_exec(["cat", "/proc/self/status"])
        for line in out2.splitlines():
            if line.startswith("VmRSS:"):
                try:
                    self.memory_mb = int(line.split()[1]) // 1024
                except Exception:
                    pass
        out3 = self._docker_logs(10)
        for line in out3.splitlines():
            m = re.search(r'Bot heartbeat.*state=\'(\w+)\'', line)
            if m:
                self.state = m.group(1)
            m2 = re.search(r'uptime.*\](.*?)(?=\s*\()', line)
            if m2:
                self.uptime = m2.group(1)

    def _gather_db_trades(self):
        import sqlite3 as sqlite3_mod
        tmp = Path(tempfile.gettempdir()) / f"obs_{self.container}.sqlite"
        try:
            subprocess.run(
                ["docker", "cp", f"{self.container}:{self.cfg['db_path']}", str(tmp)],
                capture_output=True, text=True, timeout=15, check=True,
            )
        except Exception:
            return
        if not tmp.exists() or tmp.stat().st_size == 0:
            return
        try:
            conn = sqlite3_mod.connect(str(tmp))
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM trades")
            self.total_trades = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM trades WHERE is_open=1")
            self.open_trades = int(cur.fetchone()[0])
            cur.execute("SELECT COALESCE(SUM(close_profit>0),0) FROM trades")
            self.wins = int(cur.fetchone()[0])
            cur.execute("SELECT COALESCE(SUM(close_profit<0),0) FROM trades")
            self.losses = int(cur.fetchone()[0])
            cur.execute("SELECT COALESCE(SUM(close_profit_abs),0) FROM trades")
            self.total_pnl = float(cur.fetchone()[0])
            cur.execute("SELECT COALESCE(AVG(close_profit),0) FROM trades")
            self.avg_profit = float(cur.fetchone()[0])
            try:
                cur.execute("SELECT enter_tag, COUNT(*) FROM trades WHERE enter_tag IS NOT NULL GROUP BY enter_tag ORDER BY COUNT(*) DESC")
                for tag, cnt in cur.fetchall():
                    sm = re.search(r'\[(\d+)\]', tag or "")
                    if sm:
                        sc = int(sm.group(1))
                        self.score_counts[sc] = self.score_counts.get(sc, 0) + cnt
                    sig = re.sub(r'\s*\[\d+\]', '', tag or "")
                    if sig not in self.entry_signals:
                        self.entry_signals[sig] = []
                    self.entry_signals[sig].append({"count": cnt})
            except Exception:
                pass
            try:
                cur.execute("SELECT exit_reason, AVG(close_profit), COUNT(*) FROM trades WHERE exit_reason IS NOT NULL AND exit_reason != '' GROUP BY exit_reason")
                for reason, avg_p, cnt in cur.fetchall():
                    self.exit_reasons[reason] = {"count": int(cnt), "avg_p": round(float(avg_p or 0), 4)}
            except Exception:
                pass
            try:
                cur.execute("SELECT MAX(open_date) FROM trades")
                row = cur.fetchone()
                if row and row[0]:
                    self.last_open_date = str(row[0])
            except Exception:
                pass
            conn.close()
        except Exception:
            pass
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass

    def _gather_logs(self):
        logs = self._docker_logs(3000)
        for line in logs.splitlines():
            if "BLOCK" in line and "regime" in line.lower():
                self.blocked_regime += 1
            if "CEILING" in line and "score" in line.lower():
                self.blocked_ceiling += 1
            if "SCORE GATE" in line and "reject" in line.lower():
                self.blocked_gate += 1
            if "ExchangeNotAvailable" in line or "RequestTimeout" in line:
                self.exchange_errors += 1
            mr = re.search(r'Regime\s+(\S+)\s+drift=(\w+)', line)
            if mr:
                self.regime = mr.group(1)
                self.drift_mode = mr.group(2)
            mm = re.search(r'max_open_trades\s*->\s*(\d+)', line)
            if mm:
                self.max_open = int(mm.group(1))


def fmt_pnl(val: float) -> str:
    if val > 0:
        return f"+${val:.2f}"
    elif val < 0:
        return f"-${abs(val):.2f}"
    return "$0.00"


def fmt_pct(val: float) -> str:
    if val > 0:
        return f"+{val*100:.2f}%"
    elif val < 0:
        return f"{val*100:.2f}%"
    return "0.00%"


def build_table(bots: dict[str, BotTelemetry], use_color: bool) -> str:
    C = ANSI if use_color else NO_COLOR
    lines = []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("")
    lines.append(f"{C['bold']}╔═══ Phoenix Scalper Observer ═══╗{C['reset']}")
    lines.append(f"{C['dim']}  {ts}{C['reset']}")
    lines.append("")

    header = (
        f"{C['bold']}{'Bot':<4} {'State':<9} {'Regime':<14} {'Trades':>6} {'Open':>4} "
        f"{'W/L':>9} {'P&L':>10} {'Avg%':>7} {'Mem':>5} {'Exch':>5}{C['reset']}"
    )
    lines.append(header)
    lines.append("─" * (len(header) - 8) if use_color else "─" * len(header))

    for key in ["V2", "V3", "V4"]:
        b = bots[key]
        status_color = C["green"] if b.state == "RUNNING" else C["red"]
        pnl_color = C["green"] if b.total_pnl >= 0 else C["red"]
        avg_color = C["green"] if b.avg_profit >= 0 else C["red"]
        exch = f"{C['green']}OK{C['reset']}" if b.exchange_errors == 0 else f"{C['red']}{b.exchange_errors}e{C['reset']}"
        mem_str = f"{b.memory_mb}M" if b.memory_mb else "?"
        lines.append(
            f"{C['cyan']}{key:<4}{C['reset']} "
            f"{status_color}{b.state:<9}{C['reset']} "
            f"{b.regime:<14} "
            f"{b.total_trades:>6} "
            f"{b.open_trades:>4} "
            f"{b.wins}/{b.losses:<+5} "
            f"{pnl_color}{fmt_pnl(b.total_pnl):>10}{C['reset']} "
            f"{avg_color}{fmt_pct(b.avg_profit):>7}{C['reset']} "
            f"{mem_str:>5} "
            f"{exch}"
        )
        if b.max_open != 5:
            lines.append(f"  └─ max_open={b.max_open}  drift={b.drift_mode}")
        if b.blocked_regime or b.blocked_ceiling or b.blocked_gate:
            blocks = []
            if b.blocked_regime:
                blocks.append(f"regime={b.blocked_regime}")
            if b.blocked_ceiling:
                blocks.append(f"ceiling={b.blocked_ceiling}")
            if b.blocked_gate:
                blocks.append(f"gate={b.blocked_gate}")
            lines.append(f"  └─ blocked: {', '.join(blocks)}")
        if b.score_counts:
            scores_sorted = sorted(b.score_counts.items())
            score_str = ", ".join(f"[{s}]×{c}" for s, c in scores_sorted)
            lines.append(f"  └─ scores: {score_str}")

    lines.append("")

    return "\n".join(lines)


def build_json_output(bots: dict[str, BotTelemetry]) -> str:
    out = {"timestamp": datetime.now(timezone.utc).isoformat(), "bots": {}}
    for key, b in bots.items():
        out["bots"][key] = {
            "alive": b.alive,
            "state": b.state,
            "regime": b.regime,
            "drift_mode": b.drift_mode,
            "max_open_trades": b.max_open,
            "total_trades": b.total_trades,
            "open_trades": b.open_trades,
            "wins": b.wins,
            "losses": b.losses,
            "total_pnl": round(b.total_pnl, 2),
            "avg_profit": round(b.avg_profit, 4),
            "memory_mb": b.memory_mb,
            "exchange_errors": b.exchange_errors,
            "blocked_regime": b.blocked_regime,
            "blocked_ceiling": b.blocked_ceiling,
            "blocked_gate": b.blocked_gate,
            "score_counts": b.score_counts,
            "exit_reasons": b.exit_reasons,
            "last_open_date": b.last_open_date,
        }
    return json.dumps(out, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phoenix Scalper Observer")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    while True:
        bots = {}
        for key, cfg in BOTS.items():
            bt = BotTelemetry(key, cfg)
            bt.gather()
            bots[key] = bt

        if args.json:
            print(build_json_output(bots))
        else:
            print(build_table(bots, use_color))

        sys.stdout.flush()
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
