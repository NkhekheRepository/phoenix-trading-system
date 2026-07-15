#!/usr/bin/env python3
"""Monte Carlo stress test — permutes actual backtest trades 10,000x."""
import sys, os, json, zipfile, re, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format="%(message)s")

results_dir = "/freqtrade/user_data/backtest_results"
zips = sorted([f for f in os.listdir(results_dir) if f.endswith(".zip")])
if not zips:
    print("No backtest results found.")
    sys.exit(1)

zip_path = os.path.join(results_dir, zips[-1])
print(f"Loading from: {zips[-1]}")

with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        if name.endswith(".json") and "_config" not in name:
            raw = z.read(name)
            m = re.search(rb'\{.*', raw, re.DOTALL)
            data = json.loads(m.group())

trades_data = data["strategy"]["PhoenixScalper"]["trades"]
print(f"Trades loaded: {len(trades_data)}")

profits = [t["profit_ratio"] * 100 for t in trades_data]
durations = [t.get("trade_duration", 5) / 60.0 for t in trades_data]  # min → hours

results_obj = [
    TradeResult(
        profit_pct=p, win=p > 0, duration_hours=d,
        entry_price=0, exit_price=0, regime=0, kf_direction=0, kf_confidence=0.5,
    )
    for p, d in zip(profits, durations)
]

validator = MonteCarloValidator(n_simulations=10000)
print("Running 10,000 Monte Carlo simulations...")
mc_results = validator.simulate_trade_sequences(results_obj)
validation = validator.validate_targets(mc_results)
report = validator.generate_report(mc_results, validation)

report += f"\n\nActual:{'─'*50}"
report += f"\n  Trades:        {len(trades_data)}"
report += f"\n  Win rate:      {sum(1 for t in results_obj if t.win)/len(results_obj)*100:.1f}%"
report += f"\n  Total profit:  {sum(profits):+.2f}%"
report += f"\n  Avg profit:    {np.mean(profits):+.3f}%"
report += f"\n  Median profit: {np.median(profits):+.3f}%"
report += f"\n  Std profit:    {np.std(profits):+.3f}%"
report += f"\n  Best:          {max(profits):+.2f}%"
report += f"\n  Worst:         {min(profits):+.2f}%"
print(report)
