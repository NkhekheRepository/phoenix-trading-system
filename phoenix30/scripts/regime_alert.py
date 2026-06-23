import json
import os
import sys
import urllib.request
import talib.abstract as ta
import numpy as np
import ccxt
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_PATH = os.path.join(USER_DATA, "config.json")
STATE_PATH = os.path.join(USER_DATA, "regime_state.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def fetch_btc_daily(days=70):
    ex = ccxt.binance({"options": {"defaultType": "swap"}})
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    raw = ex.fetch_ohlcv("BTC/USDT:USDT", "1d", since=since, limit=days + 1)
    data = {
        "date": [datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc) for r in raw],
        "open": np.array([r[1] for r in raw], dtype=float),
        "high": np.array([r[2] for r in raw], dtype=float),
        "low": np.array([r[3] for r in raw], dtype=float),
        "close": np.array([r[4] for r in raw], dtype=float),
        "volume": np.array([r[5] for r in raw], dtype=float),
    }
    return data


def compute_regime(d):
    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]

    ema50 = ta.EMA(close, timeperiod=50)
    ema20 = ta.EMA(close, timeperiod=20)
    rsi14 = ta.RSI(close, timeperiod=14)
    adx14 = ta.ADX(high, low, close, timeperiod=14)
    atr14 = ta.ATR(high, low, close, timeperiod=14)

    last = {
        "close": close[-1],
        "ema50": ema50[-1],
        "ema20": ema20[-1],
        "rsi14": rsi14[-1],
        "adx14": adx14[-1],
        "date": d["date"][-1],
        "high_90d": np.max(close[-90:]),
    }
    last["atr_pct"] = (atr14[-1] / close[-1]) * 100

    vol_ma20 = np.mean(volume[-20:])
    last["vol_ratio"] = volume[-1] / vol_ma20 if vol_ma20 > 0 else 1.0

    if last["close"] > last["ema50"] and last["adx14"] > 25:
        regime = "BULL"
        emoji = "\U0001f7e2"
    elif last["close"] > last["ema50"]:
        regime = "BULL_WEAK"
        emoji = "\U0001f7e1"
    elif last["close"] < last["ema50"] and last["adx14"] > 25:
        regime = "BEAR"
        emoji = "\U0001f534"
    elif last["close"] < last["ema50"]:
        regime = "BEAR_WEAK"
        emoji = "\U0001f7e0"
    else:
        regime = "NEUTRAL"
        emoji = "\U000026ab"

    if last["close"] < last["high_90d"] * 0.75:
        regime = "EXTREME"
        emoji = "\u26ab"

    trend = "Strong +" if last["adx14"] > 30 and last["close"] > last["ema20"] else \
            "Weak +" if last["close"] > last["ema20"] else \
            "Strong -" if last["adx14"] > 30 else \
            "Weak -"

    vol_label = "High" if last["atr_pct"] > 3.0 else \
                "Medium" if last["atr_pct"] > 1.5 else "Low"

    return regime, emoji, last, trend, vol_label


def load_previous_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_state(regime, last):
    state = {
        "regime": regime,
        "price": round(last["close"], 1),
        "ema50": round(last["ema50"], 1),
        "timestamp": last["date"].isoformat(),
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(config, message):
    token = config.get("telegram", {}).get("token", "")
    chat_id = config.get("telegram", {}).get("chat_id", "")
    if not token or not chat_id:
        print("Telegram not configured. Message would be:")
        print(message)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


def build_message(regime, emoji, last, trend, vol_label, changed_from=None):
    pct_from_ema50 = ((last["close"] / last["ema50"]) - 1) * 100
    prefix = "Regime Change" if changed_from else "Regime Report"

    lines = [
        f"<b>{prefix}</b>",
        "",
        f"BTC:  ${last['close']:,.0f}  |  EMA50: ${last['ema50']:,.0f}  ({pct_from_ema50:+.1f}%)",
        f"Regime:  {emoji} <b>{regime}</b>" + (f"  (was {changed_from})" if changed_from else ""),
        f"Trend:  {trend}  |  ADX: {last['adx14']:.0f}",
        f"RSI(14):  {last['rsi14']:.0f}  |  Volatility: {vol_label} ({last['atr_pct']:.1f}%)",
        f"Volume:  {last['vol_ratio']:.1f}x MA20",
        "",
        f"{last['date'].strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    return "\n".join(lines)


def main():
    config = load_config()
    force = "--force" in sys.argv

    data = fetch_btc_daily()
    regime, emoji, last, trend, vol_label = compute_regime(data)

    prev = load_previous_state()
    changed = prev is None or prev.get("regime") != regime
    changed_from = prev.get("regime") if prev and prev["regime"] != regime else None

    if changed or force:
        msg = build_message(regime, emoji, last, trend, vol_label, changed_from)
        send_telegram(config, msg)
        save_state(regime, last)
        print(f"Alert sent: {regime}" + (f" (was {changed_from})" if changed_from else ""))
    else:
        print(f"No change: {regime} (same as previous)")

    print(f"  BTC: ${last['close']:,.0f}  EMA50: ${last['ema50']:,.0f}  "
          f"RSI: {last['rsi14']:.0f}  ADX: {last['adx14']:.0f}")


if __name__ == "__main__":
    main()
