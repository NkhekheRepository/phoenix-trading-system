import json
import os
import subprocess
import urllib.request
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_PATH = os.path.join(USER_DATA, "config.json")

SERVICE_NAME = "phoenix30.service"
STATE_FILE = os.path.join(USER_DATA, "health_state.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def send_telegram(config, message):
    token = config.get("telegram", {}).get("token", "")
    chat_id = config.get("telegram", {}).get("chat_id", "")
    if not token or not chat_id:
        print(f"[HEALTH] No Telegram config. Would send:\n{message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[HEALTH] Telegram failed: {e}")


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {"alerted_down": False}
    return {"alerted_down": False}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def check_bot():
    r = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        capture_output=True, text=True, timeout=10
    )
    return r.stdout.strip() == "active"


def count_recent_restarts():
    r = subprocess.run(
        ["journalctl", "--user", "-u", SERVICE_NAME, "--since", "10 min ago",
         "--no-pager", "-o", "cat"],
        capture_output=True, text=True, timeout=10
    )
    return r.stdout.count("Started Phoenix30")


def main():
    config = load_config()
    state = load_state()
    bot_up = check_bot()

    if not bot_up:
        restarts = count_recent_restarts()
        if not state.get("alerted_down"):
            msg = (
                "\u26a0\ufe0f <b>Phoenix30 Bot DOWN</b>\n\n"
                f"Bot is not running. Restarts in last 10min: {restarts}\n"
                "Systemd will auto-restart."
            )
            send_telegram(config, msg)
            state["alerted_down"] = True
            save_state(state)
            print(f"[HEALTH] Bot DOWN - alert sent ({restarts} restarts)")
        else:
            print(f"[HEALTH] Bot DOWN - already alerted ({restarts} restarts)")
    else:
        if state.get("alerted_down"):
            msg = "\u2705 <b>Phoenix30 Bot Recovered</b>"
            send_telegram(config, msg)
            state["alerted_down"] = False
            save_state(state)
            print("[HEALTH] Bot recovered - recovery alert sent")
        else:
            print("[HEALTH] Bot OK")


if __name__ == "__main__":
    main()
