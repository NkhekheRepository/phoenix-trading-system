import logging
from pathlib import Path
from telegram import Update
from telegram.ext import CommandHandler

from core.ev_core import compute_ev, format_ev_table, EVResult

logger = logging.getLogger(__name__)

_ev_registered = False


def register_ev_command(telegram_instance) -> None:
    global _ev_registered
    if _ev_registered:
        return

    app = getattr(telegram_instance, "_app", None)
    if not app:
        logger.error("telegram_ev: Telegram instance has no _app")
        return

    authorized_users = telegram_instance._config.get("telegram", {}).get("authorized_users", [])
    config = telegram_instance._config

    async def _ev_handler(update: Update, context):
        user = update.effective_user
        if authorized_users and user and user.id not in authorized_users:
            await update.message.reply_text("⛔ Unauthorized")
            return

        try:
            db_path = Path(config["datadir"]) / "tradesv3.sqlite"
            if not db_path.exists():
                bot_name = config.get("strategy", "Bot")
                result = EVResult(bot_name=bot_name)
                result.blocked_count = _count_blocked(config)
                await update.message.reply_text(
                    format_ev_table(result),
                    parse_mode="Markdown",
                )
                return
            result = compute_ev(
                db_path=db_path,
                bot_name=config.get("strategy", "Bot"),
                score_threshold=config.get("score_threshold", 55),
                score_high_threshold=config.get("score_high_threshold", 60),
            )
            result.blocked_count = _count_blocked(config)
            await update.message.reply_text(
                format_ev_table(result),
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.exception("EV handler failed")
            await update.message.reply_text(f"Error: {e}")

    app.add_handler(CommandHandler("ev", _ev_handler))
    _ev_registered = True
    logger.info("telegram_ev: registered /ev command")


def _count_blocked(config) -> int:
    """Estimate blocked-entry count from recent logs."""
    import subprocess
    import os
    try:
        log_path = config.get("telegram", {}).get("log_path", "")
        if not log_path:
            return 0
        result = subprocess.run(
            ["grep", "-c", "not allowed", log_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        return 0
    except Exception:
        return 0
