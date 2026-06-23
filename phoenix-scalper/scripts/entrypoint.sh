#!/bin/bash
set -e

# Force IPv4-only for Binance connections
export AIODNS_RESOLVE_USE_SOCKET=1

# Substitute env vars in config if using mustache-style placeholders
if [ -f /freqtrade/config.json ]; then
    echo "Applying environment variable substitutions to config..."
    sed -i "s|{{TELEGRAM_BOT_TOKEN}}|${TELEGRAM_BOT_TOKEN:-}|g" /freqtrade/config.json
    sed -i "s|{{TELEGRAM_CHAT_ID}}|${TELEGRAM_CHAT_ID:-}|g" /freqtrade/config.json
    sed -i "s|{{EXCHANGE_API_KEY}}|${EXCHANGE_API_KEY:-}|g" /freqtrade/config.json
    sed -i "s|{{EXCHANGE_API_SECRET}}|${EXCHANGE_API_SECRET:-}|g" /freqtrade/config.json
    sed -i "s|{{EXCHANGE_PASSWORD}}|${EXCHANGE_PASSWORD:-}|g" /freqtrade/config.json
    sed -i "s|{{API_USERNAME}}|${API_USERNAME:-freqtrader}|g" /freqtrade/config.json
    sed -i "s|{{API_PASSWORD}}|${API_PASSWORD:-freqtrader}|g" /freqtrade/config.json
    sed -i "s|{{JWT_SECRET}}|${JWT_SECRET:-change-me}|g" /freqtrade/config.json
fi

# Start freqtrade with all passed arguments
exec freqtrade "$@"
