#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <sandbox>"
    echo "  sandbox: phoenix30 | phoenix-scalper"
    exit 1
fi

SANDBOX="$1"

if [ ! -d "$SANDBOX" ]; then
    echo "ERROR: Sandbox directory '$SANDBOX' not found"
    exit 1
fi

cd "$SANDBOX"

echo "=== Setting up $SANDBOX ==="

if [ ! -f .env ]; then
    cp .env.example .env
    echo "  -> .env created. EDIT IT: nano .env"
else
    echo "  -> .env already exists"
fi

if [ ! -f config.json ]; then
    cp config.json.example config.json
    echo "  -> config.json created from template"
else
    echo "  -> config.json already exists"
fi

mkdir -p data/logs

echo ""
echo "=== Ready ==="
echo "  1. Edit: nano .env"
echo "  2. Edit: nano config.json"
echo "  3. Run:  docker compose up -d"
echo "  4. Logs: docker compose logs -f"
