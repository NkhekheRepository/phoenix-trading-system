#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Phoenix Trading System — Setup              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not found. Install: curl -fsSL https://get.docker.com | bash"
    exit 1
fi
info "Docker: $(docker --version)"

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    error "Docker Compose not found."
    exit 1
fi
info "Compose: $(docker compose version)"

# Determine which bots to deploy
echo ""
echo "Select bots to deploy:"
echo "  1) V2.1 only"
echo "  2) V3.1 only"
echo "  3) V5-BTC only"
echo "  4) All 3 bots (recommended)"
echo ""
read -p "Choice [1-4]: " CHOICE

case "$CHOICE" in
  1) BOTS=("v2.1");;
  2) BOTS=("v3.1");;
  3) BOTS=("v5-btc");;
  4) BOTS=("v2.1" "v3.1" "v5-btc");;
  *) error "Invalid choice"; exit 1;;
esac

BOT_DIR="/home/$(whoami)/phoenix-trading-system/phoenix-scalper"
cd "$BOT_DIR"

# Create data dirs
mkdir -p user_data/logs user_data/data user_data/backtest_results
mkdir -p user_data/hyperopt_results user_data/plot user_data/freqaimodels

# Check .env
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        info ".env created from template"
        warn "EDIT .env with your credentials: nano $BOT_DIR/.env"
        read -p "Press Enter after editing .env (or Ctrl+C to abort)..."
    else
        error ".env.example not found"
        exit 1
    fi
fi

# Export vars from .env
set -a
source .env
set +a

deploy_bot() {
    local VER=$1
    local COMPOSE_FILE="docker-compose-${VER}.yml"
    local CONTAINER_NAME="phoenix-scalper-${VER}-bot"

    info "Building and starting $CONTAINER_NAME..."
    docker compose -f "$COMPOSE_FILE" up -d --build

    if docker ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        info "$CONTAINER_NAME is running"
    else
        error "$CONTAINER_NAME failed to start. Check logs: docker logs $CONTAINER_NAME"
        exit 1
    fi
}

echo ""
info "Deploying ${BOTS[*]}..."

for ver in "${BOTS[@]}"; do
    deploy_bot "$ver"
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Deployment Complete                          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Verify
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep phoenix

echo ""
info "Next steps:"
info "  1. Download market data:"
echo "       docker exec phoenix-scalper-v2.1-bot freqtrade download-data \\"
echo "         --config /freqtrade/config.json \\"
echo "         --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT \\"
echo "                XRP/USDT:USDT DOGE/USDT:USDT BNB/USDT:USDT \\"
echo "                ADA/USDT:USDT LINK/USDT:USDT \\"
echo "         --days 15 --timeframes 5m"
info "  2. Set up cron for data refresh:"
echo "       crontab -e"
echo "       # Add: 3 * * * * /home/$(whoami)/phoenix-trading-system/scripts/refresh_data.sh"
info "  3. Check Telegram for trade notifications"
info "  4. See docs/ for full documentation"
echo ""
info "API endpoints:"
for ver in "${BOTS[@]}"; do
    case "$ver" in
        v2.1) PORT=8082;;
        v3.1) PORT=8083;;
        v5-btc) PORT=8085;;
    esac
    echo "  $ver: curl -u freqtrader:YOUR_PASS http://127.0.0.1:$PORT/api/v1/ping"
done
echo ""
