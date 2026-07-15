# PhoenixScalper Evolution — Architecture

## Overview

The evolved PhoenixScalper transforms from a single Freqtrade strategy into a
production-grade autonomous adaptive quantitative trading platform with proper
separation of concerns, data lineage, risk management, and experiment tracking.

## Directory Structure

```
phoenix-scalper/
├── strategies/          # Freqtrade strategy modules
│   ├── PhoenixScalper.py      # Original base (unchanged)
│   ├── PhoenixScalperV2.py    # V2 with critical fixes
│   └── PhoenixScalperV3.py    # V3 with critical fixes
├── ml/                 # Machine learning modules
│   ├── hmm_regime.py          # HMM regime detection (look-ahead fixed)
│   ├── hmm_regime_v3.py       # V3 variant with StandardScaler
│   ├── kalman_filter.py       # Kalman price filter
│   ├── monte_carlo.py         # Monte Carlo validation
│   ├── feature_engine.py      # Training data generation
│   ├── train_strategy_model.py
│   ├── train_winrate_model.py
│   └── __init__.py
├── core/               # NEW: Central framework
│   ├── __init__.py
│   ├── data_quality.py        # Phase 2: Data validation + lineage
│   ├── regime_engine.py       # Phase 3: 7-state regime detection
│   ├── strategy_allocator.py  # Phase 4: Dynamic capital allocation
│   ├── trade_intel.py         # Phase 5: Trade attribution
│   ├── risk_governor.py       # Phase 12: Independent risk layer
│   ├── market_memory.py       # Phase 13: Long-term knowledge
│   ├── experiment_db.py       # Phase 9: Experiment management
│   ├── concept_drift.py       # Phase 8: Statistical drift detection
│   └── deployment.py          # Phase 14-15: Safe deployment + rollback
├── research/           # NEW: Quant research agent
│   ├── __init__.py
│   └── quant_agent.py         # Phase 6: Hypothesis generation
├── tests/              # NEW: Test suite
│   ├── core/
│   │   ├── test_data_quality.py
│   │   ├── test_regime_engine.py
│   │   ├── test_risk_governor.py
│   │   ├── test_concept_drift.py
│   │   └── test_strategy_allocator.py
│   └── ml/
│       ├── test_hmm_regime.py
│       ├── test_kalman_filter.py
│       └── test_monte_carlo.py
└── docs/
    └── architecture.md
```

## Data Flow

```
Exchange Data
    │
    ▼
DataValidator ──── Issues? ───→ Alert / Reject
    │
    ▼
populate_indicators()
    ├── HMM Regime Detection (no look-ahead)
    ├── Kalman Filter
    └── Technical Indicators
    │
    ▼
RegimeEngine ──── 7-state regime + confidence + recommendation
    │
    ▼
RiskGovernor ──── Daily DD, Weekly DD, Consec Losses, Exposure
    │
    ▼
StrategyAllocator ──── Dynamic capital allocation by regime/risk
    │
    ▼
TradeIntelligence ──── Full trade attribution + win/loss analysis
    │
    ▼
MarketMemory ──── Long-term knowledge storage
    │
    ▼
QuantResearchAgent ──── Pattern analysis → Hypothesis → Experiment
    │
    ▼
ExperimentDB ──── Full experiment versioning + results
```

## Validation Pipeline

```
Hypothesis
    │
    ▼
Backtest (historical)
    │
    ▼
Walk-Forward Validation
    │
    ▼
Monte Carlo Simulation
    │
    ▼
Shadow Mode (paper trade alongside live)
    │
    ▼
Canary Deployment (10% capital)
    │
    ▼
Full Deployment
    │
    ▼
Continuous Monitoring (Concept Drift + Risk Governor)
    │
    ▼
Automatic Rollback (if performance degrades)
```

## Critical Fixes Applied

1. **Class-level caches → instance-level** with TTL eviction
2. **`np.resize` → `_align_array`** in V2 to prevent data corruption
3. **HMM look-ahead fixed**: trains on historical window only, predicts on full
4. **Data validation layer**: NaN/Inf detection, staleness checks, gap detection
5. **Secrets removed from config.json**: env vars only via `{{VAR}}` substitution
