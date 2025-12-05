# DCA Quant Backend (Tier-3 Enterprise Grade)

The **DCA Quant Backend** is an institutional-grade quantitative trading engine. It has evolved from a static regime-based system (Tier-1) to an adaptive, risk-constrained platform (Tier-3) capable of geometric growth maximization and real-time execution.

---

## 1. System Architecture (Tier-3)

The system operates on a modular, layered architecture designed for scalability, adaptability, and robustness.

```mermaid
graph TD
    subgraph Data Layer
        YF[YFinance] -->|OHLCV| DL[Data Lake]
        WS[WebSocket] -->|Real-Time Ticks| DL
        DL -->|Price & Fundamentals| FE[Feature Engineering]
    end

    subgraph Alpha Engine (Tier-1)
        FE -->|VSM, BAB, QMJ| RE[Ranking Engine]
        HMM[HMM Regime Detector] -->|Bull/Bear| RE
        RE -->|Primary Signal| ML[ML Layer]
    end

    subgraph Machine Learning (Tier-2)
        ML -->|Triple Barrier| TB[Labeling]
        TB -->|Meta-Labeling| XGB[XGBoost Classifier]
        XGB -->|Confidence Score| PO
    end

    subgraph Portfolio Construction (Tier-2)
        PO[Portfolio Optimizer] -->|Kelly Criterion| KO[Sizing]
        KO -->|Vol Targeting| RC[Risk Control]
        RC -->|Target Weights| EX
    end

    subgraph Execution & Risk (Tier-2/3)
        EX[Execution Algo] -->|VWAP Schedule| Broker
        EX -->|Tail Hedging| Options[Put/VIX Calls]
        EX -->|Component VaR| RiskMon[Risk Monitor]
    end

    subgraph MLOps (Tier-3)
        MLflow[Model Registry] -->|Track Experiments| XGB
        Ray[Distributed Cluster] -->|Parallel Backtest| PO
    end
```

### Core Components

*   **Data Layer (`quant.data`)**:
    *   **Ingestion**: YFinance (Batch) and WebSocket (Real-Time).
    *   **Storage**: DuckDB/Parquet Data Lake.
*   **Alpha Engine (`quant.features`)**:
    *   **Factors**: VSM, BAB, QMJ, PEAD, Sentiment.
    *   **Regime**: HMM-based market state detection.
*   **ML Layer (`quant.mlops`, `quant.research`)**:
    *   **Labeling**: Triple Barrier Method (Profit/Stop/Time).
    *   **Meta-Labeling**: XGBoost model to filter false positives.
    *   **Optimization**: Genetic Algorithms (DEAP) for hyperparameter tuning.
*   **Portfolio Engine (`quant.portfolio`)**:
    *   **Sizing**: Multivariate Kelly Optimization (Growth Maximization).
    *   **Risk**: Volatility Targeting (Dynamic Leverage).
*   **Execution & Risk (`quant.execution`, `quant.risk`)**:
    *   **Algo**: VWAP Execution with Market Impact estimation.
    *   **Hedging**: Systematic Tail Hedging (Puts) and Component VaR decomposition.
*   **Infrastructure (`quant.backtest`)**:
    *   **Distributed**: Ray-based parallel backtesting.
    *   **MLOps**: MLflow for experiment tracking and model versioning.

---

## 2. Directory Structure

```
backend/
├── app/                        # FastAPI Application Layer
│   ├── api/v1/                 # REST API Endpoints
│   │   ├── endpoints/          # quant, signals, dashboard
│   │   └── router.py           # Main Router
├── quant/                      # Quantitative Core
│   ├── backtest/               # Simulation Engine
│   │   ├── distributed.py      # Ray Parallel Backtester
│   │   ├── validation.py       # Purged Walk-Forward CV
│   │   └── statistics.py       # Deflated Sharpe Ratio
│   ├── data/                   # Data Access
│   │   └── realtime/           # WebSocket & Stream Clients
│   ├── execution/              # Execution Algorithms
│   │   └── algo.py             # VWAP Execution
│   ├── features/               # Alpha Factors
│   │   ├── labeling.py         # Triple Barrier Labeling
│   │   └── meta_labeling.py    # XGBoost Meta-Labeler
│   ├── mlops/                  # MLOps Infrastructure
│   │   └── registry.py         # MLflow Model Registry
│   ├── portfolio/              # Portfolio Construction
│   │   ├── kelly.py            # Kelly Optimization
│   │   └── risk_control.py     # Volatility Targeting
│   ├── research/               # Research Tools
│   │   └── evolution.py        # Genetic Algorithms
│   ├── risk/                   # Risk Management
│   │   ├── hedging.py          # Tail Hedging
│   │   └── var.py              # Component VaR
│   └── valuation/              # Intrinsic Valuation
├── scripts/                    # Operational Workflows
└── tests/                      # Test Suite (Pytest)
```

---

## 3. Operational Guide

### Running the System
1.  **Start the Backend Server**:
    ```bash
    fastapi dev main.py
    ```
2.  **Access the Dashboard**:
    Navigate to `http://localhost:5173/advanced` (Frontend).

### Key Workflows

#### 1. Train Meta-Labeling Model
```python
from quant.features.meta_labeling import MetaLabeler
# ... load data ...
model = MetaLabeler()
model.train(X_train, y_train)
```

#### 2. Run Genetic Optimization (Distributed)
```bash
python -m scripts.run_genetic_opt --use-ray
```

#### 3. Execute Trades (VWAP)
```python
from quant.execution.algo import VWAPExecution
algo = VWAPExecution()
schedule = algo.generate_schedule(total_shares=1000)
```

### API Reference (Tier-3)
*   `GET /api/v1/quant/ml/signals`: Latest XGBoost signal confidence.
*   `GET /api/v1/quant/risk/metrics`: Real-time VaR and Hedge Cost.
*   `GET /api/v1/quant/execution/vwap`: VWAP execution schedule.

---

## 4. Development Setup

### Prerequisites
*   **Python 3.10+**
*   **Redis** (for Ray/Celery)
*   **MLflow** (for Experiment Tracking)

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Install Tier-3 deps: ray, mlflow, xgboost, cvxpy
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific Tier-3 tests
pytest tests/test_tier3_mlops.py
pytest tests/test_tier3_distributed.py
pytest tests/test_tier3_realtime.py
```
