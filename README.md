<p align="center">
  <h1 align="center">DCA Quant</h1>
  <p align="center">
    <strong>Institutional-Grade Quantitative Trading Platform</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#documentation">Docs</a>
  </p>
</p>

---

## Overview

**DCA Quant** is a full-stack quantitative trading platform combining systematic alpha generation, portfolio optimization, and risk management with professional-grade analytics visualization. Built for quantitative analysts and portfolio managers who demand institutional-quality tools.

| Component | Description |
|-----------|-------------|
| **Backend** | Python/FastAPI quant engine with multi-factor models, HMM regime detection, and portfolio optimization |
| **Frontend** | React 19 dashboard with real-time analytics, backtesting lab, and risk monitoring |
| **Infrastructure** | Docker microservices with PostgreSQL, Redis, and Nginx gateway |

---

## Features

### 🧮 Alpha Generation
- **8+ Academic Factors**: VSM, BAB, QMJ, PEAD, Sentiment, Accruals, IVOL, Revisions
- **HMM Regime Detection**: Bull/Bear market classification with dynamic factor weighting
- **Triple Barrier Labeling**: Path-dependent target generation for ML
- **XGBoost Meta-Labeling**: Confidence scoring for primary signals

### 📊 Portfolio Construction
- **Optimizers**: Mean-Variance, HRP, Black-Litterman, Multivariate Kelly
- **Risk Controls**: Volatility targeting, sector caps, position limits
- **Execution**: VWAP scheduling with market impact estimation

### 📈 Analytics Dashboard
- **Real-time P&L**: Portfolio tracking with position-level attribution
- **Backtest Lab**: Walk-forward CV, Monte Carlo simulations, factor attribution
- **Risk Monitor**: Component VaR, correlation heatmaps, tail hedging

### 🔬 Valuation Models
- **DCF**: Discounted Cash Flow with WACC estimation
- **DDM**: Dividend Discount Model with growth staging
- **RIM**: Residual Income Model for factor scoring
- **REIT**: Specialized FFO-based valuation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Nginx Gateway                            │
│                          (Port 8080)                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Valuation   │    │   Portfolio   │    │     Data      │
│    Service    │    │    Service    │    │    Service    │
│  (Port 8001)  │    │  (Port 8002)  │    │  (Port 8003)  │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      ┌──────────────┐              ┌──────────────┐
      │  PostgreSQL  │              │    Redis     │
      │   (5432)     │              │   (6379)     │
      └──────────────┘              └──────────────┘
```

### Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 19, TypeScript, Vite, TailwindCSS, Recharts, Framer Motion |
| **Backend** | Python 3.11, FastAPI, Pydantic, SQLAlchemy, DuckDB |
| **Quant** | NumPy, Pandas, SciPy, CVXPY, XGBoost, scikit-learn |
| **Infrastructure** | Docker, PostgreSQL, Redis, Nginx, Ray |
| **MLOps** | MLflow, Parquet, SHAP |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional)

### Option 1: Local Development

```bash
# Clone repository
git clone https://github.com/willychang21/DCA.git
cd DCA

# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Option 2: Docker Compose

```bash
# Start all services
docker-compose up -d

# Access
# - API Gateway: http://localhost:8080
# - Frontend: http://localhost:5173
```

---

## Project Structure

```
DCA/
├── backend/                 # Python quant engine
│   ├── app/                 # FastAPI application
│   │   ├── api/v1/          # REST endpoints
│   │   ├── domain/          # Models & schemas
│   │   ├── engines/         # Business logic
│   │   └── services/        # Service layer
│   ├── quant/               # Quantitative core
│   │   ├── features/        # Alpha factors
│   │   ├── portfolio/       # Optimizers
│   │   ├── regime/          # HMM detection
│   │   ├── backtest/        # Backtesting
│   │   └── risk/            # Risk management
│   ├── scripts/             # Operational scripts
│   └── tests/               # Test suite
│
├── frontend/                # React dashboard
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── pages/           # Route pages
│   │   ├── api/             # API client
│   │   └── store/           # State management
│   └── ...
│
├── docker-compose.yml       # Container orchestration
├── nginx.conf               # API gateway config
└── README.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Backend README](./backend/README.md) | Complete quant engine documentation |
| [Frontend README](./frontend/README.md) | Dashboard architecture & components |

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/quant/rankings` | GET | Multi-factor stock rankings |
| `/api/v1/quant/portfolio` | GET | Optimized portfolio weights |
| `/api/v1/quant/backtest` | POST | Run backtest simulation |
| `/api/v1/valuation/{ticker}` | GET | Stock valuation (DCF/DDM/RIM) |
| `/api/v1/portfolios` | GET/POST | Portfolio CRUD |

---

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm run test
```

### CI/CD

GitHub Actions workflow (`.github/workflows/quant_ci.yml`):
- Runs backend tests on push
- Validates script dry runs
- Linting & type checking

---

## Performance

| Metric | Target |
|--------|--------|
| API Response (rankings) | < 500ms |
| Backtest (5yr, 50 stocks) | < 30s |
| Frontend FCP | < 1.2s |
| Factor Computation | 35x faster with DuckDB |

---

## Roadmap

- [ ] Real-time streaming via WebSockets
- [ ] Options pricing & Greeks
- [ ] Distributed backtesting with Ray
- [ ] Mobile-responsive dashboard
- [ ] Alternative data integration

---

## License

MIT © 2024 DCA Quant Team

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  <sub>Built with ❤️ for quantitative finance</sub>
</p>
