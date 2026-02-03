# NEXUS AI v3.0

Advanced Sports Prediction System powered by LangGraph, ML Ensemble Models, and Claude AI.

**Repository:** https://github.com/szymonMCS/NEXUS_AI.git

---

## Overview

NEXUS AI is a production-grade sports prediction and value betting system. It collects real-time data from multiple sources, runs predictions through an ensemble of ML models and LLM analysis, identifies value bets with positive expected value, and manages risk through Kelly Criterion position sizing.

### Key Capabilities

- **Multi-agent orchestration** via LangGraph (Supervisor, Analyst, Ranker, Risk Manager, Decision Maker, Bettor)
- **ML Ensemble**: Random Forest + ARA, MLP + PCA, Transformers, GNN, Quantum NN
- **11 sports supported**: Tennis, Basketball, American Football, Baseball, Hockey, Soccer, MMA, Olympics, Greyhound Racing, Handball, Table Tennis
- **Real-time frontend** with React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui
- **FastAPI backend** with REST API, WebSocket live updates, rate limiting, health checks
- **Dual mode**: Pro (paid APIs) or Lite (free sources + scraping)
- **Database persistence**: SQLAlchemy ORM with match tracking, odds history, bet settlement
- **A/B Testing Framework** with statistical significance testing

### Model Performance

| Model | Accuracy | Dataset |
|-------|----------|---------|
| Random Forest + ARA | 81.9% | Football-Data.co.uk |
| MLP + PCA | 86.7% | Football-Data.co.uk |
| Tennis (ScoreNetwork) | 97.1% | ScoreNetworkData |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts |
| Backend | FastAPI, Python 3.11+, SQLAlchemy 2.0, Pydantic |
| Auth | Clerk (React + backend middleware) |
| ML | scikit-learn, PyTorch (Transformers, GNN, QNN) |
| Orchestration | LangGraph multi-agent workflow |
| LLM | Claude (Anthropic) / GPT (OpenAI) / Kimi (Moonshot) |
| Database | PostgreSQL (recommended) or SQLite |
| Cache | Redis |
| Real-time | WebSocket with auto-reconnection |
| Deployment | Docker Compose, Prometheus + Grafana monitoring |

---

## Data Sources

### Pro Mode (Paid APIs)
- The Odds API - live odds from 40+ bookmakers
- API-Tennis - detailed tennis statistics
- BetsAPI - multi-sport data
- NewsAPI - full news access
- **Cost: ~$150-200/month**

### Lite Mode (Free/Minimal Cost)
- TheSportsDB (free public API)
- Sofascore scraping
- Flashscore scraping
- Football-Data.co.uk historical data
- Brave Search / Serper for news
- **Cost: ~$0-50/month** (LLM API costs only)

Switch modes via `APP_MODE=lite` or `APP_MODE=pro` in `.env`.

---

## Requirements

### Backend
- Python 3.11+
- Redis (for caching)
- PostgreSQL (recommended) or SQLite

### Frontend
- Node.js 18+ (recommended: 20 LTS)
- npm 9+

---

## Installation

```bash
# Clone
git clone https://github.com/szymonMCS/NEXUS_AI.git
cd nexus

# Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium  # If using scraping in Lite mode

# Environment variables
cp .env.example .env
# Edit .env and add your API keys

# Initialize database
python scripts/init_db.py

# Frontend
cd frontend/app
npm install
cp .env.example .env
# Edit .env and add Clerk keys
```

---

## Usage

### API Server
```bash
python main.py --port 8000
```

### Frontend Dev Server
```bash
cd frontend/app
npm run dev
# Opens at http://localhost:5173
```

### CLI Analysis
```bash
python main.py --analyze tennis --date 2026-01-21
```

### Full Production Pipeline
```bash
python scripts/full_production_pipeline.py --mode full --samples 100
```

### Development (both backend + frontend)
```bash
bash scripts/start_dev.sh
```

---

## Architecture

```
NEXUS AI v3.0
+-- React Frontend (Vite + shadcn/ui)
|   +-- Dashboard with live KPIs from API
|   +-- Predictions, Handicaps, Reports, Statistics pages
|   +-- WebSocket real-time updates (auto-reconnect)
|   +-- Clerk authentication
|
+-- FastAPI Backend
|   +-- REST API + WebSocket /api/ws
|   +-- Rate limiting (60 req/min general, 5/min analysis)
|   +-- Health endpoint (/health) with DB + Redis checks
|   +-- Lifespan management (startup/shutdown)
|   +-- CORS from settings
|
+-- LangGraph Orchestrator
|   +-- Supervisor -> News Analyst -> Data Evaluator
|   +-- Analyst (LLM + Statistical) -> Match Ranker
|   +-- Risk Manager -> Decision Maker -> Bettor
|
+-- ML Ensemble
|   +-- RandomForestEnsembleModel (200 trees + ARA)
|   +-- MLPNeuralNetwork (128->64->32 + PCA)
|   +-- SportsTransformer (Multi-head attention)
|   +-- GraphNeuralNetwork (Team chemistry)
|   +-- QuantumNeuralNetwork (Simulated)
|
+-- Data Pipeline
|   +-- Collectors (fixtures, odds, news, stats)
|   +-- DataPersistenceService (DB bridge with deduplication)
|   +-- Result tracking + automatic bet settlement
|
+-- Database (SQLAlchemy ORM)
|   +-- Match, Odds, Prediction, Bet, News, MatchStats
|   +-- BettingSession, SystemMetrics
|
+-- MCP Servers (News, Odds, Tennis, Basketball, Alerts, Evaluation)
+-- Data Sources (Configurable: Paid APIs or Free sources)
```

---

## Sports Supported

| Sport | Status | Data Source |
|-------|--------|------------|
| Tennis | Active | ScoreNetworkData (1M samples) |
| Basketball | Active | ScoreNetworkData (267K samples) |
| American Football | Active | ScoreNetworkData (318K samples) |
| Baseball | Active | ScoreNetworkData (106K samples) |
| Hockey | Active | ScoreNetworkData (103K samples) |
| Soccer | Active | ScoreNetworkData (41K samples) |
| MMA | Active | ScoreNetworkData (203K samples) |
| Olympics | Active | ScoreNetworkData (83K samples) |
| Greyhound Racing | Beta | Collected data |
| Handball | Beta | Collected data |
| Table Tennis | Beta | Collected data |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (DB + Redis status) |
| GET | `/api/status` | System status |
| GET | `/api/stats` | Performance statistics from DB |
| GET | `/api/sports/available` | Available sports list |
| GET | `/api/predictions` | Get predictions |
| GET | `/api/predictions/live` | Live analysis progress |
| POST | `/api/analysis` | Run analysis for a sport |
| GET | `/api/matches` | Get matches from DB |
| GET | `/api/match/{id}` | Match details with AI analysis |
| GET | `/api/value-bets` | Current value bets |
| POST | `/api/handicap` | Handicap prediction |
| GET | `/api/handicap/markets` | Available handicap markets |
| POST | `/api/collect` | Collect fixtures without analysis |
| POST | `/api/results/check` | Check and update match results |
| WS | `/api/ws` | WebSocket for real-time updates |

---

## Project Structure

```
nexus/
+-- api/                    # FastAPI backend
|   +-- main.py             # Main API with lifespan, rate limiting, health checks
|   +-- routers.py          # Additional routers
|   +-- metrics.py          # Prometheus metrics
|
+-- agents/                 # LangGraph agents
|   +-- supervisor.py       # Main orchestrator
|   +-- analyst.py          # LLM predictions
|   +-- statistical_model_agent.py
|   +-- ranker.py           # Match ranking
|   +-- risk_manager.py     # Kelly Criterion
|   +-- sports_data_swarm/  # Multi-agent data collection
|
+-- core/                   # Core functionality
|   +-- ml/                 # ML models (RF, MLP, Transformers, GNN, QNN)
|   +-- datasets/           # Sport-specific data loaders
|   +-- quality_scorer.py   # Data quality evaluation
|
+-- data/                   # Data collection
|   +-- persistence.py      # DB bridge service
|   +-- odds/               # Odds APIs and scrapers
|   +-- news/               # News aggregation
|   +-- collectors/         # Multi-source collectors
|   +-- scrapers/           # Web scrapers (Flashscore, Sofascore)
|   +-- score_network/      # ScoreNetworkData (8 sports)
|
+-- database/               # Database layer
|   +-- db.py               # Connection, session management
|   +-- models.py           # SQLAlchemy ORM models
|   +-- crud.py             # CRUD operations
|
+-- config/                 # Configuration
|   +-- settings.py         # Pydantic settings (auto SECRET_KEY, CORS)
|   +-- free_apis.py        # Free API configuration
|
+-- models/                 # Trained models
|   +-- score_network/      # 8 disciplines x 2 models
|   +-- trained/            # Football-Data models
|
+-- frontend/app/           # React + TypeScript + Vite
|   +-- src/pages/app/      # Dashboard, Predictions, Handicaps, etc.
|   +-- src/components/     # UI components (shadcn/ui)
|   +-- src/lib/api.ts      # API client with WebSocket reconnection
|
+-- scripts/                # Utility scripts
+-- tests/                  # Unit and integration tests
+-- main.py                 # Main entry point
+-- betting_floor.py        # Main orchestration (BettingFloor)
+-- docker-compose.yml      # Docker orchestration
```

---

## Configuration

Edit `.env`:

```bash
# Mode
APP_MODE=lite   # or "pro"

# LLM (at least one required)
ANTHROPIC_API_KEY=your_key
# or OPENAI_API_KEY=your_key

# News (at least one recommended)
BRAVE_API_KEY=your_key
# or SERPER_API_KEY=your_key

# Pro mode only
ODDS_API_KEY=your_key
API_TENNIS_KEY=your_key
BETS_API_KEY=your_key

# Database
DATABASE_URL=sqlite:///./nexus.db

# Betting
DEFAULT_BANKROLL=1000.0
KELLY_FRACTION=0.25
```

---

## Monitoring

- **Health Check**: http://localhost:8000/health
- **Prometheus Metrics**: http://localhost:8000/metrics
- **Grafana Dashboard**: http://localhost:3030
- **API Status**: http://localhost:8000/api/status

---

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

---

## Testing

```bash
# Backend tests
python -m pytest tests/ -v
python -m pytest tests/ --cov=. --cov-report=term-missing

# Frontend tests
cd frontend/app && npm test

# Production pipeline test
python scripts/full_production_pipeline.py --mode test
```

---

## License

MIT

---

## Disclaimer

This software is for educational purposes only. Sports betting involves risk. Please gamble responsibly and within your means.
