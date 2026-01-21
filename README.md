# 🎯 NEXUS AI v2.2.0

Advanced Sports Prediction System powered by LangGraph, MCP, Statistical Models, and Claude AI.

## 🚀 Features

- **Multi-Sport Support**: Tennis, Basketball, Greyhound Racing, Handball, Table Tennis
- **Statistical Models**: Advanced prediction models for each sport (SVR, SEL, XGBoost ensembles)
- **Intelligent Data Aggregation**: News from multiple sources (Brave, Serper, NewsAPI)
- **Quality-Based Filtering**: Automatic data quality evaluation
- **Top 3 Ranking System**: Focus on highest value opportunities
- **MCP Server Architecture**: Modular, scalable design
- **Real-time Odds Comparison**: Multiple bookmakers (API + optional scraping)
- **Risk Management**: Kelly Criterion position sizing
- **LangGraph Orchestration**: Multi-agent workflow
- **React Frontend**: Beautiful web UI with live updates
- **FastAPI Backend**: REST API + WebSocket for real-time updates

## 💡 Flexible Data Sources

NEXUS AI supports **two configuration modes**:

### 🔸 Standard Mode (Paid APIs)
- The Odds API ($50-100/month)
- API-Tennis ($50/month)
- BetsAPI ($30/month)
- **Total: ~$150-200/month**
- Best data quality and reliability

### 🔹 Lite Mode (Free/Minimal Cost)
- TheSportsDB (free)
- Sofascore scraping (free)
- Flashscore scraping (free)
- Polish bookies scraping (free)
- **Total: ~$0-50/month** (only Claude API costs)
- Good quality with validation

**Switch modes** by setting `APP_MODE=lite` or `APP_MODE=pro` in `.env`

## 📋 Requirements

- Python 3.11+
- Redis (for caching)
- PostgreSQL (recommended) or SQLite
- API Keys (see `.env.example`)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/szymonMCS/NEXUS_AI.git
cd nexus
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium  # If using scraping in Lite mode
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. Initialize database:
```bash
python scripts/init_db.py
```

## 🚀 Usage

### Run the API Server:
```bash
python main.py --port 8000
```

### Run the React Frontend:
```bash
cd frontend/app
npm run dev
```

### Run Development Mode:
```bash
bash scripts/start_dev.sh
```

### Run Production Mode:
```bash
bash scripts/start_prod.sh
```

### Run CLI Analysis:
```bash
python main.py --analyze tennis --date 2026-01-21
```

## 📊 Architecture

```
NEXUS AI v2.2.0
├── React Frontend (Vite + shadcn/ui)
│   ├── API Client with TypeScript
│   ├── WebSocket real-time updates
│   └── Sports Selector
│
├── FastAPI Backend
│   ├── REST API Endpoints
│   ├── WebSocket /api/ws
│   ├── Prometheus Metrics
│   └── CORS configured
│
├── LangGraph Orchestrator (Multi-agent workflow)
│   ├── Supervisor Agent
│   ├── News Analyst Agent
│   ├── Data Evaluator Agent
│   ├── Analyst Agent (LLM + Statistical)
│   ├── Statistical Model Agent (NEW)
│   ├── Match Ranker Agent
│   ├── Risk Manager Agent
│   ├── Decision Maker Agent
│   └── Bettor Agent
│
├── Statistical Prediction Models
│   ├── TennisModel (ELO, Form, H2H, Surface, Fatigue)
│   ├── BasketballModel (Ratings, Rest, Home/Away)
│   ├── GreyhoundModel (SVR/SVM ensemble)
│   ├── HandballModel (SEL approach with CMP)
│   └── TableTennisModel (XGBoost/RF ensemble)
│
├── MCP Servers (News, Odds, Tennis, Basketball, Alerts, Evaluation)
└── Data Sources (Configurable: Paid APIs or Free sources)
```

## 📦 Sports Supported

| Sport | Status | Model Type |
|-------|--------|------------|
| Tennis | ✅ Active | ELO-based with form factors |
| Basketball | ✅ Active | Ratings-based with rest analysis |
| Greyhound Racing | 🟡 Beta | SVR/SVM ensemble |
| Handball | 🟡 Beta | SEL (CMP distribution) |
| Table Tennis | 🟡 Beta | XGBoost/RF ensemble |

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_api_endpoints.py -v

# Run frontend tests
cd frontend/app && npm test
```

## 📁 Project Structure

```
nexus/
├── api/                    # FastAPI backend
│   ├── main.py            # Main API application
│   ├── routers.py         # Additional routers
│   └── metrics.py         # Prometheus metrics
│
├── agents/                # LangGraph agents
│   ├── supervisor.py      # Main orchestrator
│   ├── analyst.py         # LLM predictions
│   ├── statistical_model_agent.py  # Statistical predictions
│   ├── ranker.py          # Match ranking
│   ├── risk_manager.py    # Kelly Criterion
│   └── ...
│
├── core/                  # Core functionality
│   ├── models/            # Statistical models
│   │   ├── tennis_model.py
│   │   ├── basketball_model.py
│   │   ├── greyhound_model.py
│   │   ├── handball_model.py
│   │   └── table_tennis_model.py
│   ├── quality_scorer.py  # Data quality evaluation
│   ├── value_calculator.py # Value bet calculations
│   └── state.py           # LangGraph state
│
├── data/                  # Data collection
│   ├── odds/              # Odds APIs and scrapers
│   ├── news/              # News aggregation
│   └── collectors/        # Multi-source collectors
│
├── frontend/app/          # React frontend
│   ├── src/
│   │   ├── lib/          # API client, utilities
│   │   ├── hooks/        # React hooks
│   │   ├── components/   # UI components
│   │   └── sections/     # Page sections
│   └── ...
│
├── scripts/               # Utility scripts
│   ├── start_dev.sh      # Dev startup
│   ├── start_prod.sh     # Production startup
│   ├── run_tests.sh      # Test runner
│   └── ...
│
├── tests/                 # Unit and integration tests
├── main.py               # Main entry point
├── pyproject.toml        # Project configuration
└── docker-compose.yml    # Docker orchestration
```

## 🔄 Configuration

Edit `.env` to choose your mode:

```bash
# Standard Mode (Paid APIs)
APP_MODE=pro
ODDS_API_KEY=your_key
API_TENNIS_KEY=your_key
BETS_API_KEY=your_key

# Lite Mode (Free sources)
APP_MODE=lite
USE_WEB_SCRAPING=True
USE_FREE_APIS=True
# Only Brave/Serper + Anthropic keys needed
```

## 📈 Monitoring

- **Prometheus Metrics**: http://localhost:8000/metrics
- **Grafana Dashboard**: http://localhost:3030
- **API Health**: http://localhost:8000/api/status

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d
```

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss changes.

## ⚠️ Disclaimer

This software is for educational purposes only. Sports betting involves risk.
Please gamble responsibly and within your means.
