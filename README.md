# 🎯 NEXUS AI v3.0 - Cutting Edge

Advanced Sports Prediction System powered by LangGraph, MCP, Statistical Models, Cutting-Edge ML, and Claude AI.

**Repository:** https://github.com/szymonMCS/NEXUS_AI.git

---

## 🚀 What's New in v3.0

### ✨ Major Features:
- **🎾 ScoreNetworkData Integration**: 2.1M+ samples from 8 sport disciplines
- **🧠 Cutting-Edge ML Models**: Random Forest + ARA, MLP + PCA, Transformers, GNN, Quantum NN
- **⚡ A/B Testing Framework**: Statistical significance testing with p-value calculation
- **🔄 Automated Training Pipeline**: Full production pipeline with data collection, retraining, deployment
- **📊 Multi-Sport Support**: Tennis, Basketball, American Football, Baseball, Hockey, Soccer, MMA, Olympics

### 🏆 Model Performance:
| Model | Accuracy | Dataset |
|-------|----------|---------|
| Random Forest + ARA | 81.9% | Football-Data.co.uk |
| MLP + PCA | 86.7% | Football-Data.co.uk |
| Tennis (ScoreNetwork) | 97.1% | ScoreNetworkData |

---

## 🚀 Features

- **Multi-Sport Support**: Tennis, Basketball, Greyhound Racing, Handball, Table Tennis, American Football, Baseball, Hockey, Soccer, MMA, Olympics
- **Statistical Models**: Advanced prediction models (SVR, SEL, XGBoost, Random Forest, MLP, Transformers, GNN)
- **Cutting-Edge ML**: ARA (Artificial Raindrop Algorithm), PCA, Quantum Neural Networks, Graph Neural Networks
- **Intelligent Data Aggregation**: News from multiple sources (Brave, Serper, NewsAPI)
- **Quality-Based Filtering**: Automatic data quality evaluation
- **Top 3 Ranking System**: Focus on highest value opportunities
- **MCP Server Architecture**: Modular, scalable design
- **Real-time Odds Comparison**: Multiple bookmakers (API + optional scraping)
- **Risk Management**: Kelly Criterion position sizing + RL Staking Optimizer
- **LangGraph Orchestration**: Multi-agent workflow
- **React Frontend**: Beautiful web UI with live updates
- **FastAPI Backend**: REST API + WebSocket for real-time updates
- **A/B Testing**: Compare baseline vs cutting-edge models statistically

---

## 💡 Flexible Data Sources

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
- Football-Data.co.uk (free)
- **Total: ~$0-50/month** (only Claude API costs)
- Good quality with validation

**Switch modes** by setting `APP_MODE=lite` or `APP_MODE=pro` in `.env`

---

## 📋 Requirements

- Python 3.11+
- Redis (for caching)
- PostgreSQL (recommended) or SQLite
- API Keys (see `.env.example`)

---

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

---

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

### Run Full Production Pipeline:
```bash
python scripts/full_production_pipeline.py --mode full --samples 100
```

### Train Models on ScoreNetworkData:
```bash
python scripts/train_score_network_models.py
```

---

## 📊 Architecture

```
NEXUS AI v3.0 - Cutting Edge
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
│   ├── Statistical Model Agent
│   ├── Match Ranker Agent
│   ├── Risk Manager Agent
│   ├── Decision Maker Agent
│   └── Bettor Agent
│
├── Cutting-Edge ML Models
│   ├── RandomForestEnsembleModel (200 trees + ARA)
│   ├── MLPNeuralNetwork (128→64→32 + PCA)
│   ├── SportsTransformer (Multi-head attention)
│   ├── GraphNeuralNetwork (Team chemistry)
│   └── QuantumNeuralNetwork (Simulated)
│
├── A/B Testing Framework
│   ├── ABTestingFramework
│   ├── PredictionRecord tracking
│   └── Statistical significance (p-value)
│
├── MCP Servers (News, Odds, Tennis, Basketball, Alerts, Evaluation)
└── Data Sources (Configurable: Paid APIs or Free sources)
    ├── Football-Data.co.uk (38,780 matches)
    └── ScoreNetworkData (2.1M samples, 8 sports)
```

---

## 📦 Sports Supported

| Sport | Status | Model Type | Data Source |
|-------|--------|------------|-------------|
| Tennis | ✅ Active | ELO + Form + ScoreNetwork RF/MLP | ScoreNetworkData (1M samples) |
| Basketball | ✅ Active | Ratings + ScoreNetwork RF/MLP | ScoreNetworkData (267K samples) |
| American Football | ✅ Active | ScoreNetwork RF/MLP | ScoreNetworkData (318K samples) |
| Baseball | ✅ Active | ScoreNetwork RF/MLP | ScoreNetworkData (106K samples) |
| Hockey | ✅ Active | ScoreNetwork RF/MLP | ScoreNetworkData (103K samples) |
| Soccer | ✅ Active | ScoreNetwork RF/MLP | ScoreNetworkData (41K samples) |
| MMA | ✅ Active | ScoreNetwork RF/MLP | ScoreNetworkData (203K samples) |
| Olympics | ✅ Active | ScoreNetwork RF/MLP | ScoreNetworkData (83K samples) |
| Greyhound Racing | 🟡 Beta | SVR/SVM ensemble | Collected data |
| Handball | 🟡 Beta | SEL (CMP distribution) | Collected data |
| Table Tennis | 🟡 Beta | XGBoost/RF ensemble | Collected data |

---

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

# Run production pipeline test
python scripts/full_production_pipeline.py --mode test
```

---

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
│   ├── statistical_model_agent.py
│   ├── ranker.py          # Match ranking
│   ├── risk_manager.py    # Kelly Criterion
│   └── sports_data_swarm/ # Multi-agent data collection
│
├── core/                  # Core functionality
│   ├── ml/                # ML models
│   │   ├── models/        # RF, MLP, Transformers, GNN, QNN
│   │   ├── evaluation/    # A/B testing framework
│   │   ├── features/      # Feature engineering
│   │   └── training/      # Online training
│   ├── datasets/          # Sport-specific data loaders
│   └── quality_scorer.py  # Data quality evaluation
│
├── data/                  # Data collection
│   ├── odds/              # Odds APIs and scrapers
│   ├── news/              # News aggregation
│   ├── collectors/        # Multi-source collectors
│   ├── score_network/     # ScoreNetworkData (8 sports)
│   └── sports/            # Integrated sports data
│
├── models/                # Trained models
│   ├── score_network/     # 8 disciplines × 2 models
│   └── trained/           # Football-Data models
│
├── scripts/               # Utility scripts
│   ├── full_production_pipeline.py
│   ├── train_score_network_models.py
│   ├── organize_and_train_score_data.py
│   └── integrate_score_network_to_nexus.py
│
├── frontend/app/          # React frontend
├── tests/                 # Unit and integration tests
├── main.py               # Main entry point
├── pyproject.toml        # Project configuration
├── docker-compose.yml    # Docker orchestration
└── README.md             # This file
```

---

## 📊 ScoreNetworkData Integration

### Data Processing Pipeline:
```
D:\ScoreNetworkData (300 files, 3.5GB)
    ↓
Organize & Segregate (8 disciplines)
    ↓
Data Augmentation (2x increase)
    ↓
Train/Test Split (80/20)
    ↓
Train Models (RF + MLP per discipline)
    ↓
models/score_network/
```

### Disciplines:
- **Tennis**: 1M samples (500K original + 500K augmented)
- **Basketball**: 267K samples
- **American Football**: 318K samples
- **MMA**: 203K samples
- **Baseball**: 106K samples
- **Hockey**: 103K samples
- **Olympics**: 83K samples
- **Soccer**: 41K samples

---

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

---

## 📈 Monitoring

- **Prometheus Metrics**: http://localhost:8000/metrics
- **Grafana Dashboard**: http://localhost:3030
- **API Health**: http://localhost:8000/api/status

---

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d
```

---

## 📝 License

MIT

---

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss changes.

**Active Contributors:**
- szymonMCS - Main developer

---

## ⚠️ Disclaimer

This software is for educational purposes only. Sports betting involves risk.
Please gamble responsibly and within your means.

---

## 📚 Additional Documentation

- `BETTING_SPORTS_ANALYSIS.md` - Analysis of betting sports
- `CUTTING_EDGE_DEPLOYMENT.md` - Deployment guide
- `IMPLEMENTATION_REPORT.md` - Implementation details
- `ML_RESEARCH_IMPLEMENTATION.md` - ML research summary
- `PRODUCTION_DEPLOYMENT_REPORT.md` - Production deployment report
- `ROADMAP.md` - Project roadmap
- `SCORE_NETWORK_DATA_REPORT.md` - ScoreNetworkData integration report
- `TRAINING_REPORT.md` - Model training report
