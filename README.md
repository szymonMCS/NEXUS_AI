# 🎯 NEXUS AI v2.0

Advanced Sports Prediction System powered by LangGraph, MCP, and Claude AI.

## 🚀 Features

- **Multi-Sport Support**: Tennis & Basketball predictions
- **Intelligent Data Aggregation**: News from multiple sources (Brave, Serper, NewsAPI)
- **Quality-Based Filtering**: Automatic data quality evaluation
- **Top 3 Ranking System**: Focus on highest value opportunities
- **MCP Server Architecture**: Modular, scalable design
- **Real-time Odds Comparison**: Multiple bookmakers (API + optional scraping)
- **Risk Management**: Kelly Criterion position sizing
- **LangGraph Orchestration**: Multi-agent workflow
- **Beautiful UI**: Gradio dashboard with live updates

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

### Run the Gradio UI:
```bash
python app.py
```

### Run the betting floor (headless):
```bash
python betting_floor.py
```

### Run MCP servers:
```bash
python scripts/setup_mcp.py
```

## 📊 Architecture

```
NEXUS AI
├── Gradio UI (Dashboard, Top 3, News, History)
├── LangGraph Orchestrator (Multi-agent workflow)
│   ├── Supervisor Agent
│   ├── News Analyst Agent
│   ├── Data Evaluator Agent
│   ├── Analyst Agent
│   ├── Match Ranker Agent
│   ├── Risk Manager Agent
│   ├── Decision Maker Agent
│   └── Bettor Agent
├── MCP Servers (News, Odds, Tennis, Basketball, Alerts)
└── Data Sources (Configurable: Paid APIs or Free sources)
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

## 🧪 Testing

```bash
pytest tests/
```

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss changes.

## ⚠️ Disclaimer

This software is for educational purposes only. Sports betting involves risk.
Please gamble responsibly and within your means.
