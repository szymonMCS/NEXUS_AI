# 2. STRUKTURA PROJEKTU

```
nexus-ai/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── app.py                              # Gradio entry point
├── betting_floor.py                    # Główna pętla systemu
│
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Wszystkie ustawienia + API keys
│   ├── thresholds.py                   # Progi jakości danych
│   └── leagues.py                      # Konfiguracja lig (popular/medium/unpopular)
│
├── data/
│   ├── __init__.py
│   ├── sports_api.py                   # Bazowy klient API
│   │
│   ├── news/
│   │   ├── __init__.py
│   │   ├── aggregator.py               # Agregacja newsów z wielu źródeł
│   │   ├── validator.py                # Walidacja wiarygodności źródeł
│   │   └── injury_extractor.py         # Ekstrakcja kontuzji przez LLM
│   │
│   ├── tennis/
│   │   ├── __init__.py
│   │   ├── api_tennis_client.py        # api-tennis.com
│   │   └── sofascore_scraper.py        # Fallback scraper
│   │
│   ├── basketball/
│   │   ├── __init__.py
│   │   ├── bets_api_client.py          # BetsAPI
│   │   └── euroleague_scraper.py       # Scraper dla EuroLeague
│   │
│   ├── odds/
│   │   ├── __init__.py
│   │   ├── odds_api_client.py          # The Odds API
│   │   ├── pl_scraper.py               # Fortuna/STS/Betclic scraper
│   │   └── odds_merger.py              # Łączenie kursów z wielu źródeł
│   │
│   └── quality/
│       ├── __init__.py
│       └── metrics.py                  # DataQualityMetrics class
│
├── database/
│   ├── __init__.py
│   ├── models.py                       # SQLAlchemy models
│   ├── crud.py                         # CRUD operations
│   └── db.py                           # Database connection + init
│
├── mcp_servers/
│   ├── __init__.py
│   ├── news_server.py                  # Brave + Serper + NewsAPI
│   ├── evaluation_server.py            # Agent ewaluujący
│   ├── odds_server.py                  # Kursy (API + scraping)
│   ├── tennis_server.py                # Dane tenisowe
│   ├── basketball_server.py            # Dane koszykarskie
│   └── alerts_server.py                # Powiadomienia
│
├── agents/
│   ├── __init__.py
│   ├── supervisor.py                   # Koordynator LangGraph
│   ├── news_analyst.py                 # Analiza newsów
│   ├── data_evaluator.py               # Ewaluacja jakości danych
│   ├── analyst.py                      # Analiza meczów
│   ├── ranker.py                       # Ranking Top 3
│   ├── risk_manager.py                 # Ocena ryzyka
│   ├── decision_maker.py               # Decyzje końcowe
│   └── bettor.py                       # Wykonywanie zakładów
│
├── core/
│   ├── __init__.py
│   ├── state.py                        # Pydantic state models
│   ├── quality_scorer.py               # Quality score 0-1
│   ├── value_calculator.py             # Value + Kelly Criterion
│   ├── ranker_engine.py                # Logika rankingu
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tennis_model.py             # Model predykcji tenisa
│   │   └── basketball_model.py         # Model predykcji koszykówki
│   │
│   └── utils/
│       ├── __init__.py
│       ├── rate_limiter.py             # Rate limiting dla API
│       └── cache.py                    # Redis cache wrapper
│
├── ui/
│   ├── __init__.py
│   ├── dashboard.py                    # Główny dashboard
│   ├── top3_tab.py                     # Zakładka Top 3
│   ├── news_feed.py                    # Live news feed
│   ├── history_tab.py                  # Historia zakładów
│   └── components.py                   # Reusable Gradio components
│
├── tests/
│   ├── __init__.py
│   ├── test_news_aggregator.py
│   ├── test_data_evaluator.py
│   ├── test_value_calculator.py
│   └── test_ranker.py
│
├── scripts/
│   ├── setup_mcp.py
│   ├── init_db.py
│   ├── backtest.py
│   └── run_daily.py
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── .env.docker
```
