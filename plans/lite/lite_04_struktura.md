## 4. STRUKTURA PROJEKTU

```
nexus-ai-lite/
│
├── nexus.py                          # 🚀 Główny entry point CLI
├── requirements.txt
├── .env.example
│
├── config/
│   ├── __init__.py
│   ├── settings.py                   # Konfiguracja główna
│   ├── free_apis.py                  # Konfiguracja darmowych API
│   └── leagues.py                    # Klasyfikacja lig
│
├── data/
│   ├── __init__.py
│   │
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── fixture_collector.py      # Zbiera fixtures z wielu źródeł
│   │   └── data_enricher.py          # Wzbogaca dane o statystyki/news
│   │
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── sofascore_scraper.py      # Sofascore (stats, H2H)
│   │   ├── flashscore_scraper.py     # Flashscore (fixtures, odds)
│   │   └── pl_bookies_scraper.py     # Fortuna/STS/Betclic
│   │
│   ├── apis/
│   │   ├── __init__.py
│   │   ├── thesportsdb_client.py     # TheSportsDB (darmowe)
│   │   └── api_sports_client.py      # API-Sports free tier
│   │
│   └── news/
│       ├── __init__.py
│       ├── web_search.py             # Brave + Serper
│       └── injury_extractor.py       # Ekstrakcja kontuzji z newsów
│
├── evaluator/
│   ├── __init__.py
│   ├── web_data_evaluator.py         # 🔑 Ewaluator jakości danych web
│   ├── source_agreement.py           # Sprawdza zgodność źródeł
│   └── freshness_checker.py          # Sprawdza świeżość danych
│
├── prediction/
│   ├── __init__.py
│   ├── tennis_model.py               # Model predykcji tenisa
│   ├── basketball_model.py           # Model predykcji koszykówki
│   └── value_calculator.py           # Obliczanie value vs kursy
│
├── ranking/
│   ├── __init__.py
│   └── match_ranker.py               # Ranking i selekcja Top betów
│
├── reports/
│   ├── __init__.py
│   ├── report_generator.py           # Generator raportów
│   └── templates/
│       ├── report_template.md
│       └── report_template.html
│
├── ui/
│   ├── __init__.py
│   └── gradio_app.py                 # Opcjonalny interfejs Gradio
│
└── outputs/                          # Wygenerowane raporty
    └── .gitkeep
```
