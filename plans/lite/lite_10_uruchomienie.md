## 10. URUCHOMIENIE I UŻYCIE

### 10.1 Instalacja

```bash
# 1. Sklonuj/utwórz projekt
mkdir nexus-ai-lite
cd nexus-ai-lite

# 2. Utwórz virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Skonfiguruj .env (opcjonalnie dla newsów)
cp .env.example .env
# Edytuj .env i dodaj klucze API (Brave, Serper)
```

### 10.2 Requirements.txt

```
# requirements.txt

# HTTP & Async
httpx>=0.24.0
aiohttp>=3.9.0

# Web Scraping
playwright>=1.40.0
beautifulsoup4>=4.12.0

# Data Processing
pandas>=2.0.0
pydantic>=2.0.0

# UI (opcjonalnie)
gradio>=4.0.0

# Utilities
python-dotenv>=1.0.0
tenacity>=8.2.0  # Retry logic

# Dev
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### 10.3 .env.example

```bash
# .env.example

# === NEWS APIs (opcjonalne - system działa też bez nich) ===
BRAVE_API_KEY=your_brave_key_here
SERPER_API_KEY=your_serper_key_here

# === LLM (opcjonalne - do ekstrakcji kontuzji) ===
ANTHROPIC_API_KEY=your_anthropic_key_here

# === API-Sports (opcjonalne) ===
API_SPORTS_KEY=your_api_sports_key_here
```

### 10.4 Przykłady Użycia

```bash
# Analiza tenisa na dziś
python nexus.py --sport tennis

# Analiza koszykówki na konkretny dzień
python nexus.py --sport basketball --date 2026-01-20

# Z wyższym progiem jakości
python nexus.py -s tennis -q 60

# Więcej betów w raporcie
python nexus.py -s tennis -n 7

# Tryb cichy (tylko raport)
python nexus.py -s tennis --quiet

# Uruchom interfejs Gradio
python ui/gradio_app.py
```

### 10.5 Przykładowy Output

```
╔══════════════════════════════════════════════════════════════╗
║  🎯 NEXUS AI Lite - Analiza On-Demand                        ║
╠══════════════════════════════════════════════════════════════╣
║  Sport: TENNIS      Data: 2026-01-19                         ║
╚══════════════════════════════════════════════════════════════╝

📅 [1/5] Zbieranie meczów z internetu...
  ✅ thesportsdb: 12 matches
  ✅ sofascore: 45 matches
  ✅ flashscore: 38 matches
   ✅ Znaleziono 52 meczów

🔍 [2/5] Wzbogacanie danych (newsy, statystyki, kursy)...
   ✅ Wzbogacono 52 meczów

📊 [3/5] Ewaluacja jakości danych z internetu...
   ⚠️ Qualifier A vs Qualifier B: quality 32% (SKIP)
   ⚠️ Unknown Player vs Unknown: quality 28% (SKIP)
   ✅ 34/52 meczów przeszło filtr jakości (>= 45%)

🧠 [4/5] Obliczanie predykcji i szukanie value...
   💰 Sinner J. vs Alcaraz C.: edge +4.2%
   💰 Sabalenka A. vs Swiatek I.: edge +3.8%
   ✅ Znaleziono 5 value betów

📝 [5/5] Generowanie raportu...
   ✅ Raport zapisany: outputs/raport_2026-01-19_tennis.md

============================================================
# 🎯 NEXUS AI - Raport Predykcji

**Sport:** TENNIS
**Data:** 2026-01-19
**Wygenerowano:** 2026-01-19 14:35

---

## 🏆 TOP 5 VALUE BETS

### 🥇 Sinner J. vs Alcaraz C.

**Liga:** Australian Open
**Typ:** HOME
**Kurs:** 2.15 @ Fortuna
**Edge:** +4.2%
**Jakość danych:** 78/100
**Stawka:** 1.5-2% bankroll

**Uzasadnienie:**
> HOME at 2.15 (prob: 52.3%, edge: 4.2%)

---
...
============================================================

✅ Gotowe! Raport: outputs/raport_2026-01-19_tennis.md
```

---

## PODSUMOWANIE

### Co zawiera NEXUS AI Lite:

| Komponent | Opis |
|-----------|------|
| **FixtureCollector** | Zbiera mecze z TheSportsDB, Sofascore, Flashscore |
| **DataEnricher** | Wzbogaca o kursy (PL bookies), newsy (Brave/Serper) |
| **WebDataEvaluator** | 🔑 Ewaluuje jakość danych web (agreement, freshness, completeness) |
| **TennisModel** | Predykcja na podstawie rankingu, formy, nawierzchni, H2H |
| **BasketballModel** | Predykcja na podstawie ratings, rest, home advantage |
| **ValueCalculator** | Oblicza edge i Kelly stake |
| **MatchRanker** | Composite score = edge × quality × confidence |
| **ReportGenerator** | Generuje raporty MD/HTML |

### Koszty:

| Serwis | Koszt | Użycie |
|--------|-------|--------|
| TheSportsDB | $0 | Fixtures (key=3) |
| API-Sports | $0 | 100 req/dzień/API |
| Sofascore | $0 | Scraping stats |
| Flashscore | $0 | Scraping odds |
| Brave Search | $0 | 2000 req/mies |
| Serper | $0 | 2500 req/mies |
| **RAZEM** | **~$0/mies** | |

### Następne Kroki:

1. Utwórz strukturę katalogów
2. Zaimplementuj scrapers (Sofascore, Flashscore, PL bookies)
3. Zaimplementuj WebDataEvaluator
4. Dodaj modele predykcji
5. Testuj na rzeczywistych danych
