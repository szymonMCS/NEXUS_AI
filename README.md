# 🎯 NEXUS AI v2.0

Advanced Sports Prediction System powered by LangGraph, MCP, and Claude AI.

## 🌟 Wybierz Swój Tryb

NEXUS AI oferuje **dwa tryby** działania:

### 🔹 **Lite Mode** (Domyślny - $0-50/mies)
**Idealne do: development, testów, użytku osobistego**

- ✅ **On-demand CLI** - uruchom gdy potrzebujesz
- ✅ **Darmowe źródła danych** - web scraping + free APIs
- ✅ **Zero kosztów infrastruktury** - działa lokalnie
- ✅ **WebDataEvaluator** - inteligentna walidacja danych z internetu
- ✅ **Proste w użyciu** - jeden plik, jedna komenda

**Źródła danych Lite:**
- TheSportsDB (darmowe API)
- Sofascore (scraping)
- Flashscore (scraping)
- Fortuna/STS/Betclic (scraping)
- Brave Search + Serper (darmowe limity)

### 🔸 **Pro Mode** ($150-200/mies)
**Idealne do: produkcji, ciągłego monitoringu, biznesu**

- ✅ **Background service** - działa 24/7
- ✅ **Płatne API** - The Odds API, API-Tennis, BetsAPI
- ✅ **MCP Servers** - skalowalna architektura
- ✅ **LangGraph Agents** - zaawansowana orkiestracja
- ✅ **Live tracking** - monitoring kursów w czasie rzeczywistym
- ✅ **PostgreSQL + Redis** - profesjonalna baza danych

## 🚀 Quick Start (Lite Mode)

### 1. Instalacja

```bash
git clone <your-repo-url>
cd nexus
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium  # Dla web scrapingu
```

### 2. Konfiguracja

```bash
cp .env.example .env
# Edytuj .env i dodaj klucze API:
# - BRAVE_API_KEY (darmowe 2000 req/mies)
# - SERPER_API_KEY (darmowe 2500 req/mies)
# - ANTHROPIC_API_KEY (dla Claude)
```

### 3. Uruchomienie

```bash
# Wygeneruj raport dziennych betów dla tenisa
python nexus.py --sport tennis --date today

# Dla koszykówki
python nexus.py --sport basketball --date 2026-01-20

# Zobacz wszystkie opcje
python nexus.py --help
```

### 4. Rezultat

System wygeneruje raport w `outputs/raport_2026-01-19_tennis.md` z:
- ✅ Top 3-5 najlepszych betów
- ✅ Analiza jakości danych dla każdego meczu
- ✅ Prawdopodobieństwa i value
- ✅ Rekomendowane stawki (Kelly Criterion)
- ✅ Podsumowanie newsów i kontuzji

## 📊 Architektura

```
┌─────────────────────────────────────────┐
│         CLI Interface (Lite)            │
│      lub Gradio UI (Pro opcjonalnie)    │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│     Fixture Collector                   │
│  (TheSportsDB, Sofascore, Flashscore)   │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│       Data Enricher (parallel)          │
│  News, Stats, H2H, Odds, Rankings       │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│    WebDataEvaluator (🔑 KLUCZOWY!)     │
│  Cross-validation, Freshness, Quality   │
└───────────────┬─────────────────────────┘
                │
         [Filter: Quality > 40%]
                │
┌───────────────▼─────────────────────────┐
│      Prediction Engine                  │
│   Tennis/Basketball Models + Value      │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│   Match Ranker → Select Top 3-5         │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      Report Generator                   │
│      outputs/raport_*.md                │
└─────────────────────────────────────────┘
```

## 🔄 Przełączanie między Lite a Pro

W pliku `.env` ustaw:

```bash
# Lite Mode (domyślny)
APP_MODE=lite
USE_WEB_SCRAPING=True
USE_FREE_APIS=True

# Pro Mode
APP_MODE=pro
# Dodaj klucze do płatnych API w .env
```

## 📋 Porównanie Trybów

| Aspekt | Lite | Pro |
|--------|------|-----|
| **Koszt/miesiąc** | $0-50 | $150-200 |
| **Tryb działania** | On-demand CLI | Background 24/7 |
| **Źródła danych** | Scraping + Free APIs | Płatne APIs |
| **Jakość danych** | Dobra (z validacją) | Bardzo dobra |
| **Live tracking** | ❌ | ✅ |
| **Deployment** | Lokalnie | Docker + VPS |
| **Baza danych** | Brak (cache w pamięci) | PostgreSQL + Redis |
| **MCP Servers** | ❌ | ✅ |
| **LangGraph** | ❌ | ✅ |
| **Idealne dla** | Dev, testy, hobby | Produkcja, biznes |

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

## 🛠️ Development Roadmap

### ✅ Phase 1: Lite Mode (Obecne)
- [x] Konfiguracja hybrydowa (Lite/Pro)
- [ ] Web scrapers (Sofascore, Flashscore, PL bookies)
- [ ] WebDataEvaluator
- [ ] Tennis/Basketball models
- [ ] Report generator
- [ ] CLI interface

### 🔜 Phase 2: Pro Mode (Opcjonalne)
- [ ] MCP Servers
- [ ] LangGraph Agents
- [ ] PostgreSQL + Redis
- [ ] Background scheduler
- [ ] Live odds tracking
- [ ] Gradio advanced UI

## 📚 Dokumentacja

Szczegółowa dokumentacja dostępna w katalogu `plans/`:
- `plans/lite/` - Specyfikacja Lite Mode
- `plans/` - Specyfikacja Pro Mode

## 💡 Wsparcie

Masz pytania? Otwórz [issue](https://github.com/your-repo/issues)!
