# NEXUS AI - Plan Rozwoju v3.0

## Spis Treści
1. [Aktualny Stan Projektu](#1-aktualny-stan-projektu)
2. [Plan C+: Hybrid Implementation](#2-plan-c-hybrid-implementation)
3. [Następne Kroki](#3-nastepne-kroki)
4. [Timeline](#4-timeline)

---

## 1. Aktualny Stan Projektu

### **Ocena: 8.5/10 (Bardzo Dobry)**

### Co Jest Zrobione (FAZA 0-5 COMPLETE)

| Komponent | Status | Pliki |
|-----------|--------|-------|
| **Data Schemas** | ✅ Done | `core/data/schemas.py`, `core/data/enums.py` |
| **Feature Pipeline** | ✅ Done | `core/ml/features/` (goals, handicap, form extractors) |
| **ML Models** | ✅ Done | `core/ml/models/` (GoalsModel Poisson, HandicapModel GBM) |
| **Model Registry** | ✅ Done | `core/ml/registry/` (versioning, rollback) |
| **Online Training** | ✅ Done | `core/ml/training/` (incremental learning) |
| **Accuracy Tracking** | ✅ Done | `core/ml/tracking/` (ROI, accuracy) |
| **ML Prediction Service** | ✅ Done | `core/ml/service/` (API integration) |
| **API Clients (Free)** | ✅ Done | 9 APIs working (Odds, Football-Data, API-Sports, etc.) |
| **API Clients (Premium)** | ✅ Ready | 6 premium APIs (Sportradar, SportsDataIO, etc.) - activate with keys |
| **API Tier Manager** | ✅ Done | Auto-fallback between API tiers |
| **Historical Collector** | ✅ Done | `data/collectors/historical_collector.py` |
| **Collection Scripts** | ✅ Done | `scripts/collect_historical.py`, `scripts/train_initial_models.py` |

### Co Działa Teraz

```bash
# Zbieranie danych historycznych
python scripts/collect_historical.py --sport football --leagues PL --days 30
# Wynik: 52 mecze zebrane w 5.9s

# Sprawdzenie API
python scripts/test_api_tiers.py
# Wynik: 7 APIs available, all free tier working
```

### Czego Brakuje

| Komponent | Priorytet | Opis |
|-----------|-----------|------|
| **Więcej danych treningowych** | 🔴 Wysoki | Min. 1000 meczów do sensownego treningu |
| **Kimi/LLM Integration** | 🔴 Wysoki | Reasoning, injury extraction, news analysis |
| **Przetrenowane modele** | 🔴 Wysoki | Modele na prawdziwych danych |
| **Frontend ↔ Backend ML** | 🟡 Średni | Połączenie React z ML API |
| **Live Betting** | 🟢 Niski | Wymaga szybszych API |

---

## 2. Plan C+: Hybrid Implementation

**Cel**: Lokalne modele ML + Kimi do reasoning = najlepsza jakość przy minimalnych kosztach.

### Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                    NEXUS Hybrid System                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Data APIs  │    │  Historical  │    │    News      │  │
│  │  (9 working) │    │  Collector   │    │   Scraper    │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Feature Pipeline                         │  │
│  │   (Goals, Handicap, Form extractors + normalization) │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│         ┌───────────────────┼───────────────────┐          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Goals Model │    │  Handicap   │    │  Kimi K2.5  │    │
│  │  (Poisson)  │    │   Model     │    │  + Thinking │    │
│  │   LOCAL     │    │   (GBM)     │    │ Agent Swarm │    │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │
│         │                  │                   │           │
│         └──────────────────┼───────────────────┘           │
│                            ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Ensemble Integration                        │  │
│  │   (Weighted average: 60% ML + 40% Kimi reasoning)    │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Final Prediction + Recommendation        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Koszty Miesięczne

| Komponent | Koszt |
|-----------|-------|
| API-Sports (free tier) | $0 |
| Football-Data.org (free) | $0 |
| Odds API (free tier) | $0 |
| Kimi API (~100 req/day) | ~$5-10 |
| **RAZEM** | **$5-10/mies** |

---

## 3. Następne Kroki

### Faza 6: Data Collection & Training

| ID | Zadanie | Czas | Status |
|----|---------|------|--------|
| 6.1 | Zebrać 365 dni danych (5 lig) | 1h | [ ] |
| 6.2 | Przetrenować GoalsModel | 30min | [ ] |
| 6.3 | Przetrenować HandicapModel | 30min | [ ] |
| 6.4 | Zapisać modele do registry | 15min | [ ] |

### Faza 7: Kimi K2.5 Integration ✅ COMPLETE

| ID | Zadanie | Czas | Status |
|----|---------|------|--------|
| 7.1 | Utworzyć `core/llm/kimi_client.py` (K2.5 + Agent Swarm) | 1h | [x] |
| 7.2 | Utworzyć `core/llm/injury_extractor.py` | 2h | [x] |
| 7.3 | Utworzyć `core/llm/match_analyzer.py` | 2h | [x] |
| 7.4 | Utworzyć `core/llm/hybrid_predictor.py` | 2h | [x] |
| 7.5 | Testy integracyjne (27 testów passed) | 1h | [x] |

**Kimi K2.5 Features Implemented:**
- `kimi-k2.5-preview` - latest multimodal agentic model
- `kimi-k2-thinking` - deep reasoning with Chain-of-Thought
- **Agent Swarm** - complex task decomposition into parallel sub-tasks
- Thinking mode with `reasoning_content` traces
- OpenAI-compatible API at `https://api.moonshot.ai/v1`

### Faza 8: Frontend Integration

| ID | Zadanie | Czas | Status |
|----|---------|------|--------|
| 8.1 | API endpoint dla predictions | 1h | [ ] |
| 8.2 | WebSocket dla live updates | 2h | [ ] |
| 8.3 | Dashboard z predykcjami | 3h | [ ] |

---

## 4. Timeline

```
Tydzień 1 (Teraz):
├── Dzień 1-2: Zbieranie danych + trening modeli
├── Dzień 3-4: Kimi integration (client, extractors)
└── Dzień 5: Hybrid predictor + testy

Tydzień 2:
├── Dzień 1-2: Frontend integration
├── Dzień 3-4: End-to-end testing
└── Dzień 5: Production deployment

Po 2 tygodniach: System gotowy do produkcji
```

---

## Pliki do Utworzenia

### Nowe (Faza 7):
```
core/llm/
├── __init__.py
├── kimi_client.py         # Kimi API client
├── injury_extractor.py    # Extract injuries from news
├── match_analyzer.py      # Deep match analysis
└── hybrid_predictor.py    # ML + Kimi ensemble
```

### Do Edycji:
```
config/settings.py         # MOONSHOT_API_KEY (done!)
.env                       # MOONSHOT_API_KEY= (get from platform.moonshot.ai)
requirements.txt           # httpx already included
```

---

## Success Metrics

| Metryka | Target | Deadline |
|---------|--------|----------|
| Dane treningowe | >1000 meczów | Tydzień 1 |
| Goals Model accuracy | >60% | Tydzień 1 |
| Handicap Model accuracy | >55% | Tydzień 1 |
| Kimi K2.5 integration | ✅ Working | Done |
| End-to-end prediction | <3s | Tydzień 2 |
| Monthly cost | <$15 | Ongoing |

---

## Environment Variables

```bash
# .env - dodaj te klucze:

# Moonshot Kimi K2.5 API (https://platform.moonshot.ai)
# Get your key at: https://platform.moonshot.ai/console/api-keys
MOONSHOT_API_KEY=your_moonshot_api_key_here
KIMI_MODEL=kimi-k2.5-preview  # or kimi-k2-thinking for deep reasoning

# Available models:
# - kimi-k2.5-preview     (latest, multimodal, agentic)
# - kimi-k2-thinking      (deep reasoning with CoT)
# - kimi-k2-0905-preview  (September 2025)
# - moonshot-v1-8k/32k/128k (legacy)

# Opcjonalnie - jeśli chcesz używać OpenAI jako fallback
# OPENAI_API_KEY=already_configured
```

---

**Ostatnia aktualizacja**: 2026-01-27
**Wersja**: 3.1 (po FAZA 7 - Kimi K2.5 Integration Complete)
