## 1. ZAŁOŻENIA PROJEKTU

### 1.1 Czym jest NEXUS AI Lite?

System on-demand do generowania dziennych raportów z rekomendacjami zakładów sportowych.
**Uruchamiasz → System zbiera dane z internetu → Ewaluuje jakość → Generuje raport Top 3-5 betów → Koniec.**

### 1.2 Kluczowe Różnice vs Wersja Pro

| Aspekt | NEXUS AI Pro | NEXUS AI Lite |
|--------|--------------|---------------|
| Tryb działania | Ciągły (background) | On-demand |
| Źródła danych | Płatne API ($150+/mies) | Web scraping + darmowe API |
| Live tracking | ✅ | ❌ |
| Deployment | Docker + VPS | Lokalnie / jeden plik |
| Koszt | ~$200/mies | ~$0-50/mies |
| Złożoność | Wysoka | Niska-średnia |

### 1.3 Flow Działania

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NEXUS AI Lite - Flow                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [USER] ──► python nexus.py --sport tennis --date 2026-01-19       │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  1. FIXTURE COLLECTOR                                      │     │
│  │     - TheSportsDB (darmowe)                               │     │
│  │     - Sofascore scraping                                  │     │
│  │     - Flashscore scraping                                 │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  2. DATA ENRICHMENT (parallel)                            │     │
│  │     - Brave Search → newsy, kontuzje                      │     │
│  │     - Serper → dodatkowe źródła                           │     │
│  │     - Sofascore → statystyki, H2H                         │     │
│  │     - Flashscore → kursy                                  │     │
│  │     - PL bookies scraping → Fortuna/STS/Betclic          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  3. WEB DATA EVALUATOR (KLUCZOWY!)                        │     │
│  │     - Czy dane z internetu są spójne?                     │     │
│  │     - Czy mamy wystarczająco źródeł?                      │     │
│  │     - Czy informacje są świeże?                           │     │
│  │     - Cross-validation między źródłami                    │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│         [Odrzuć mecze z quality < 40%]                             │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  4. PREDICTION ENGINE                                     │     │
│  │     - Model tenisowy (ranking, forma, nawierzchnia)       │     │
│  │     - Model koszykarski (ratings, rest, home advantage)   │     │
│  │     - Value calculation vs kursy                          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  5. RANKING & REPORT                                      │     │
│  │     - Sortuj po: edge × quality × confidence              │     │
│  │     - Wybierz Top 3-5                                     │     │
│  │     - Wygeneruj raport MD/HTML                            │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  [OUTPUT] ──► raport_2026-01-19_tennis.md                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
