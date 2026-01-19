# 1. ARCHITEKTURA SYSTEMU

### 1.1 Diagram Przepływu Danych

```mermaid
flowchart TD
    A[Start: Sprawdź mecze na dziś] --> B[Pobierz newsy Brave/Serper/NewsAPI]
    B --> C[News Aggregator - deduplikacja]
    C --> D[News Validator - quality score]
    D --> E{Data Evaluator Agent}
    E -->|Quality > 60%| F[Pobierz kursy z API + Scraping]
    E -->|Quality 40-60%| G[Oznacz jako HIGH RISK]
    E -->|Quality < 40%| H[Odrzuć mecz - INSUFFICIENT_DATA]
    F --> I[Analyst Agent - oblicz prawdopodobieństwa]
    G --> I
    I --> J[Value Calculator - znajdź edge]
    J --> K[Match Ranker - sortuj po composite score]
    K --> L[Wybierz Top 3 mecze]
    L --> M[Decision Agent - final verification]
    M --> N[Execute bets z Kelly Criterion]
    N --> O[Update bankroll + Send alerts]
    H --> P[Log to database: skipped_matches]

    subgraph "MCP Servers Layer"
        B
        F
    end

    subgraph "Quality Control"
        D
        E
    end

    subgraph "Analysis Engine"
        I
        J
        K
    end
```

### 1.2 Architektura Komponentów

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GRADIO INTERFACE                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Dashboard    │ │ Top 3 Tab   │ │ Live News   │ │ History     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH ORCHESTRATOR                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Supervisor  │◀─│ NewsAnalyst │──│ DataEvaluator│──│ Analyst    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                                                   │                    │
│         ▼                                                   ▼                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ MatchRanker │──│ RiskManager │──│DecisionMaker│──│ BettorAgent │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVERS LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ news_server │  │evaluation_  │  │ odds_server │  │ tennis_     │             │
│  │ (Brave+     │  │server       │  │ (TheOddsAPI │  │ server      │             │
│  │  Serper)    │  │             │  │  + Scraping)│  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                              │
│  │ basketball_ │  │ alerts_     │  │ pl_bookies_ │                              │
│  │ server      │  │ server      │  │ scraper     │                              │
│  └─────────────┘  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                        │
│  [Brave Search] [Serper] [NewsAPI] [The Odds API] [api-tennis] [BetsAPI]        │
│  [Sofascore Scraping] [Fortuna Scraping] [STS Scraping] [Betclic Scraping]      │
└─────────────────────────────────────────────────────────────────────────────────┘
```
