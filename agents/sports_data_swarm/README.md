# Sports Data Swarm v2.0

System agentów do pozyskiwania danych sportowych z internetu za pomocą web scrapingu i API. Tworzy zestawy danych do treningu sztucznej inteligencji z możliwością augmentacji danych.

## Nowości w wersji 2.0

✨ **Football/Soccer** - Nowa dyscyplina z danymi xG (expected goals)  
✨ **Data Augmentation** - Powielanie danych 2-5x dla lepszego treningu modeli  
✨ **Advanced Feature Engineering** - Automatyczne generowanie cech ML  
✨ **Więcej źródeł dla Tenisa** - 9 stron do scrapowania  

## Architektura Agentów

```
┌─────────────────────────────────────────────────────────────────┐
│                      MANAGER AGENT                              │
│                 (Koordynator całego procesu)                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┬──────────────┐
        │              │              │              │              │
   ┌────▼─────┐   ┌───▼────┐    ┌────▼─────┐   ┌───▼────┐   ┌───▼────┐
   │Basketball│   │Volley- │    │ Handball │   │ Tennis │   │Football│
   │  Agent   │   │ ball   │    │  Agent   │   │ Agent  │   │ Agent  │
   └────┬─────┘   └───┬────┘    └────┬─────┘   └───┬────┘   └───┬────┘
        │             │              │             │            │
        └─────────────┴──────────────┴─────────────┴────────────┘
                                    │
                             ┌──────▼──────┐
                             │  DATA ACQ   │
                             │   AGENT     │
                             │(Web Scraping│
                             │  & API)     │
                             └──────┬──────┘
                                    │
                             ┌──────▼──────┐
                             │  FORMATTING │
                             │    AGENT    │
                             └──────┬──────┘
                                    │
                             ┌──────▼──────┐
                             │   STORAGE   │
                             │    AGENT    │
                             └──────┬──────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
               ┌──────▼──────┐            ┌──────▼──────┐
               │   EVALUATOR │            │AUGMENTATION │
               │    AGENTS   │            │    AGENT    │
               │  (5 sportów)│            │(2-5x data)  │
               └─────────────┘            └─────────────┘
```

## Struktura Plików

```
agents/sports_data_swarm/
├── __init__.py                    # Pakiet
├── base_agent.py                  # Klasa bazowa
├── manager_agent.py               # Koordynator
├── sport_agents.py                # 4 dyscypliny sportowe
├── football_agent.py              # NOWOŚĆ: Piłka nożna
├── data_acquisition_agent.py      # Web scraping + API
├── formatting_agent.py            # Normalizacja danych
├── storage_agent.py               # Zapis do plików
├── evaluator_agents.py            # Ewaluacja (4 sporty)
├── football_evaluator.py          # NOWOŚĆ: Ewaluacja piłki
├── data_augmentation_agent.py     # NOWOŚĆ: Augmentacja danych
├── run_collection.py              # Główny skrypt
├── test_swarm.py                 # Testy
├── demo.py                       # Demonstracja
└── README.md                     # Dokumentacja
```

## Wymagania

```bash
pip install aiohttp beautifulsoup4 python-dotenv pandas
```

## Konfiguracja API

Utwórz lub edytuj plik `.env` w katalogu głównym:

```env
BRAVE_API_KEY=your_brave_api_key
SERPER_API_KEY=your_serper_api_key
```

## Użycie

### Szybki start

```bash
# Wszystkie sporty
python run_collection.py --all --target 10000

# Tylko piłka nożna z xG
python run_collection.py --sport football --target 5000

# Z augmentacją danych (3x więcej danych)
python run_collection.py --sport basketball --target 1000 --augment 3.0

# Wiele sportów z augmentacją
python run_collection.py --sports football tennis --target 2000 --augment 2.5
```

### Opcje

| Parametr | Opis | Przykład |
|----------|------|----------|
| `--sport` | Pojedynczy sport | `--sport football` |
| `--sports` | Wiele sportów | `--sports football tennis` |
| `--all` | Wszystkie 5 sportów | `--all` |
| `--target` | Liczba rekordów | `--target 10000` |
| `--augment` | Mnożnik augmentacji | `--augment 3.0` |
| `--format` | Format wyjściowy | `--format csv/json/parquet` |
| `--start-date` | Data początkowa | `--start-date 2020-01-01` |
| `--end-date` | Data końcowa | `--end-date 2024-12-31` |

## Techniki Augmentacji Danych

System wykorzystuje 5 technik augmentacji dla danych tabelarycznych:

### 1. Gaussian Noise Injection
Dodaje losowy szum (2%) do wartości numerycznych, np.:
- Punkty: 100 → 100.4
- Procenty: 0.45 → 0.452

### 2. Synthetic Sample Generation
Tworzy syntetyczne rekordy poprzez interpolację między podobnymi meczami:
```
Mecz A: home_score=100, away_score=90
Mecz B: home_score=110, away_score=85
Syntetyczny: home_score=105, away_score=87.5
```

### 3. Feature Engineering
Automatycznie generuje nowe cechy:
- **Koszykówka**: TS% (True Shooting), eFG% (Effective FG%), AST/TO ratio
- **Piłka nożna**: Shot accuracy, Conversion rate, xG performance diff
- **Tennis**: Serve efficiency, Break point conversion, Aggression index
- **Siatkówka**: Attack efficiency, Points per set
- **Piłka ręczna**: Shot efficiency, 7m conversion, GK efficiency

### 4. Rolling Averages
Dodaje wskaźniki formy (symulowane):
- `home_form_goals_avg` - średnia goli w ostatnich 5 meczach
- `player1_form_sets_won_avg` - średnia wygranych setów

### 5. Interaction Features
Łączy cechy dla lepszych predykcji:
- `home_fg_x_rebounds` = FG% × Rebounds
- `home_poss_x_shots` = Possession × Shots
- `player1_risk_indicator` = Aces - Double Faults

## Zebrane Dane

### Koszykówka (Basketball)
- **Ligi**: NBA, EuroLeague, EuroCup, ACB, Legabasket, BBL, LNH
- **Pola**: wynik, kwarty, FG%, 3P%, FT%, zbiórki, asysty, przechwyty, bloki
- **Cechy ML**: point_diff, total_points, home_win, ts_pct, efg_pct, ast_to_ratio

### Piłka Nożna (Football) 🆕
- **Ligi**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League
- **Pola**: gole, połowa, strzały, na celu, rożne, faule, kartki, posiadanie
- **Dane xG**: Expected goals (Understat, FBref)
- **Cechy ML**: shot_accuracy, conversion_rate, xg_performance_diff, dominance_index

### Siatkówka (Volleyball)
- **Ligi**: SuperLega, PlusLiga, Russian Super League, Turkish Efeler Ligi
- **Pola**: sety, punkty, ataki, bloki, asy, przyjęcie
- **Cechy ML**: sets_diff, attack_efficiency, points_per_set

### Piłka Ręczna (Handball)
- **Ligi**: EHF Champions League, Bundesliga, Liga ASOBAL, LNH
- **Pola**: gole, rzuty, obrony, strata, karny, szybkie ataki
- **Cechy ML**: goal_diff, shot_efficiency, 7m_conversion, gk_efficiency

### Tenis (Tennis)
- **Turnieje**: Grand Slam, ATP Masters 1000, ATP/WTA Tours
- **Pola**: sety, gemy, asy, podwójne błędy, % serwisu, break pointy
- **Cechy ML**: sets_diff, serve_efficiency, bp_conversion, aggression

## Dane Wyjściowe

### Struktura plików
```
datasets/sports_data/processed/
├── basketball_dataset_YYYYMMDD_HHMMSS.csv
├── basketball_dataset_YYYYMMDD_HHMMSS_augmented_3x.csv  # Augmentowane
├── football_dataset_YYYYMMDD_HHMMSS.csv                 # NOWOŚĆ
├── football_dataset_YYYYMMDD_HHMMSS_augmented_2.5x.csv  # NOWOŚĆ
├── tennis_dataset_YYYYMMDD_HHMMSS.csv
└── ...
```

### Przykładowe cechy ML (po augmentacji)
```json
{
  "game_id": "nba_001",
  "home_score": 102.4,
  "away_score": 98.2,
  "home_fg_pct": 0.452,
  "home_ts_pct": 0.568,
  "home_efg_pct": 0.512,
  "home_ast_to_ratio": 1.85,
  "home_fg_x_rebounds": 18.5,
  "home_form_goals_avg": 104.2,
  "point_diff": 4.2,
  "home_win": 1,
  "augmented": true,
  "augmentation_type": "noise"
}
```

## Jak To Działa

1. **Manager Agent** inicjalizuje wszystkich agentów
2. **Sport Agent** tworzy strategię kolekcji
3. **Data Acquisition Agent**:
   - Wyszukuje dane przez Brave Search API
   - Wyszukuje dane przez Serper API
   - Scrapuje 7-9 stron dla każdego sportu
4. **Formatting Agent** normalizuje dane
5. **Storage Agent** zapisuje dane + podział train/test
6. **Evaluator Agent** ocenia jakość
7. **Augmentation Agent** (opcjonalnie):
   - Dodaje szum Gaussowski
   - Generuje syntetyczne próbki
   - Inżynieruje cechy
   - Tworzy interakcje

## Ograniczenia

- **API Rate Limits**: Brave (2000/miesiąc), Serper (2500/miesiąc)
- **Web Scraping**: Niektóre strony blokują (403 Forbidden)
- **Wymagane API Keys**: Bez nich tylko web scraping (mniej danych)

## Porady dla Lepszego Pozyskiwania Danych

### 1. Zwiększ limit rekordów
```bash
python run_collection.py --all --target 20000
```

### 2. Użyj augmentacji
```bash
# 3x więcej danych = lepszy trening modelu
python run_collection.py --sport football --target 5000 --augment 3.0
```

### 3. Zbieraj dane etapami
```bash
# Etap 1: Podstawowe dane
python run_collection.py --sport basketball --target 5000

# Etap 2: Wzbogać o augmentację
python run_collection.py --sport basketball --target 5000 --augment 2.0
```

### 4. Użyj formatu Parquet dla dużych zbiorów
```bash
python run_collection.py --all --target 20000 --format parquet --augment 2.0
```

## Rozszerzenia

Aby dodać nowy sport:
1. Stwórz `new_sport_agent.py` dziedziczący po `BaseAgent`
2. Zdefiniuj `required_fields` i `optional_fields`
3. Dodaj ewaluator w `new_sport_evaluator.py`
4. Zarejestruj w `run_collection.py`

## Licencja

System stworzony jako część projektu NEXUS AI v2.0
