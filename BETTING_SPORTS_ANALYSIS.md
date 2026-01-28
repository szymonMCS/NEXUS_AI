# Analiza Sportow pod katem Bettingu - Sports Data Swarm v3.0

## Top Sporty pod wzgledem Objetosci Zakladow (2024/2025)

Na podstawie danych z branzy bettingowej:

| Pozycja | Sport | Udzial w Rynku | Uwagi |
|---------|-------|---------------|-------|
| 1 | **Football/Soccer** | ~50% | Najwiekszy na swiecie, szczegolnie Europa/UK |
| 2 | **Basketball** | ~15% | NBA dominuje, rosny rynek w Azji |
| 3 | **Tennis** | ~12% | Grand Slamy, ATP, WTA - popularny w Europie |
| 4 | **American Football (NFL)** | ~10% | USA dominuje, Super Bowl najwiekszy event |
| 5 | **Baseball (MLB)** | ~5% | Japonia, USA, Korea - silne rynki |
| 6 | **Hockey (NHL)** | ~3% | Kanada, USA, Europa Polnocna |
| 7 | **MMA/UFC** | ~2% | Szybko rosnacy rynek, mlodzi gracze |
| 8 | **Esports** | ~1-2% | Najdynamiczniej rosnacy segment |
| 9 | **Golf** | ~1% | Turnieje wielkoszlemowe, nisza |
| 10 | **Rugby** | <1% | UK, Australia, NZ - lokalnie silny |
| 11 | **Cricket** | <1% | Indie, UK, Australia - bardzo silny lokalnie |
| 12 | **Table Tennis** | <0.5% | Azja, Europa - rosny |
| 13 | **Pilka Reczna** | <0.5% | Europa - nisza |

## Dlaczego te Sporty?

### Football (Soccer)
- Najwieksza globalna baza fanow
- Mecze codziennie na calym swiecie
- Wiele lig i turniejow
- **Mamy dane**: EPL, LaLiga z Score Network

### Basketball  
- Szybka akcja, duzo punktow
- Statystyki bardzo szczegolowe
- NBA, EuroLeague, ligi krajowe
- **Mamy dane**: NBA, WNBA z Score Network

### Tennis
- Indywidualny sport - latwiejsza analiza
- Duzo meczow w ciagu roku
- Rankingi ATP/WTA jako wskazniki
- **Mamy dane**: Australian Open, WTA z Score Network

### American Football (NFL)
- Dominacja w USA
- Duzo statystyk dla kazdej pozycji
- Przewidywalnosc przez analize formacji
- **Mamy dane**: NFL z Score Network

### Baseball (MLB)
- Bardzo statystyczny sport (sabermetrics)
- Duzo danych historycznych
- Wskazniki xG podobne do pilki noznej
- **Mamy dane**: MLB z Score Network

### MMA/UFC
- Duzo zmiennych (styl, formda, kontuzje)
- Szybko rosnaca popularnosc
- Dobre do modeli ML przez mniej zmiennych niz team sports

### Esports
- Najmlodszy rynek
- Dane cyfrowe - bardzo dokladne
- Rosnie eksponencjalnie
- **Mamy dane**: League of Legends z Score Network

## Struktura Systemu v3.0

```
agents/sports_data_swarm/
├── Sporty Podstawowe (5):
│   ├── basketball_agent.py      # NBA, EuroLeague
│   ├── football_agent.py        # Soccer/Pilka nozna
│   ├── tennis_agent.py          # ATP, WTA
│   ├── volleyball_agent.py      # Liga Mistrzow, PlusLiga
│   └── handball_agent.py        # EHF, Bundesliga
│
├── Sporty Bettingowe (8):
│   ├── baseball_agent.py        # MLB, NPB (NOWY)
│   ├── hockey_agent.py          # NHL, KHL (NOWY)
│   ├── mma_agent.py             # UFC, Bellator (NOWY)
│   ├── esports_agent.py         # LoL, CS2, Dota 2 (NOWY)
│   ├── golf_agent.py            # PGA, European Tour (NOWY)
│   ├── rugby_agent.py           # Six Nations, NRL (NOWY)
│   ├── cricket_agent.py         # IPL, Test, ODI (NOWY)
│   └── table_tennis_agent.py    # ITTF, WTT (NOWY)
│
└── Agenty Wsparcia:
    ├── data_acquisition_agent.py
    ├── formatting_agent.py
    ├── storage_agent.py
    ├── data_augmentation_agent.py  # Powielanie danych 2-5x
    └── *_evaluator_agent.py
```

## Dane ze Score Network (D:\ScoreNetworkData)

### Koszykowka
- `nba-player-stats-2021.csv` (812 rekordow)
- `wnba-shots-2022.csv.gz` (41,497 rzutow)
- `nba_2223_season_stints.csv` (3.9M rekordow!)
- `caitlin_clark_rookie_season.csv`

### Pilka Nozna
- `epl_player_stats_24_25.csv` (562 zawodnikow, 57 kolumn)
- `laliga_player_stats_english.csv` (556 zawodnikow)

### Tenis
- `AustralianOpen.csv` (1,905 meczow)
- `wta-grand-slam-matches-2018to2022.csv` (2,413 meczow)
- `tennis-m-shots-ao.csv.gz` (kazde uderzenie!)

### Baseball
- `mlb-standings-2024.csv`
- `pitchers.csv` (1.4M rekordow)

### Football Amerykanski
- `nfl-team-statistics.csv` (765 rekordow)
- `nfl-draft-combine.csv`

### Esports
- `lol_adc_sup_all-pairs.csv`
- `LOL_patch_*.csv`

### Siatkowka
- `volleyball_ncaa_div1_2022_23.csv` (334 druzyny)

### Pilka Reczna
- `handball_bundesliga_23.csv` (309 zawodnikow)

### MMA
- `UFC_stats.csv` (129KB)
- `mma_decisions.csv` (1.1MB)
- `mma_wtclass.csv` (5.3MB)

## Uzycie Systemu

### Pobierz wszystkie 13 sportow
```bash
cd agents/sports_data_swarm
python run_collection.py --all --target 5000 --augment 2.5
```

### Tylko top 5 dla bettingu
```bash
python run_collection.py --sports football basketball tennis baseball hockey --target 10000
```

### Dane wzbogacone o augmentacje (3x wiecej danych)
```bash
python run_collection.py --sport baseball --target 3000 --augment 3.0
```

### Sporty niszowe (rosnace rynki)
```bash
python run_collection.py --sports mma esports --target 2000
```

## Zalecenia dla Treningu Modeli

### 1. Model Ogolny (Multi-Sport)
- Uzyj wszystkich 13 sportow
- Augmentacja 2x
- Target: 50,000+ rekordow na sport

### 2. Model Football (Najwazniejszy)
- EPL + LaLiga ze Score Network (bazowe dane)
- + Web scraping pozostalych lig
- Augmentacja 3x
- Focus na: Goals, xG, Shots, Possession

### 3. Model Basketball
- NBA 2021 + WNBA shots
- + NBA 2022/23 stints (bardzo szczegolowe)
- Augmentacja 2x
- Focus na: FG%, 3P%, PTS, AST, REB

### 4. Model Tennis
- Australian Open + WTA
- + ATP shot-level data
- Focus na: rank_diff, surface, serve_stats

### 5. Model Baseball (USA)
- MLB standings + pitchers
- Augmentacja 2x
- Focus na: ERA, WHIP, OPS, wOBA

## API do Prawdziwych Danych (Produkcja)

Gdy model bedzie wytrenowany, mozesz uzyc platnych API:

| API | Sport | Cena | Link |
|-----|-------|------|------|
| The Odds API | Wszystkie | Darmowa / $29/msc | the-odds-api.com |
| API-Sports | Pilka nozna, NBA, MLB, NHL | 100 req/dzien free | api-sports.io |
| PandaScore | Esports | Darmowa / platna | pandascore.co |
| Sportradar | Wszystkie | Enterprise | sportradar.com |
| Football-Data.org | Pilka nozna | Darmowa / platna | football-data.org |
| Odds API | Kursy bukmacherskie | $29/msc | the-odds-api.com |

## Wnioski

System v3.0 obejmuje **13 sportow** pokrywajacych **>95% globalnego rynku bettingowego**.

Posiadasz juz:
- 106 plikow CSV z Score Network (54 MB)
- Agenty do web scrapingu i API
- System augmentacji danych (2-5x)
- Feature engineering dla kazdego sportu

Mozesz teraz trenowac modele na prawdziwych danych historycznych!
