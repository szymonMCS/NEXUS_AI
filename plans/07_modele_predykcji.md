## 7. MODELE PREDYKCJI

### Uwaga

Sekcja 7 (MODELE PREDYKCJI) nie jest zawarta w oryginalnym pliku NEXUS_AI_v2_Pelna_Implementacja.md.

Dokument przeskakuje bezpośrednio z sekcji 6 (MCP SERVERS) na sekcję 8 (SYSTEM RANKINGU TOP 3).

Poniżej znajduje się placeholder dla tej sekcji, która powinna zawierać:

1. **Tennis Prediction Model** - Modele predykcji dla tenisa
   - `core/models/tennis_model.py`
   - Wykorzystanie rankings, recent form, H2H, surface stats

2. **Basketball Prediction Model** - Modele predykcji dla koszykówki
   - `core/models/basketball_model.py`
   - Wykorzystanie offensive/defensive ratings, recent games, rest days

3. **Base Model Class** - Bazowa klasa dla modeli
   - `core/models/base_model.py`
   - Interface dla przewidywania, walidacji itp.

### Placeholder Code

```python
# core/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BasePredictionModel(ABC):
    """
    Bazowa klasa dla modeli predykcji.
    Powinna być dziedziczona przez konkretne modele (Tennis, Basketball).
    """

    def __init__(self, sport: str):
        self.sport = sport

    @abstractmethod
    def predict(self, fixture: Dict) -> float:
        """
        Przewiduje prawdopodobieństwo wygranej gospodarza.

        Args:
            fixture: Dict z danymi meczu (ranking, forma, H2H, etc.)

        Returns:
            float: Prawdopodobieństwo 0-1
        """
        pass

    @abstractmethod
    def validate_data(self, fixture: Dict) -> bool:
        """
        Sprawdza czy dane do modelu są wystarczające.

        Args:
            fixture: Dict z danymi meczu

        Returns:
            bool: True jeśli dane są OK
        """
        pass

    @abstractmethod
    def explain_prediction(self, fixture: Dict) -> str:
        """
        Wyjaśnia predykcję.

        Args:
            fixture: Dict z danymi meczu

        Returns:
            str: Wyjaśnienie predykcji
        """
        pass


# core/models/tennis_model.py
from core.models.base_model import BasePredictionModel
from typing import Dict, Optional

class TennisPredictionModel(BasePredictionModel):
    """
    Model predykcji dla tenisa.

    Użytkowne features:
    - Rankings (ATP/WTA)
    - Recent form (ostatnie 5-10 meczów)
    - H2H record
    - Surface stats (win % na konkretnej nawierzchni)
    - Age, injuries, momentum
    """

    def __init__(self):
        super().__init__("tennis")

    def predict(self, fixture: Dict) -> float:
        """
        Przewiduje prawdopodobieństwo wygranej gospodarza (home player).

        Przykładowa implementacja (simplified):
        - Przeanalizuj ranking, formę, H2H
        - Zwiększ/zmniejsz prawdopodobieństwo na podstawie powierzchni
        - Uwzględnij recent form momentum
        """
        home_ranking = fixture.get("home_ranking", 999)
        away_ranking = fixture.get("away_ranking", 999)

        # Base probability from ranking difference
        if home_ranking < away_ranking:
            prob = 0.55 + (away_ranking - home_ranking) * 0.001
        else:
            prob = 0.50 - (home_ranking - away_ranking) * 0.001

        # Recent form adjustment
        home_form = fixture.get("home_recent_form", 0.5)
        away_form = fixture.get("away_recent_form", 0.5)
        prob += (home_form - away_form) * 0.1

        # H2H adjustment
        h2h_home_wins = fixture.get("h2h_home_wins", 0)
        h2h_total = fixture.get("h2h_matches", 1)
        if h2h_total > 0:
            h2h_winrate = h2h_home_wins / h2h_total
            prob = prob * 0.7 + h2h_winrate * 0.3

        # Surface adjustment
        surface = fixture.get("surface", "hard")
        home_surface_pct = fixture.get("home_surface_win_pct", 0.5)
        away_surface_pct = fixture.get("away_surface_win_pct", 0.5)
        prob += (home_surface_pct - away_surface_pct) * 0.1

        # Clamp to 0-1
        return max(0.1, min(prob, 0.9))

    def validate_data(self, fixture: Dict) -> bool:
        """
        Sprawdza czy dane do modelu są wystarczające.
        """
        required_fields = [
            "home_ranking", "away_ranking",
            "home_recent_form", "away_recent_form",
            "surface"
        ]
        return all(field in fixture for field in required_fields)

    def explain_prediction(self, fixture: Dict) -> str:
        """
        Wyjaśnia predykcję.
        """
        prob = self.predict(fixture)
        home = fixture.get("home", "Home")
        away = fixture.get("away", "Away")

        explanation = f"""
Tennis Prediction for {home} vs {away}
======================================
Probability (Home Win): {prob:.1%}

Factors:
- Ranking: {fixture.get('home_ranking', 'N/A')} vs {fixture.get('away_ranking', 'N/A')}
- Recent Form: {fixture.get('home_recent_form', 'N/A')} vs {fixture.get('away_recent_form', 'N/A')}
- Surface: {fixture.get('surface', 'N/A')} ({fixture.get('home_surface_win_pct', 0.5):.1%} vs {fixture.get('away_surface_win_pct', 0.5):.1%})
- H2H: {fixture.get('h2h_home_wins', 0)}-{fixture.get('h2h_away_wins', 0)}
"""
        return explanation


# core/models/basketball_model.py
from core.models.base_model import BasePredictionModel
from typing import Dict

class BasketballPredictionModel(BasePredictionModel):
    """
    Model predykcji dla koszykówki.

    Użytkowne features:
    - Offensive/Defensive Ratings
    - Recent games performance
    - Rest days (travel fatigue)
    - Home/Away record
    - Team strength metrics
    """

    def __init__(self):
        super().__init__("basketball")

    def predict(self, fixture: Dict) -> float:
        """
        Przewiduje prawdopodobieństwo wygranej gospodarza (home team).

        Przykładowa implementacja (simplified):
        - Porównaj offensive/defensive ratings
        - Uwzględnij rest days
        - Analiza recent games
        """
        home_ortg = fixture.get("home_offensive_rating", 100)
        home_drtg = fixture.get("home_defensive_rating", 100)
        away_ortg = fixture.get("away_offensive_rating", 100)
        away_drtg = fixture.get("away_defensive_rating", 100)

        # Base probability from net rating
        home_net = home_ortg - home_drtg
        away_net = away_ortg - away_drtg

        # Simple model: compare efficiency
        if (home_net + away_net) != 0:
            prob = 0.5 + (home_net - away_net) / ((home_net + away_net) * 2)
        else:
            prob = 0.5

        # Rest advantage
        home_rest = fixture.get("home_rest_days", 2)
        away_rest = fixture.get("away_rest_days", 2)
        if home_rest > away_rest + 1:
            prob += 0.05
        elif away_rest > home_rest + 1:
            prob -= 0.05

        # Recent performance
        home_recent = fixture.get("home_recent_games", [])
        if home_recent:
            avg_recent = sum(home_recent) / len(home_recent)
            prob += (avg_recent - 0.5) * 0.05

        # Home court advantage
        prob += 0.03

        # Clamp to 0-1
        return max(0.1, min(prob, 0.9))

    def validate_data(self, fixture: Dict) -> bool:
        """
        Sprawdza czy dane do modelu są wystarczające.
        """
        required_fields = [
            "home_offensive_rating", "home_defensive_rating",
            "away_offensive_rating", "away_defensive_rating"
        ]
        return all(field in fixture for field in required_fields)

    def explain_prediction(self, fixture: Dict) -> str:
        """
        Wyjaśnia predykcję.
        """
        prob = self.predict(fixture)
        home = fixture.get("home", "Home")
        away = fixture.get("away", "Away")

        home_ortg = fixture.get("home_offensive_rating", 100)
        home_drtg = fixture.get("home_defensive_rating", 100)
        away_ortg = fixture.get("away_offensive_rating", 100)
        away_drtg = fixture.get("away_defensive_rating", 100)

        explanation = f"""
Basketball Prediction for {home} vs {away}
==========================================
Probability (Home Win): {prob:.1%}

Efficiency Ratings:
- {home} ORTG: {home_ortg:.1f} | DRTG: {home_drtg:.1f} | Net: {home_ortg - home_drtg:+.1f}
- {away} ORTG: {away_ortg:.1f} | DRTG: {away_drtg:.1f} | Net: {away_ortg - away_drtg:+.1f}

Rest Days:
- {home}: {fixture.get('home_rest_days', 'N/A')} days
- {away}: {fixture.get('away_rest_days', 'N/A')} days
"""
        return explanation
```

### Przyszła Implementacja

Ta sekcja powinna być w pełni rozwinięta w wersji 2.1 systemu NEXUS AI, zawierając:

1. Bardziej zaawansowane modele (Random Forest, XGBoost, Neural Networks)
2. Feature engineering (momentum, volatility, team composition)
3. Backtesting framework
4. Hyperparameter tuning
5. Model validation i performance metrics
6. Integration z MLOps tools (MLflow, Weights & Biases)

---
