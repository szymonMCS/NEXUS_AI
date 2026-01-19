## 7. SYSTEM PREDYKCJI

### 7.1 Model Tenisowy

```python
# prediction/tennis_model.py

from dataclasses import dataclass
from typing import Dict, Optional
import math

@dataclass
class TennisPrediction:
    """Wynik predykcji meczu tenisowego"""
    home_win_prob: float
    away_win_prob: float
    confidence: float
    factors: Dict[str, float]
    reasoning: str


class TennisModel:
    """
    Model predykcji tenisa oparty na:
    - Rankingu (ELO-like)
    - Formie (ostatnie 5 meczów)
    - Nawierzchni
    - H2H
    """

    # Wagi czynników
    WEIGHTS = {
        "ranking": 0.35,
        "form": 0.25,
        "surface": 0.20,
        "h2h": 0.15,
        "fatigue": 0.05
    }

    def predict(self, match: Dict) -> TennisPrediction:
        """
        Oblicza prawdopodobieństwo wygranej.
        """
        stats = match.get("stats", {})

        factors = {}
        reasoning_parts = []

        # 1. RANKING FACTOR
        ranking_prob = self._ranking_probability(match, stats)
        factors["ranking"] = ranking_prob

        # 2. FORM FACTOR (jeśli dostępne)
        form_prob = self._form_probability(stats)
        factors["form"] = form_prob if form_prob else 0.5

        # 3. SURFACE FACTOR (jeśli dostępne)
        surface_prob = self._surface_probability(match, stats)
        factors["surface"] = surface_prob if surface_prob else 0.5

        # 4. H2H FACTOR (jeśli dostępne)
        h2h_prob = self._h2h_probability(stats)
        factors["h2h"] = h2h_prob if h2h_prob else 0.5

        # 5. FATIGUE FACTOR
        fatigue_adj = self._fatigue_adjustment(stats)
        factors["fatigue"] = 0.5 + fatigue_adj

        # WEIGHTED AVERAGE
        home_prob = sum(
            factors[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS.keys()
        )

        # Normalize to 0-1
        home_prob = max(0.1, min(0.9, home_prob))
        away_prob = 1 - home_prob

        # CONFIDENCE based on data quality
        confidence = self._calculate_confidence(match, factors)

        # REASONING
        reasoning = self._generate_reasoning(match, factors)

        return TennisPrediction(
            home_win_prob=round(home_prob, 3),
            away_win_prob=round(away_prob, 3),
            confidence=round(confidence, 2),
            factors=factors,
            reasoning=reasoning
        )

    def _ranking_probability(self, match: Dict, stats: Dict) -> float:
        """
        Oblicza prawdopodobieństwo na podstawie rankingu.

        Używa formuły ELO-like:
        P(home) = 1 / (1 + 10^(-diff/50))

        gdzie diff = strength_home - strength_away
        i strength = 1000 / (1 + log10(rank))
        """
        home_rank = stats.get("ranking_home", 100)
        away_rank = stats.get("ranking_away", 100)

        # Fallback z newsów jeśli brak danych
        if not home_rank or not away_rank:
            return 0.5

        # Oblicz strength
        home_strength = 1000 / (1 + math.log10(max(home_rank, 1)))
        away_strength = 1000 / (1 + math.log10(max(away_rank, 1)))

        diff = home_strength - away_strength

        # ELO formula
        prob = 1 / (1 + 10 ** (-diff / 50))

        return prob

    def _form_probability(self, stats: Dict) -> Optional[float]:
        """
        Oblicza prawdopodobieństwo na podstawie formy.
        """
        home_form = stats.get("home_last5_wins", None)
        away_form = stats.get("away_last5_wins", None)

        if home_form is None or away_form is None:
            return None

        # Prosta proporcja
        total = home_form + away_form
        if total == 0:
            return 0.5

        return home_form / total

    def _surface_probability(self, match: Dict, stats: Dict) -> Optional[float]:
        """
        Korekta za nawierzchnię.
        """
        surface = match.get("surface", "").lower()

        home_surface_pct = stats.get(f"home_{surface}_win_pct")
        away_surface_pct = stats.get(f"away_{surface}_win_pct")

        if home_surface_pct and away_surface_pct:
            total = home_surface_pct + away_surface_pct
            return home_surface_pct / total if total > 0 else 0.5

        return None

    def _h2h_probability(self, stats: Dict) -> Optional[float]:
        """
        Korekta za historię H2H.
        """
        h2h = stats.get("h2h", {})

        home_wins = h2h.get("home_wins", 0)
        away_wins = h2h.get("away_wins", 0)

        total = home_wins + away_wins
        if total < 2:  # Za mało danych
            return None

        return home_wins / total

    def _fatigue_adjustment(self, stats: Dict) -> float:
        """
        Korekta za zmęczenie (mecze w ostatnich 14 dniach).
        """
        home_matches = stats.get("home_matches_14d", 3)
        away_matches = stats.get("away_matches_14d", 3)

        # Optymalnie: 3-5 meczów
        def fatigue_penalty(matches):
            if matches < 2:
                return -0.05  # Za mało gier = brak rytmu
            elif matches > 6:
                return -0.05 * (matches - 6)  # Zmęczenie
            return 0

        home_penalty = fatigue_penalty(home_matches)
        away_penalty = fatigue_penalty(away_matches)

        return home_penalty - away_penalty

    def _calculate_confidence(self, match: Dict, factors: Dict) -> float:
        """
        Oblicza confidence na podstawie ilości dostępnych danych.
        """
        base = 0.3

        # Bonus za każdy dostępny factor
        available_factors = sum(1 for v in factors.values() if v != 0.5)
        base += available_factors * 0.1

        # Bonus za wiele źródeł
        sources = len(match.get("sources", []))
        base += min(sources * 0.1, 0.2)

        # Bonus za kursy
        if match.get("odds"):
            base += 0.1

        return min(base, 0.95)

    def _generate_reasoning(self, match: Dict, factors: Dict) -> str:
        """
        Generuje tekstowe uzasadnienie.
        """
        parts = []

        home = match.get("home", "Home")
        away = match.get("away", "Away")

        if factors["ranking"] > 0.6:
            parts.append(f"{home} has ranking advantage")
        elif factors["ranking"] < 0.4:
            parts.append(f"{away} has ranking advantage")

        if factors.get("form", 0.5) > 0.6:
            parts.append(f"{home} in better recent form")
        elif factors.get("form", 0.5) < 0.4:
            parts.append(f"{away} in better recent form")

        return "; ".join(parts) if parts else "Balanced match"
```

### 7.2 Value Calculator

```python
# prediction/value_calculator.py

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ValueBet:
    """Reprezentacja value bet"""
    has_value: bool
    selection: str  # "home" lub "away"
    probability: float
    best_odds: float
    best_bookmaker: str
    edge: float  # (prob * odds) - 1
    kelly_stake: float  # % bankroll
    reasoning: str


class ValueCalculator:
    """
    Oblicza value bety na podstawie prawdopodobieństw i kursów.
    """

    # Minimalne edge per typ ligi
    MIN_EDGE = {
        "popular": 0.03,    # 3%
        "medium": 0.04,     # 4%
        "unpopular": 0.05   # 5%
    }

    # Kelly fraction (conservative)
    KELLY_FRACTION = 0.25  # 1/4 Kelly

    # Max stake
    MAX_STAKE = 0.03  # 3% bankroll

    def calculate_value(
        self,
        home_prob: float,
        away_prob: float,
        odds: Dict,
        league_type: str = "medium",
        quality_score: float = 100.0
    ) -> Optional[ValueBet]:
        """
        Szuka value betu.

        Args:
            home_prob: Prawdopodobieństwo wygranej home (0-1)
            away_prob: Prawdopodobieństwo wygranej away (0-1)
            odds: Dict {bookmaker: {home: float, away: float}}
            league_type: "popular", "medium", "unpopular"
            quality_score: Score jakości danych (0-100)

        Returns:
            ValueBet lub None jeśli brak value
        """
        if not odds:
            return None

        min_edge = self.MIN_EDGE.get(league_type, 0.04)

        # Quality adjustment - wymagaj większego edge przy niższej jakości
        if quality_score < 60:
            min_edge *= 1.5

        # Znajdź najlepsze kursy
        best_home_odds = 0
        best_home_bookie = None
        best_away_odds = 0
        best_away_bookie = None

        for bookie, bookie_odds in odds.items():
            if not isinstance(bookie_odds, dict):
                continue

            home_odd = bookie_odds.get("home", 0)
            away_odd = bookie_odds.get("away", 0)

            if home_odd > best_home_odds:
                best_home_odds = home_odd
                best_home_bookie = bookie

            if away_odd > best_away_odds:
                best_away_odds = away_odd
                best_away_bookie = bookie

        if best_home_odds == 0 and best_away_odds == 0:
            return None

        # Oblicz edge dla obu stron
        home_edge = (home_prob * best_home_odds) - 1
        away_edge = (away_prob * best_away_odds) - 1

        # Wybierz lepszy bet
        if home_edge >= min_edge and home_edge >= away_edge:
            selection = "home"
            probability = home_prob
            best_odds = best_home_odds
            best_bookmaker = best_home_bookie
            edge = home_edge
        elif away_edge >= min_edge:
            selection = "away"
            probability = away_prob
            best_odds = best_away_odds
            best_bookmaker = best_away_bookie
            edge = away_edge
        else:
            return None  # Brak value

        # Kelly Criterion
        kelly_stake = self._kelly_criterion(probability, best_odds)

        # Quality adjustment for stake
        quality_multiplier = quality_score / 100
        adjusted_stake = kelly_stake * quality_multiplier

        # Cap stake
        final_stake = min(adjusted_stake, self.MAX_STAKE)

        return ValueBet(
            has_value=True,
            selection=selection,
            probability=probability,
            best_odds=best_odds,
            best_bookmaker=best_bookmaker,
            edge=round(edge, 4),
            kelly_stake=round(final_stake, 4),
            reasoning=self._generate_reasoning(
                selection, probability, best_odds, edge
            )
        )

    def _kelly_criterion(self, prob: float, odds: float) -> float:
        """
        Oblicza optymalną stawkę Kelly.

        f* = (bp - q) / b
        gdzie:
        - b = odds - 1
        - p = probability of winning
        - q = 1 - p
        """
        b = odds - 1
        p = prob
        q = 1 - p

        if b <= 0:
            return 0

        kelly = (b * p - q) / b

        # Apply fraction
        return max(0, kelly * self.KELLY_FRACTION)

    def _generate_reasoning(
        self,
        selection: str,
        prob: float,
        odds: float,
        edge: float
    ) -> str:
        """Generuje uzasadnienie"""
        return (
            f"{selection.upper()} at {odds:.2f} "
            f"(prob: {prob:.1%}, edge: {edge:.1%})"
        )
```
