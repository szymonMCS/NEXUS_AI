## 5. AGENT EWALUUJĄCY JAKOŚĆ DANYCH

### 5.1 `agents/data_evaluator.py` - Główny agent ewaluujący

```python
# agents/data_evaluator.py
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from data.news.aggregator import NewsAggregator
from data.news.validator import NewsValidator, NewsQualityReport
from data.news.injury_extractor import extract_injuries_from_news
from config.thresholds import thresholds, LEAGUE_REQUIREMENTS
from config.leagues import classify_league

@dataclass
class DataQualityReport:
    """Pełny raport jakości danych dla meczu"""

    # Identyfikacja
    match_id: str
    match_name: str
    sport: str
    league: str
    league_type: str  # popular/medium/unpopular

    # Scores (0-100)
    overall_score: float
    news_quality_score: float
    stats_completeness_score: float
    odds_quality_score: float

    # Sub-reports
    news_report: Optional[NewsQualityReport] = None

    # Metadata
    odds_sources_count: int = 0
    stats_fields_available: int = 0
    injuries_found: List[Dict] = field(default_factory=list)

    # Issues & Warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Final verdict
    is_ready: bool = False
    recommendation: str = "SKIP"  # PROCEED / CAUTION / SKIP

    @property
    def overall_score_text(self) -> str:
        return f"{self.overall_score:.1f}/100"

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "match_name": self.match_name,
            "sport": self.sport,
            "league": self.league,
            "league_type": self.league_type,
            "overall_score": self.overall_score,
            "overall_score_text": self.overall_score_text,
            "news_quality_score": self.news_quality_score,
            "stats_completeness_score": self.stats_completeness_score,
            "odds_quality_score": self.odds_quality_score,
            "odds_sources_count": self.odds_sources_count,
            "stats_fields_available": self.stats_fields_available,
            "injuries_found": self.injuries_found,
            "issues": self.issues,
            "warnings": self.warnings,
            "is_ready": self.is_ready,
            "recommendation": self.recommendation
        }


class DataEvaluator:
    """
    Agent ewaluujący jakość wszystkich danych przed analizą meczu.

    KLUCZOWY KOMPONENT SYSTEMU - chroni przed złymi predykcjami
    wynikającymi z niewystarczających lub nierzetelnych danych.
    """

    def __init__(self):
        self.news_aggregator = NewsAggregator()
        self.news_validator = NewsValidator()
        self.thresholds = thresholds

    async def evaluate_match(
        self,
        match_id: str,
        home: str,
        away: str,
        sport: str,
        league: str,
        stats: Optional[Dict] = None,
        odds: Optional[Dict] = None
    ) -> DataQualityReport:
        """
        Przeprowadza pełną ewaluację jakości danych dla meczu.

        Args:
            match_id: Unikalny identyfikator meczu
            home: Nazwa zawodnika/drużyny gospodarza
            away: Nazwa zawodnika/drużyny gościa
            sport: "tennis" lub "basketball"
            league: Nazwa ligi
            stats: Opcjonalne - już pobrane statystyki
            odds: Opcjonalne - już pobrane kursy

        Returns:
            DataQualityReport z pełną oceną
        """
        match_name = f"{home} vs {away}"
        league_type = classify_league(league, sport)
        requirements = LEAGUE_REQUIREMENTS[league_type]

        issues = []
        warnings = []

        # === 1. EWALUACJA NEWSÓW ===
        print(f"[DataEvaluator] Fetching news for {match_name}...")
        news_data = await self.news_aggregator.get_match_news(home, away, sport)
        news_report = self.news_validator.validate_news_quality(news_data)

        # Sprawdź podejrzane wzorce
        suspicious = self.news_validator.detect_suspicious_patterns(news_data)
        warnings.extend(suspicious)

        # Ekstrakcja kontuzji
        injuries = []
        if news_data["articles"]:
            injuries = await extract_injuries_from_news(news_data["articles"])
            if injuries:
                for inj in injuries:
                    if inj["status"] == "out":
                        warnings.append(f"Warning {inj['player']} is OUT: {inj['injury_type']}")

        # News quality score (0-100)
        news_quality_score = news_report.quality_score * 100

        if not news_report.is_sufficient:
            issues.append("Insufficient news coverage")

        # === 2. EWALUACJA STATYSTYK ===
        stats_completeness_score = self._evaluate_stats(stats, sport, requirements)

        if stats_completeness_score < 50:
            issues.append(f"Stats completeness only {stats_completeness_score:.0f}%")

        # === 3. EWALUACJA KURSÓW ===
        odds_quality_score, odds_count = self._evaluate_odds(odds, requirements)

        if odds_count < requirements.min_bookmakers:
            issues.append(f"Only {odds_count}/{requirements.min_bookmakers} bookmakers")

        # === 4. OBLICZ OVERALL SCORE ===
        # Wagi zależne od typu ligi
        if league_type == "unpopular":
            # Dla niepopularnych lig bardziej liczy się ilość danych
            weights = {
                "news": 0.25,
                "stats": 0.40,  # Wyższa waga!
                "odds": 0.35
            }
        elif league_type == "medium":
            weights = {
                "news": 0.30,
                "stats": 0.35,
                "odds": 0.35
            }
        else:  # popular
            weights = {
                "news": 0.30,
                "stats": 0.30,
                "odds": 0.40
            }

        overall_score = (
            news_quality_score * weights["news"] +
            stats_completeness_score * weights["stats"] +
            odds_quality_score * weights["odds"]
        )

        # === 5. OKREŚL RECOMMENDATION ===
        if overall_score >= thresholds.quality_good * 100:
            recommendation = "PROCEED"
            is_ready = True
        elif overall_score >= thresholds.quality_moderate * 100:
            recommendation = "CAUTION"
            is_ready = True
            warnings.append("Moderate data quality - proceed with caution")
        elif overall_score >= thresholds.quality_high_risk * 100:
            recommendation = "HIGH_RISK"
            is_ready = True  # Still allow but with warnings
            warnings.append("HIGH RISK - minimal stake recommended")
        else:
            recommendation = "SKIP"
            is_ready = False
            issues.append(f"Quality score {overall_score:.1f} below threshold {thresholds.quality_reject * 100}")

        # Dodatkowe penalizacje dla niepopularnych lig
        if league_type == "unpopular" and overall_score < 60:
            recommendation = "SKIP"
            is_ready = False
            issues.append("Unpopular league requires higher data quality")

        return DataQualityReport(
            match_id=match_id,
            match_name=match_name,
            sport=sport,
            league=league,
            league_type=league_type,
            overall_score=round(overall_score, 1),
            news_quality_score=round(news_quality_score, 1),
            stats_completeness_score=round(stats_completeness_score, 1),
            odds_quality_score=round(odds_quality_score, 1),
            news_report=news_report,
            odds_sources_count=odds_count,
            stats_fields_available=len(stats) if stats else 0,
            injuries_found=injuries,
            issues=issues,
            warnings=warnings,
            is_ready=is_ready,
            recommendation=recommendation
        )

    def _evaluate_stats(
        self,
        stats: Optional[Dict],
        sport: str,
        requirements
    ) -> float:
        """
        Ocenia kompletność statystyk.

        Returns:
            Score 0-100
        """
        if not stats:
            return 0.0

        # Wymagane pola per sport
        if sport == "tennis":
            required_fields = [
                "home_ranking", "away_ranking",
                "home_recent_form", "away_recent_form",
                "surface", "home_surface_win_pct", "away_surface_win_pct",
                "h2h_matches", "home_last5_wins", "away_last5_wins"
            ]
        else:  # basketball
            required_fields = [
                "home_offensive_rating", "home_defensive_rating",
                "away_offensive_rating", "away_defensive_rating",
                "home_recent_games", "away_recent_games",
                "home_rest_days", "away_rest_days",
                "home_home_record", "away_away_record"
            ]

        # Policz dostępne pola
        available = sum(1 for field in required_fields if field in stats and stats[field] is not None)

        completeness = (available / len(required_fields)) * 100

        # Bonus za dodatkowe pola
        extra_fields = len([k for k in stats.keys() if k not in required_fields])
        bonus = min(extra_fields * 2, 20)  # Max 20% bonus

        return min(completeness + bonus, 100)

    def _evaluate_odds(
        self,
        odds: Optional[Dict],
        requirements
    ) -> tuple[float, int]:
        """
        Ocenia jakość kursów.

        Returns:
            (score 0-100, bookmakers_count)
        """
        if not odds:
            return 0.0, 0

        # Policz bukmacherów
        bookmakers_count = len(odds)

        if bookmakers_count == 0:
            return 0.0, 0

        # Bazowy score za liczbę bukmacherów
        min_required = requirements.min_bookmakers
        bookmaker_score = min(bookmakers_count / min_required, 1.0) * 50

        # Sprawdź spójność kursów (variance)
        consistency_score = self._calculate_odds_consistency(odds) * 50

        return bookmaker_score + consistency_score, bookmakers_count

    def _calculate_odds_consistency(self, odds: Dict) -> float:
        """
        Oblicza spójność kursów między bukmacherami.
        Niski rozrzut = wysoka spójność = lepiej.

        Returns:
            Score 0-1
        """
        if len(odds) < 2:
            return 0.5  # Can't compare

        # Zbierz wszystkie kursy na "home"
        home_odds = []
        for bookie, markets in odds.items():
            if isinstance(markets, dict):
                home_odd = markets.get("home") or markets.get("h2h_home")
                if home_odd:
                    home_odds.append(float(home_odd))

        if len(home_odds) < 2:
            return 0.5

        # Oblicz variance
        mean = sum(home_odds) / len(home_odds)
        variance = sum((x - mean) ** 2 for x in home_odds) / len(home_odds)
        std_dev = variance ** 0.5

        # Coefficient of variation
        cv = std_dev / mean if mean > 0 else 1

        # Convert to score (lower CV = higher score)
        # CV < 0.02 = excellent, CV > 0.10 = poor
        if cv < 0.02:
            return 1.0
        elif cv < 0.05:
            return 0.8
        elif cv < 0.10:
            return 0.5
        else:
            return 0.2

    async def batch_evaluate(
        self,
        matches: List[Dict],
        sport: str
    ) -> List[DataQualityReport]:
        """
        Ewaluuje wiele meczów równolegle.

        Args:
            matches: Lista słowników z danymi meczów
            sport: "tennis" lub "basketball"

        Returns:
            Lista DataQualityReport
        """
        tasks = []

        for match in matches:
            task = self.evaluate_match(
                match_id=match.get("id", "unknown"),
                home=match.get("home", ""),
                away=match.get("away", ""),
                sport=sport,
                league=match.get("league", ""),
                stats=match.get("stats"),
                odds=match.get("odds")
            )
            tasks.append(task)

        # Execute in parallel with semaphore to limit concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        reports = []
        for result in results:
            if isinstance(result, DataQualityReport):
                reports.append(result)
            else:
                print(f"Evaluation error: {result}")

        return reports


# === UTILITY FUNCTIONS ===

def print_quality_report(report: DataQualityReport) -> str:
    """
    Formatuje raport jakości do czytelnego formatu.
    """
    status_emoji = {
        "PROCEED": "Check",
        "CAUTION": "Warning",
        "HIGH_RISK": "CircleX",
        "SKIP": "X"
    }

    output = f"""
╔══════════════════════════════════════════════════════════════╗
║  DATA QUALITY REPORT: {report.match_name[:40]:<40}
╠══════════════════════════════════════════════════════════════╣
║  League: {report.league} ({report.league_type})
║  Sport:  {report.sport}
╠══════════════════════════════════════════════════════════════╣
║  OVERALL SCORE: {report.overall_score_text:>8}  {status_emoji.get(report.recommendation, '?')} {report.recommendation}
╠──────────────────────────────────────────────────────────────╣
║  News Quality:      {report.news_quality_score:>6.1f}/100
║  Stats Complete:    {report.stats_completeness_score:>6.1f}/100
║  Odds Quality:      {report.odds_quality_score:>6.1f}/100
╠──────────────────────────────────────────────────────────────╣
║  Bookmakers: {report.odds_sources_count}  |  Stats fields: {report.stats_fields_available}
"""

    if report.injuries_found:
        output += "╠──────────────────────────────────────────────────────────────╣\n"
        output += "║  INJURIES:\n"
        for inj in report.injuries_found:
            output += f"║    - {inj['player']}: {inj['status'].upper()} ({inj['injury_type']})\n"

    if report.warnings:
        output += "╠──────────────────────────────────────────────────────────────╣\n"
        output += "║  WARNINGS:\n"
        for warning in report.warnings[:5]:
            output += f"║    - {warning[:55]}\n"

    if report.issues:
        output += "╠──────────────────────────────────────────────────────────────╣\n"
        output += "║  ISSUES:\n"
        for issue in report.issues[:5]:
            output += f"║    - {issue[:55]}\n"

    output += "╚══════════════════════════════════════════════════════════════╝"

    return output
```

### 5.2 `core/quality_scorer.py` - Obliczanie composite quality score

```python
# core/quality_scorer.py
from typing import Dict, Optional
from dataclasses import dataclass
from agents.data_evaluator import DataQualityReport
from config.thresholds import thresholds

@dataclass
class QualityAdjustedPrediction:
    """Predykcja z korektą za jakość danych"""
    raw_probability: float
    adjusted_probability: float
    raw_edge: float
    adjusted_edge: float
    quality_multiplier: float
    quality_penalty: float
    confidence: float


class QualityScorer:
    """
    Oblicza composite quality score i stosuje korekty do predykcji.

    KLUCZOWE: Niższa jakość danych = niższy efektywny edge
    """

    def __init__(self):
        self.thresholds = thresholds

    def calculate_quality_multiplier(self, quality_report: DataQualityReport) -> float:
        """
        Oblicza mnożnik jakości dla edge.

        Quality 85-100%: multiplier = 1.0 (pełny edge)
        Quality 70-85%:  multiplier = 0.9
        Quality 50-70%:  multiplier = 0.7
        Quality 40-50%:  multiplier = 0.5
        Quality <40%:    multiplier = 0.3 (minimal)

        Returns:
            Multiplier 0.3-1.0
        """
        score = quality_report.overall_score / 100  # Normalize to 0-1

        if score >= self.thresholds.quality_excellent:
            return 1.0
        elif score >= self.thresholds.quality_good:
            return 0.9
        elif score >= self.thresholds.quality_moderate:
            return 0.7
        elif score >= self.thresholds.quality_high_risk:
            return 0.5
        else:
            return 0.3

    def adjust_prediction(
        self,
        raw_probability: float,
        best_odds: float,
        quality_report: DataQualityReport
    ) -> QualityAdjustedPrediction:
        """
        Dostosowuje predykcję na podstawie jakości danych.

        Args:
            raw_probability: Surowe prawdopodobieństwo z modelu
            best_odds: Najlepszy dostępny kurs
            quality_report: Raport jakości danych

        Returns:
            QualityAdjustedPrediction z adjusted values
        """
        quality_multiplier = self.calculate_quality_multiplier(quality_report)
        quality_penalty = 1 - quality_multiplier

        # Raw edge
        raw_edge = (raw_probability * best_odds) - 1

        # Adjusted edge - mnożnik za jakość
        adjusted_edge = raw_edge * quality_multiplier

        # Adjusted probability (conservative adjustment)
        # Przy niskiej jakości przesuwamy prawdopodobieństwo bliżej 0.5
        if quality_multiplier < 0.7:
            # Move towards 0.5
            adjustment_strength = (0.7 - quality_multiplier) / 0.4  # 0-1
            adjusted_probability = raw_probability + (0.5 - raw_probability) * adjustment_strength * 0.3
        else:
            adjusted_probability = raw_probability

        # Confidence = quality * (1 - variance_penalty)
        confidence = quality_report.overall_score / 100
        if quality_report.news_report and quality_report.news_report.diversity_score < 0.3:
            confidence *= 0.8  # Penalty for low diversity

        return QualityAdjustedPrediction(
            raw_probability=raw_probability,
            adjusted_probability=adjusted_probability,
            raw_edge=raw_edge,
            adjusted_edge=adjusted_edge,
            quality_multiplier=quality_multiplier,
            quality_penalty=quality_penalty,
            confidence=confidence
        )

    def should_bet(
        self,
        adjusted_prediction: QualityAdjustedPrediction,
        league_type: str
    ) -> tuple[bool, str]:
        """
        Określa czy zakład powinien zostać postawiony.

        Returns:
            (should_bet, reason)
        """
        # Minimum edge per league type
        min_edges = {
            "popular": self.thresholds.min_edge_popular_league,
            "medium": self.thresholds.min_edge_medium_league,
            "unpopular": self.thresholds.min_edge_unpopular_league
        }

        min_edge = min_edges.get(league_type, 0.05)

        # Check adjusted edge
        if adjusted_prediction.adjusted_edge < min_edge:
            return False, f"Adjusted edge {adjusted_prediction.adjusted_edge:.2%} < min {min_edge:.2%}"

        # Check confidence
        if adjusted_prediction.confidence < 0.4:
            return False, f"Confidence {adjusted_prediction.confidence:.2%} too low"

        # Check quality penalty
        if adjusted_prediction.quality_penalty > 0.7:
            return False, f"Quality penalty {adjusted_prediction.quality_penalty:.2%} too high"

        return True, "All checks passed"


# === INTEGRATION WITH VALUE CALCULATOR ===

class QualityAwareValueCalculator:
    """
    Value Calculator z uwzględnieniem jakości danych.
    """

    def __init__(self, sport: str):
        self.sport = sport
        self.quality_scorer = QualityScorer()

    async def calculate_value_with_quality(
        self,
        match_id: str,
        raw_probability: float,
        odds_by_bookmaker: Dict,
        quality_report: DataQualityReport
    ) -> Dict:
        """
        Oblicza value z pełnym uwzględnieniem jakości danych.
        """
        # Find best odds
        best_odds = 0
        best_bookmaker = None

        for bookie, markets in odds_by_bookmaker.items():
            if isinstance(markets, dict):
                odds = markets.get("home", 0)
                if odds > best_odds:
                    best_odds = odds
                    best_bookmaker = bookie

        if best_odds == 0:
            return {
                "has_value": False,
                "reason": "No odds available"
            }

        # Calculate adjusted prediction
        adjusted = self.quality_scorer.adjust_prediction(
            raw_probability=raw_probability,
            best_odds=best_odds,
            quality_report=quality_report
        )

        # Check if should bet
        should_bet, reason = self.quality_scorer.should_bet(
            adjusted,
            quality_report.league_type
        )

        return {
            "has_value": should_bet,
            "reason": reason,
            "raw_probability": adjusted.raw_probability,
            "adjusted_probability": adjusted.adjusted_probability,
            "best_odds": best_odds,
            "best_bookmaker": best_bookmaker,
            "raw_edge": adjusted.raw_edge,
            "adjusted_edge": adjusted.adjusted_edge,
            "quality_multiplier": adjusted.quality_multiplier,
            "quality_penalty": adjusted.quality_penalty,
            "confidence": adjusted.confidence,
            "quality_score": quality_report.overall_score
        }
```

---
