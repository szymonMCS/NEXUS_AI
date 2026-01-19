## 6. EWALUATOR JAKOŚCI DANYCH WEB

### 6.1 Web Data Evaluator - Główny komponent

```python
# evaluator/web_data_evaluator.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class WebQualityReport:
    """Raport jakości danych z internetu"""

    overall_score: float  # 0-100

    # Sub-scores (0-1)
    source_agreement_score: float
    freshness_score: float
    completeness_score: float
    reliability_score: float

    # Metadata
    sources_found: int
    sources_agreeing: int
    data_age_hours: Optional[float]
    missing_fields: List[str] = field(default_factory=list)

    # Verdict
    is_trustworthy: bool = False
    recommendation: str = "SKIP"  # PROCEED / CAUTION / SKIP
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class WebDataEvaluator:
    """
    Ewaluator jakości danych zebranych z internetu.

    KLUCZOWY KOMPONENT - chroni przed złymi predykcjami
    wynikającymi z niespójnych lub nieaktualnych danych web.

    Kryteria:
    1. SOURCE_AGREEMENT (35%) - Czy różne źródła zgadzają się?
    2. FRESHNESS (25%) - Jak stare są dane?
    3. COMPLETENESS (25%) - Czy mamy wszystkie potrzebne dane?
    4. RELIABILITY (15%) - Czy źródła są wiarygodne?
    """

    # Wagi dla composite score
    WEIGHTS = {
        "source_agreement": 0.35,
        "freshness": 0.25,
        "completeness": 0.25,
        "reliability": 0.15
    }

    # Progi jakości
    THRESHOLD_PROCEED = 0.65   # >= 65% = PROCEED
    THRESHOLD_CAUTION = 0.45   # >= 45% = CAUTION
    # < 45% = SKIP

    # Wymagane pola per sport
    REQUIRED_FIELDS = {
        "tennis": [
            "home", "away", "league", "time",
            "odds",  # Kursy od min. 1 bukmachera
        ],
        "basketball": [
            "home", "away", "league", "time",
            "odds",
        ]
    }

    # Pożądane pola (bonus)
    DESIRED_FIELDS = {
        "tennis": ["stats", "news", "h2h", "ranking_home", "ranking_away"],
        "basketball": ["stats", "news", "home_form", "away_form"]
    }

    def evaluate(self, match: Dict, sport: str) -> WebQualityReport:
        """
        Przeprowadza pełną ewaluację jakości danych meczu.

        Args:
            match: Słownik z danymi meczu (z DataEnricher)
            sport: "tennis" lub "basketball"

        Returns:
            WebQualityReport z oceną i rekomendacją
        """
        issues = []
        warnings = []

        # 1. SOURCE AGREEMENT - Czy źródła się zgadzają?
        agreement_score, agreement_issues = self._check_source_agreement(match)
        issues.extend(agreement_issues)

        # 2. FRESHNESS - Jak świeże są dane?
        freshness_score, data_age, freshness_issues = self._check_freshness(match)
        issues.extend(freshness_issues)

        # 3. COMPLETENESS - Czy mamy wszystkie dane?
        completeness_score, missing, completeness_issues = self._check_completeness(
            match, sport
        )
        issues.extend(completeness_issues)

        # 4. RELIABILITY - Czy źródła są wiarygodne?
        reliability_score, reliability_warnings = self._check_reliability(match)
        warnings.extend(reliability_warnings)

        # OVERALL SCORE (weighted average)
        overall = (
            agreement_score * self.WEIGHTS["source_agreement"] +
            freshness_score * self.WEIGHTS["freshness"] +
            completeness_score * self.WEIGHTS["completeness"] +
            reliability_score * self.WEIGHTS["reliability"]
        ) * 100

        # RECOMMENDATION
        if overall >= self.THRESHOLD_PROCEED * 100:
            recommendation = "PROCEED"
            is_trustworthy = True
        elif overall >= self.THRESHOLD_CAUTION * 100:
            recommendation = "CAUTION"
            is_trustworthy = True
            warnings.append("Moderate data quality - smaller stake recommended")
        else:
            recommendation = "SKIP"
            is_trustworthy = False
            issues.append(f"Quality score {overall:.1f}% below minimum threshold")

        return WebQualityReport(
            overall_score=round(overall, 1),
            source_agreement_score=round(agreement_score, 3),
            freshness_score=round(freshness_score, 3),
            completeness_score=round(completeness_score, 3),
            reliability_score=round(reliability_score, 3),
            sources_found=len(match.get("sources", [])),
            sources_agreeing=self._count_agreeing_sources(match),
            data_age_hours=data_age,
            missing_fields=missing,
            is_trustworthy=is_trustworthy,
            recommendation=recommendation,
            issues=issues,
            warnings=warnings
        )

    def _check_source_agreement(self, match: Dict) -> tuple[float, List[str]]:
        """
        Sprawdza czy różne źródła zgadzają się co do danych.

        Sprawdzane elementy:
        - Nazwy zawodników/drużyn
        - Czas meczu
        - Kursy (variance < 10%)
        """
        issues = []
        score = 0.0

        sources = match.get("sources", [])

        # Jeśli tylko jedno źródło - średni score
        if len(sources) <= 1:
            issues.append("Only one data source - cannot verify")
            return 0.5, issues

        # Bonus za wiele źródeł
        source_bonus = min(len(sources) / 3, 1.0) * 0.3
        score += source_bonus

        # Sprawdź spójność kursów
        odds = match.get("odds", {})
        if len(odds) >= 2:
            odds_variance = self._calculate_odds_variance(odds)
            if odds_variance < 0.05:
                score += 0.4  # Bardzo spójne
            elif odds_variance < 0.10:
                score += 0.25
            else:
                issues.append(f"High odds variance: {odds_variance:.1%}")
                score += 0.1
        else:
            score += 0.2  # Neutralne

        # Sprawdź czy mamy dane z różnych typów źródeł
        source_types = set()
        for s in sources:
            if s in ["sofascore", "flashscore"]:
                source_types.add("scraping")
            elif s in ["thesportsdb", "api-sports"]:
                source_types.add("api")

        if len(source_types) >= 2:
            score += 0.3  # Różnorodność źródeł

        return min(score, 1.0), issues

    def _check_freshness(self, match: Dict) -> tuple[float, Optional[float], List[str]]:
        """
        Sprawdza świeżość danych.

        Priorytet: newsy < 24h, kursy < 1h
        """
        issues = []
        score = 0.5  # Domyślnie neutralne
        data_age = None

        # Sprawdź wiek newsów
        news = match.get("news", {}).get("combined", [])
        if news:
            # Jeśli mamy newsy, to dobrze
            score += 0.3

            # Sprawdź czy newsy są świeże (< 48h)
            # TODO: Parsowanie dat z newsów
        else:
            issues.append("No recent news found")

        # Sprawdź czy mamy świeże kursy
        odds = match.get("odds", {})
        if odds:
            score += 0.2
        else:
            issues.append("No odds data found")

        return score, data_age, issues

    def _check_completeness(
        self,
        match: Dict,
        sport: str
    ) -> tuple[float, List[str], List[str]]:
        """
        Sprawdza kompletność danych.
        """
        issues = []
        missing = []

        required = self.REQUIRED_FIELDS.get(sport, [])
        desired = self.DESIRED_FIELDS.get(sport, [])

        # Sprawdź wymagane pola
        required_present = 0
        for field in required:
            if field == "odds":
                if match.get("odds"):
                    required_present += 1
                else:
                    missing.append("odds")
            elif match.get(field):
                required_present += 1
            else:
                missing.append(field)

        required_score = required_present / len(required) if required else 1.0

        # Sprawdź pożądane pola (bonus)
        desired_present = sum(1 for f in desired if match.get(f))
        desired_score = desired_present / len(desired) if desired else 0

        # Combined score
        score = required_score * 0.7 + desired_score * 0.3

        if missing:
            issues.append(f"Missing required fields: {missing}")

        return score, missing, issues

    def _check_reliability(self, match: Dict) -> tuple[float, List[str]]:
        """
        Sprawdza wiarygodność źródeł.
        """
        warnings = []
        score = 0.5

        sources = match.get("sources", [])

        # Wiarygodne źródła
        reliable_sources = ["sofascore", "flashscore", "thesportsdb"]
        reliable_count = sum(1 for s in sources if s in reliable_sources)

        if reliable_count >= 2:
            score = 0.9
        elif reliable_count >= 1:
            score = 0.7
        else:
            warnings.append("No reliable sources found")
            score = 0.3

        # Sprawdź wiarygodność newsów
        news = match.get("news", {})
        brave_count = len(news.get("brave", []))
        serper_count = len(news.get("serper", []))

        if brave_count > 0 and serper_count > 0:
            score += 0.1  # Bonus za oba źródła newsów

        return min(score, 1.0), warnings

    def _calculate_odds_variance(self, odds: Dict) -> float:
        """Oblicza variance kursów między bukmacherami"""
        home_odds = []

        for bookie, odds_data in odds.items():
            if isinstance(odds_data, dict) and "home" in odds_data:
                home_odds.append(odds_data["home"])

        if len(home_odds) < 2:
            return 0.0

        mean = sum(home_odds) / len(home_odds)
        variance = sum((x - mean) ** 2 for x in home_odds) / len(home_odds)

        # Coefficient of variation
        cv = (variance ** 0.5) / mean if mean > 0 else 0
        return cv

    def _count_agreeing_sources(self, match: Dict) -> int:
        """Liczy ile źródeł ma zgodne dane"""
        return len(match.get("sources", []))


# === BATCH EVALUATION ===

async def evaluate_all_matches(
    matches: List[Dict],
    sport: str
) -> List[tuple[Dict, WebQualityReport]]:
    """
    Ewaluuje wszystkie mecze i zwraca z raportami.
    """
    evaluator = WebDataEvaluator()

    results = []
    for match in matches:
        report = evaluator.evaluate(match, sport)
        results.append((match, report))

    return results


def filter_by_quality(
    matches_with_reports: List[tuple[Dict, WebQualityReport]],
    min_quality: float = 45.0
) -> List[tuple[Dict, WebQualityReport]]:
    """
    Filtruje mecze po minimalnej jakości.
    """
    return [
        (match, report)
        for match, report in matches_with_reports
        if report.overall_score >= min_quality
    ]
```
