## 8. SYSTEM RANKINGU TOP 3

### 8.1 `agents/ranker.py` - Agent rankingowy

```python
# agents/ranker.py
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from agents.data_evaluator import DataEvaluator, DataQualityReport
from core.quality_scorer import QualityAwareValueCalculator, QualityAdjustedPrediction
from core.value_calculator import ValueCalculator, KellyCriterion
from config.thresholds import thresholds
from config.leagues import classify_league

@dataclass
class RankedMatch:
    """Pojedynczy mecz w rankingu"""
    rank: int
    match_id: str
    match_name: str
    sport: str
    league: str
    league_type: str

    # Value metrics
    best_edge: float
    adjusted_edge: float
    best_odds: float
    best_bookmaker: str
    selection: str  # "home" / "away" / "over" / "under"

    # Quality metrics
    quality_score: float
    quality_report: DataQualityReport

    # Recommendation
    composite_score: float
    stake_recommendation: str
    risk_level: str  # "LOW" / "MEDIUM" / "HIGH"

    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "match_id": self.match_id,
            "match_name": self.match_name,
            "sport": self.sport,
            "league": self.league,
            "league_type": self.league_type,
            "best_edge": f"{self.best_edge:.2%}",
            "adjusted_edge": f"{self.adjusted_edge:.2%}",
            "best_odds": self.best_odds,
            "best_bookmaker": self.best_bookmaker,
            "selection": self.selection,
            "quality_score": f"{self.quality_score:.1f}/100",
            "composite_score": f"{self.composite_score:.3f}",
            "stake": self.stake_recommendation,
            "risk": self.risk_level,
            "reasoning": self.reasoning,
            "warnings": self.warnings
        }


class MatchRanker:
    """
    Agent rankingowy - wybiera Top 3 mecze dnia na podstawie:
    1. Value Edge (adjusted for quality)
    2. Data Quality Score
    3. Confidence Score

    KLUCZOWE: Composite score = edge * quality * confidence
    """

    def __init__(self, sport: str):
        self.sport = sport
        self.evaluator = DataEvaluator()
        self.value_calculator = QualityAwareValueCalculator(sport)
        self.kelly = KellyCriterion()

    async def rank_top_3_matches(
        self,
        date: str,
        fixtures: List[Dict],
        max_unpopular: int = 1
    ) -> List[RankedMatch]:
        """
        Znajdź 3 najlepsze mecze do zagrania.

        Args:
            date: Data w formacie YYYY-MM-DD
            fixtures: Lista meczów z danymi
            max_unpopular: Max meczów z lig niepopularnych (default: 1)

        Returns:
            Lista max 3 RankedMatch posortowanych po composite_score
        """
        print(f"[Ranker] Analyzing {len(fixtures)} matches for {date}...")

        # 1. Ewaluacja jakości wszystkich meczów
        quality_reports = await self.evaluator.batch_evaluate(fixtures, self.sport)

        # 2. Filtruj mecze z wystarczającą jakością
        valid_matches = []
        for fixture, report in zip(fixtures, quality_reports):
            if report.is_ready:
                valid_matches.append((fixture, report))
            else:
                print(f"[Ranker] Skipping {fixture.get('home')} vs {fixture.get('away')}: {report.recommendation}")

        print(f"[Ranker] {len(valid_matches)} matches passed quality filter")

        if not valid_matches:
            return []

        # 3. Oblicz value dla każdego meczu
        analyzed_matches = []

        for fixture, quality_report in valid_matches:
            # Oblicz prawdopodobieństwo (z modelu)
            raw_prob = await self._calculate_probability(fixture)

            # Oblicz value z korektą jakości
            value_result = await self.value_calculator.calculate_value_with_quality(
                match_id=fixture.get("id", ""),
                raw_probability=raw_prob,
                odds_by_bookmaker=fixture.get("odds", {}),
                quality_report=quality_report
            )

            if not value_result["has_value"]:
                continue

            # Oblicz composite score
            composite = self._calculate_composite_score(
                edge=value_result["adjusted_edge"],
                quality=quality_report.overall_score / 100,
                confidence=value_result["confidence"]
            )

            # Określ risk level
            risk = self._determine_risk_level(quality_report)

            # Stake recommendation
            stake = self._calculate_stake_recommendation(
                edge=value_result["adjusted_edge"],
                quality=quality_report.overall_score,
                league_type=quality_report.league_type
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(fixture, value_result, quality_report)

            analyzed_matches.append(RankedMatch(
                rank=0,  # Will be set after sorting
                match_id=fixture.get("id", ""),
                match_name=f"{fixture.get('home')} vs {fixture.get('away')}",
                sport=self.sport,
                league=fixture.get("league", ""),
                league_type=quality_report.league_type,
                best_edge=value_result["raw_edge"],
                adjusted_edge=value_result["adjusted_edge"],
                best_odds=value_result["best_odds"],
                best_bookmaker=value_result["best_bookmaker"],
                selection="home",  # Simplified - could be any market
                quality_score=quality_report.overall_score,
                quality_report=quality_report,
                composite_score=composite,
                stake_recommendation=stake,
                risk_level=risk,
                reasoning=reasoning,
                warnings=quality_report.warnings
            ))

        # 4. Sortuj po composite score
        analyzed_matches.sort(key=lambda x: x.composite_score, reverse=True)

        # 5. Wybierz Top 3 z ograniczeniami
        top_3 = self._select_top_3(analyzed_matches, max_unpopular)

        # 6. Ustaw ranki
        for i, match in enumerate(top_3):
            match.rank = i + 1

        return top_3

    def _calculate_composite_score(
        self,
        edge: float,
        quality: float,
        confidence: float
    ) -> float:
        """
        Oblicza composite score.

        Wzór: edge^0.5 * quality^0.3 * confidence^0.2

        Edge ma większą wagę, ale quality i confidence też są istotne.
        """
        # Normalize edge (cap at 20%)
        normalized_edge = min(edge, 0.20) / 0.20

        # Weighted geometric mean
        composite = (
            (normalized_edge ** 0.5) *
            (quality ** 0.3) *
            (confidence ** 0.2)
        )

        return composite

    def _determine_risk_level(self, quality_report: DataQualityReport) -> str:
        """Określa poziom ryzyka"""
        score = quality_report.overall_score

        if score >= 70 and quality_report.league_type == "popular":
            return "LOW"
        elif score >= 60 or quality_report.league_type == "popular":
            return "MEDIUM"
        else:
            return "HIGH"

    def _calculate_stake_recommendation(
        self,
        edge: float,
        quality: float,
        league_type: str
    ) -> str:
        """Oblicza zalecaną stawkę"""
        # Base stake from Kelly
        base_stake = self.kelly.calculate_stake(edge, quality / 100)

        # Adjust for league type
        if league_type == "unpopular":
            base_stake *= 0.5
        elif league_type == "medium":
            base_stake *= 0.75

        # Cap and format
        stake_pct = min(base_stake, 0.02) * 100  # Max 2%

        if stake_pct >= 1.5:
            return "2% bankroll"
        elif stake_pct >= 1.0:
            return "1.5% bankroll"
        else:
            return "1% bankroll"

    def _generate_reasoning(
        self,
        fixture: Dict,
        value_result: Dict,
        quality_report: DataQualityReport
    ) -> List[str]:
        """Generuje uzasadnienie dla rekomendacji"""
        reasons = []

        # Edge reasoning
        if value_result["adjusted_edge"] >= 0.05:
            reasons.append(f"Strong value: {value_result['adjusted_edge']:.1%} edge")
        else:
            reasons.append(f"Positive value: {value_result['adjusted_edge']:.1%} edge")

        # Quality reasoning
        if quality_report.overall_score >= 70:
            reasons.append(f"Good data quality ({quality_report.overall_score:.0f}/100)")

        # News reasoning
        if quality_report.news_report and quality_report.news_report.article_count >= 5:
            reasons.append(f"Strong news coverage ({quality_report.news_report.article_count} articles)")

        # Injury reasoning
        if quality_report.injuries_found:
            for inj in quality_report.injuries_found:
                if inj["status"] == "out":
                    reasons.append(f"Key absence: {inj['player']} OUT")

        return reasons[:4]  # Max 4 reasons

    def _select_top_3(
        self,
        matches: List[RankedMatch],
        max_unpopular: int
    ) -> List[RankedMatch]:
        """
        Wybiera Top 3 z ograniczeniami:
        - Max 1 mecz z ligi niepopularnej
        - Nie więcej niż 1 mecz z tego samego turnieju
        """
        selected = []
        unpopular_count = 0
        tournaments_used = set()

        for match in matches:
            if len(selected) >= 3:
                break

            # Check unpopular limit
            if match.league_type == "unpopular":
                if unpopular_count >= max_unpopular:
                    continue
                unpopular_count += 1

            # Check tournament uniqueness
            tournament = self._extract_tournament(match.league)
            if tournament in tournaments_used:
                continue
            tournaments_used.add(tournament)

            selected.append(match)

        return selected

    def _extract_tournament(self, league: str) -> str:
        """Wyciąga nazwę turnieju z ligi"""
        # Simplified - in real implementation would parse properly
        return league.split()[0] if league else "unknown"

    async def _calculate_probability(self, fixture: Dict) -> float:
        """
        Oblicza prawdopodobieństwo z modelu.
        Placeholder - w rzeczywistej implementacji użyłby modelu predykcji.
        """
        # Import model based on sport
        if self.sport == "tennis":
            from core.models.tennis_model import TennisPredictionModel
            model = TennisPredictionModel()
        else:
            from core.models.basketball_model import BasketballPredictionModel
            model = BasketballPredictionModel()

        return model.predict(fixture)


# === FORMATTING FUNCTIONS ===

def format_top_3_report(matches: List[RankedMatch], date: str) -> str:
    """
    Formatuje Top 3 do czytelnego raportu.
    """
    if not matches:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  TOP 3 VALUE BETS - {date}
╠══════════════════════════════════════════════════════════════╣
║  No value bets found today
║  All matches either lack sufficient data or have no edge
╚══════════════════════════════════════════════════════════════╝
"""

    report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  TOP 3 VALUE BETS - {date}
╠══════════════════════════════════════════════════════════════════════════╣
"""

    rank_emojis = ["Gold", "Silver", "Bronze"]

    for match in matches:
        emoji = rank_emojis[match.rank - 1] if match.rank <= 3 else "Medal"

        report += f"""║
║  {emoji} #{match.rank}: {match.match_name}
║  ────────────────────────────────────────────────────────────────────────
║  League: {match.league} ({match.league_type.upper()})
║
║  Edge: {match.adjusted_edge:.1%} (raw: {match.best_edge:.1%})
║  Odds: {match.best_odds:.2f} @ {match.best_bookmaker}
║  Quality: {match.quality_score:.0f}/100 | Composite: {match.composite_score:.3f}
║
║  Stake: {match.stake_recommendation} | Risk: {match.risk_level}
║
║  Reasoning:
"""
        for reason in match.reasoning:
            report += f"║    • {reason}\n"

        if match.warnings:
            report += "║  \n║  Warnings:\n"
            for warning in match.warnings[:2]:
                report += f"║    • {warning[:60]}\n"

    report += """║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    return report
```

---
