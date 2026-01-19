## 8. GENERATOR RAPORTÓW

### 8.1 Match Ranker

```python
# ranking/match_ranker.py

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from evaluator.web_data_evaluator import WebQualityReport
from prediction.value_calculator import ValueBet

@dataclass
class RankedBet:
    """Bet z pełnym rankingiem"""
    rank: int
    match_name: str
    league: str
    sport: str

    # Value
    selection: str
    probability: float
    odds: float
    bookmaker: str
    edge: float
    stake_recommendation: str

    # Quality
    quality_score: float
    data_sources: int

    # Composite
    composite_score: float

    # Reasoning
    prediction_reasoning: str
    value_reasoning: str
    quality_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MatchRanker:
    """
    Rankuje mecze i wybiera Top N betów.

    Composite Score = Edge × Quality × Confidence
    """

    def rank_bets(
        self,
        matches_with_predictions: List[Tuple[Dict, WebQualityReport, ValueBet]],
        top_n: int = 5,
        max_unpopular: int = 1
    ) -> List[RankedBet]:
        """
        Rankuje bety i wybiera Top N.

        Args:
            matches_with_predictions: Lista (match, quality_report, value_bet)
            top_n: Ile betów zwrócić
            max_unpopular: Max betów z niepopularnych lig

        Returns:
            Lista RankedBet posortowana po composite_score
        """
        scored = []

        for match, quality_report, value_bet in matches_with_predictions:
            if not value_bet or not value_bet.has_value:
                continue

            if not quality_report.is_trustworthy:
                continue

            # Calculate composite score
            composite = self._calculate_composite(
                edge=value_bet.edge,
                quality=quality_report.overall_score / 100,
                confidence=value_bet.probability  # Use probability as proxy
            )

            # Stake recommendation
            stake = self._stake_recommendation(value_bet.kelly_stake)

            scored.append(RankedBet(
                rank=0,  # Set later
                match_name=f"{match['home']} vs {match['away']}",
                league=match.get("league", "Unknown"),
                sport=match.get("sport", "unknown"),
                selection=value_bet.selection,
                probability=value_bet.probability,
                odds=value_bet.best_odds,
                bookmaker=value_bet.best_bookmaker,
                edge=value_bet.edge,
                stake_recommendation=stake,
                quality_score=quality_report.overall_score,
                data_sources=quality_report.sources_found,
                composite_score=composite,
                prediction_reasoning=match.get("prediction", {}).get("reasoning", ""),
                value_reasoning=value_bet.reasoning,
                quality_issues=quality_report.issues,
                warnings=quality_report.warnings
            ))

        # Sort by composite score
        scored.sort(key=lambda x: x.composite_score, reverse=True)

        # Apply constraints and select top N
        selected = self._select_top_n(scored, top_n, max_unpopular)

        # Set ranks
        for i, bet in enumerate(selected):
            bet.rank = i + 1

        return selected

    def _calculate_composite(
        self,
        edge: float,
        quality: float,
        confidence: float
    ) -> float:
        """
        Oblicza composite score.

        Używa geometric mean z wagami.
        """
        # Normalize edge (cap at 15%)
        norm_edge = min(edge, 0.15) / 0.15

        # Geometric weighted mean
        composite = (
            (norm_edge ** 0.4) *
            (quality ** 0.35) *
            (confidence ** 0.25)
        )

        return composite

    def _stake_recommendation(self, kelly_stake: float) -> str:
        """Konwertuje Kelly stake na rekomendację"""
        pct = kelly_stake * 100

        if pct >= 2.5:
            return "2.5% bankroll (HIGH CONFIDENCE)"
        elif pct >= 1.5:
            return "1.5-2% bankroll"
        elif pct >= 1.0:
            return "1% bankroll"
        else:
            return "0.5-1% bankroll (LOW)"

    def _select_top_n(
        self,
        scored: List[RankedBet],
        top_n: int,
        max_unpopular: int
    ) -> List[RankedBet]:
        """
        Wybiera Top N z ograniczeniami:
        - Max N betów z niepopularnych lig
        - Max 1 bet na turniej
        """
        selected = []
        unpopular_count = 0
        tournaments_used = set()

        for bet in scored:
            if len(selected) >= top_n:
                break

            # Check unpopular limit
            league_lower = bet.league.lower()
            is_unpopular = any(
                term in league_lower
                for term in ["itf", "challenger", "futures", "2. liga", "3. liga"]
            )

            if is_unpopular:
                if unpopular_count >= max_unpopular:
                    continue
                unpopular_count += 1

            # Check tournament uniqueness
            tournament = league_lower.split()[0] if league_lower else "unknown"
            if tournament in tournaments_used:
                continue
            tournaments_used.add(tournament)

            selected.append(bet)

        return selected
```

### 8.2 Report Generator

```python
# reports/report_generator.py

from datetime import datetime
from typing import List
from pathlib import Path
from ranking.match_ranker import RankedBet

class ReportGenerator:
    """
    Generuje raporty w formatach MD i HTML.
    """

    def generate_markdown(
        self,
        bets: List[RankedBet],
        sport: str,
        date: str
    ) -> str:
        """
        Generuje raport w formacie Markdown.
        """
        if not bets:
            return self._generate_no_bets_report(sport, date)

        lines = [
            f"# 🎯 NEXUS AI - Raport Predykcji",
            f"",
            f"**Sport:** {sport.upper()}  ",
            f"**Data:** {date}  ",
            f"**Wygenerowano:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"---",
            f"",
            f"## 🏆 TOP {len(bets)} VALUE BETS",
            f""
        ]

        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for bet in bets:
            emoji = rank_emoji[bet.rank - 1] if bet.rank <= 5 else f"{bet.rank}."

            lines.extend([
                f"### {emoji} {bet.match_name}",
                f"",
                f"**Liga:** {bet.league}  ",
                f"**Typ:** {bet.selection.upper()}  ",
                f"**Kurs:** {bet.odds:.2f} @ {bet.bookmaker}  ",
                f"**Edge:** +{bet.edge:.1%}  ",
                f"**Jakość danych:** {bet.quality_score:.0f}/100  ",
                f"**Stawka:** {bet.stake_recommendation}",
                f"",
                f"**Uzasadnienie:**",
                f"> {bet.value_reasoning}",
                f""
            ])

            if bet.warnings:
                lines.append(f"**⚠️ Ostrzeżenia:**")
                for w in bet.warnings:
                    lines.append(f"- {w}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Summary
        avg_edge = sum(b.edge for b in bets) / len(bets)
        avg_quality = sum(b.quality_score for b in bets) / len(bets)

        lines.extend([
            f"## 📊 Podsumowanie",
            f"",
            f"- **Znaleziono betów:** {len(bets)}",
            f"- **Średni edge:** {avg_edge:.1%}",
            f"- **Średnia jakość danych:** {avg_quality:.0f}/100",
            f"",
            f"---",
            f"",
            f"*Raport wygenerowany przez NEXUS AI Lite v1.0*"
        ])

        return "\n".join(lines)

    def _generate_no_bets_report(self, sport: str, date: str) -> str:
        """Raport gdy brak betów"""
        return f"""# 🎯 NEXUS AI - Raport Predykcji

**Sport:** {sport.upper()}
**Data:** {date}
**Wygenerowano:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## ❌ Brak Value Betów

System nie znalazł żadnych betów spełniających kryteria jakości i value.

**Możliwe przyczyny:**
- Niewystarczająca jakość danych z internetu
- Brak meczów z dodatnim edge
- Kursy bukmacherów są zbyt efektywne

**Zalecenie:** Spróbuj ponownie później lub sprawdź inny sport.

---

*Raport wygenerowany przez NEXUS AI Lite v1.0*
"""

    def save_report(
        self,
        content: str,
        sport: str,
        date: str,
        output_dir: str = "outputs"
    ) -> str:
        """
        Zapisuje raport do pliku.

        Returns:
            Ścieżka do zapisanego pliku
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        filename = f"raport_{date}_{sport}.md"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return str(filepath)
```
