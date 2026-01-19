## 2. ARCHITEKTURA WEB-FIRST

### 2.1 Filozofia: Internet jako baza danych

Zamiast polegać na płatnych API, traktujemy internet jako źródło prawdy:
- **Web Search** (Brave/Serper) → newsy, kontuzje, forma
- **Web Scraping** (Sofascore, Flashscore) → statystyki, kursy
- **Darmowe API** (TheSportsDB, API-Sports free tier) → fixtures, podstawowe dane

### 2.2 Ewaluator Jakości Danych Web

**KLUCZOWY KOMPONENT** - Ponieważ dane z internetu mogą być niespójne/nieaktualne:

```python
class WebDataEvaluator:
    """
    Ocenia jakość danych zebranych z internetu.

    Kryteria:
    1. SOURCE_AGREEMENT - Czy różne źródła zgadzają się?
    2. FRESHNESS - Jak stare są dane?
    3. COMPLETENESS - Czy mamy wszystkie potrzebne pola?
    4. RELIABILITY - Czy źródło jest wiarygodne?
    """

    def evaluate(self, match_data: dict) -> QualityReport:
        scores = {
            "source_agreement": self._check_source_agreement(match_data),
            "freshness": self._check_freshness(match_data),
            "completeness": self._check_completeness(match_data),
            "reliability": self._check_source_reliability(match_data)
        }

        overall = weighted_average(scores, weights={
            "source_agreement": 0.35,  # Najważniejsze!
            "freshness": 0.25,
            "completeness": 0.25,
            "reliability": 0.15
        })

        return QualityReport(
            overall_score=overall,
            is_trustworthy=overall >= 0.5,
            issues=self._identify_issues(scores)
        )
```
