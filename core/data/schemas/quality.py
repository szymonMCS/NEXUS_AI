from dataclasses import dataclass


@dataclass
class DataQuality:
    """Jakość danych - kluczowe dla unikania halucynacji."""
    completeness: float  # 0.0 - 1.0
    freshness_hours: int  # ile godzin temu dane aktualne
    sources_count: int  # z ilu źródeł
    has_h2h: bool = False
    has_form: bool = False
    has_odds: bool = False

    @property
    def is_sufficient(self) -> bool:
        """Czy dane wystarczające do predykcji."""
        return self.completeness >= 0.6 and self.sources_count >= 1

    @property
    def quality_level(self) -> str:
        """Poziom jakości: excellent/good/moderate/insufficient."""
        if self.completeness >= 0.9 and self.sources_count >= 3:
            return "excellent"
        elif self.completeness >= 0.7 and self.sources_count >= 2:
            return "good"
        elif self.is_sufficient:
            return "moderate"
        return "insufficient"