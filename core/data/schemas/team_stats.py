from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamMatchStats:
    """Statystyki drużyny do meczu."""
    goals_scored_avg: float  # średnia bramek strzelonych
    goals_conceded_avg: float  # średnia bramek straconych
    home_goals_avg: Optional[float] = None  # tylko dom
    away_goals_avg: Optional[float] = None  # tylko wyjazd
    form_points: float = 0.0  # forma 0.0-1.0
    rest_days: int = 0  # dni od ostatniego meczu

    @property
    def attack_strength(self) -> float:
        """Siła ataku względem średniej (1.0 = średnia)."""
        league_avg = 1.3  # TODO: dynamicznie
        return self.goals_scored_avg / league_avg if league_avg > 0 else 1.0

    @property
    def defense_strength(self) -> float:
        """Siła obrony (niższe = lepsze)."""
        league_avg = 1.3
        return self.goals_conceded_avg / league_avg if league_avg > 0 else 1.0
