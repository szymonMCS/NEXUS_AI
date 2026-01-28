from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class HistoricalMatch:
    """Historyczny mecz do treningu ML."""
    match_id: str
    date: datetime
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    league: str = ""
    season: str = ""
    home_goals_ht: Optional[int] = None  # pierwsza połowa
    away_goals_ht: Optional[int] = None

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def goal_diff(self) -> int:
        """Różnica z perspektywy gospodarza."""
        return self.home_goals - self.away_goals

    @property
    def is_over_25(self) -> bool:
        return self.total_goals > 2.5

    @property
    def home_win(self) -> bool:
        return self.home_goals > self.away_goals

    @property
    def away_win(self) -> bool:
        return self.away_goals > self.home_goals

    @property
    def draw(self) -> bool:
        return self.home_goals == self.away_goals