from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from core.data.enums import Sport
from core.data.schemas.quality import DataQuality
from core.data.schemas.team_stats import TeamMatchStats
from core.data.schemas.historical import HistoricalMatch


@dataclass
class TeamData:
    """Podstawowe dane drużyny."""
    team_id: str
    name: str
    ranking: Optional[int] = None
    elo_rating: Optional[float] = None


@dataclass
class OddsData:
    """Kursy bukmacherskie."""
    home_win: float
    draw: Optional[float] = None
    away_win: float = 0.0
    over_25: Optional[float] = None
    under_25: Optional[float] = None
    handicap_line: Optional[float] = None
    handicap_home: Optional[float] = None
    handicap_away: Optional[float] = None
    bookmaker: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MatchData:
    """Zunifikowana struktura meczu - JEDNO ŹRÓDŁO PRAWDY."""
    match_id: str
    sport: Sport
    home_team: TeamData
    away_team: TeamData
    league: str
    start_time: datetime

    # Statystyki (wypełniane stopniowo)
    home_stats: Optional[TeamMatchStats] = None
    away_stats: Optional[TeamMatchStats] = None
    h2h_history: Optional[List[HistoricalMatch]] = None
    odds: Optional[OddsData] = None

    # Jakość danych
    data_quality: DataQuality = field(default_factory=lambda: DataQuality(
        completeness=0.0, freshness_hours=0, sources_count=0
    ))
