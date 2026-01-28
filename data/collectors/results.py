"""
Collection result dataclasses.

Checkpoint: 5.1
Responsibility: Data structures for historical data collection results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

from core.data.enums import Sport


class CollectionStatus(Enum):
    """Status of a collection operation."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some data collected, some errors
    FAILED = "failed"
    NO_DATA = "no_data"  # No errors but no data found


@dataclass
class SourceResult:
    """Result from a single data source."""
    source_name: str
    status: CollectionStatus
    records_collected: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.records_collected + self.records_failed
        if total == 0:
            return 0.0
        return self.records_collected / total


@dataclass
class CollectedMatch:
    """
    A single collected historical match.

    Minimal structure that can be converted to TrainingExample.
    """
    # Identifiers
    match_id: str
    source: str

    # Basic info
    sport: Sport
    league: str
    season: str
    match_date: datetime

    # Teams
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str

    # Results (required for training)
    home_goals: int
    away_goals: int

    # Optional: Half-time scores
    home_goals_ht: Optional[int] = None
    away_goals_ht: Optional[int] = None

    # Optional: Additional stats for features
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None

    # Optional: Odds at kickoff
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    odds_over_25: Optional[float] = None
    odds_under_25: Optional[float] = None

    # Metadata
    collected_at: datetime = field(default_factory=datetime.utcnow)
    raw_data: Optional[Dict[str, Any]] = None

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def goal_difference(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def result(self) -> str:
        """H/D/A result."""
        if self.home_goals > self.away_goals:
            return "H"
        elif self.home_goals < self.away_goals:
            return "A"
        return "D"

    @property
    def is_over_25(self) -> bool:
        return self.total_goals > 2.5

    @property
    def btts(self) -> bool:
        """Both teams to score."""
        return self.home_goals > 0 and self.away_goals > 0

    @property
    def has_stats(self) -> bool:
        """Check if match has detailed statistics."""
        return self.home_shots is not None or self.home_possession is not None

    @property
    def has_odds(self) -> bool:
        """Check if match has odds data."""
        return self.odds_home is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "match_id": self.match_id,
            "source": self.source,
            "sport": self.sport.value,
            "league": self.league,
            "season": self.season,
            "match_date": self.match_date.isoformat(),
            "home_team_id": self.home_team_id,
            "home_team_name": self.home_team_name,
            "away_team_id": self.away_team_id,
            "away_team_name": self.away_team_name,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_goals_ht": self.home_goals_ht,
            "away_goals_ht": self.away_goals_ht,
            "home_shots": self.home_shots,
            "away_shots": self.away_shots,
            "home_shots_on_target": self.home_shots_on_target,
            "away_shots_on_target": self.away_shots_on_target,
            "home_possession": self.home_possession,
            "away_possession": self.away_possession,
            "home_corners": self.home_corners,
            "away_corners": self.away_corners,
            "odds_home": self.odds_home,
            "odds_draw": self.odds_draw,
            "odds_away": self.odds_away,
            "odds_over_25": self.odds_over_25,
            "odds_under_25": self.odds_under_25,
            "collected_at": self.collected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectedMatch":
        """Create from dictionary."""
        return cls(
            match_id=data["match_id"],
            source=data["source"],
            sport=Sport(data["sport"]),
            league=data["league"],
            season=data["season"],
            match_date=datetime.fromisoformat(data["match_date"]),
            home_team_id=data["home_team_id"],
            home_team_name=data["home_team_name"],
            away_team_id=data["away_team_id"],
            away_team_name=data["away_team_name"],
            home_goals=data["home_goals"],
            away_goals=data["away_goals"],
            home_goals_ht=data.get("home_goals_ht"),
            away_goals_ht=data.get("away_goals_ht"),
            home_shots=data.get("home_shots"),
            away_shots=data.get("away_shots"),
            home_shots_on_target=data.get("home_shots_on_target"),
            away_shots_on_target=data.get("away_shots_on_target"),
            home_possession=data.get("home_possession"),
            away_possession=data.get("away_possession"),
            home_corners=data.get("home_corners"),
            away_corners=data.get("away_corners"),
            odds_home=data.get("odds_home"),
            odds_draw=data.get("odds_draw"),
            odds_away=data.get("odds_away"),
            odds_over_25=data.get("odds_over_25"),
            odds_under_25=data.get("odds_under_25"),
            collected_at=datetime.fromisoformat(data["collected_at"]) if "collected_at" in data else datetime.utcnow(),
        )


@dataclass
class CollectionResult:
    """
    Result of a historical data collection operation.

    Aggregates results from multiple sources.
    """
    # Identification
    collection_id: str
    sport: Sport
    league: str

    # Time range
    start_date: datetime
    end_date: datetime

    # Overall status
    status: CollectionStatus = CollectionStatus.SUCCESS

    # Results
    matches: List[CollectedMatch] = field(default_factory=list)
    source_results: List[SourceResult] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def total_collected(self) -> int:
        return len(self.matches)

    @property
    def total_errors(self) -> int:
        return len(self.errors)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def sources_used(self) -> List[str]:
        return [sr.source_name for sr in self.source_results]

    @property
    def matches_with_stats(self) -> int:
        return sum(1 for m in self.matches if m.has_stats)

    @property
    def matches_with_odds(self) -> int:
        return sum(1 for m in self.matches if m.has_odds)

    def add_match(self, match: CollectedMatch) -> None:
        """Add a collected match."""
        self.matches.append(match)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_source_result(self, result: SourceResult) -> None:
        """Add a source result."""
        self.source_results.append(result)

    def finalize(self) -> None:
        """Mark collection as complete and determine final status."""
        self.completed_at = datetime.utcnow()

        if not self.matches and not self.errors:
            self.status = CollectionStatus.NO_DATA
        elif not self.matches and self.errors:
            self.status = CollectionStatus.FAILED
        elif self.matches and self.errors:
            self.status = CollectionStatus.PARTIAL
        else:
            self.status = CollectionStatus.SUCCESS

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the collection."""
        return {
            "collection_id": self.collection_id,
            "sport": self.sport.value,
            "league": self.league,
            "date_range": f"{self.start_date.date()} to {self.end_date.date()}",
            "status": self.status.value,
            "total_matches": self.total_collected,
            "matches_with_stats": self.matches_with_stats,
            "matches_with_odds": self.matches_with_odds,
            "sources_used": self.sources_used,
            "errors": self.total_errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class CollectionConfig:
    """Configuration for data collection."""
    # Sports and leagues to collect
    sports: List[Sport] = field(default_factory=lambda: [Sport.FOOTBALL])
    leagues: Dict[Sport, List[str]] = field(default_factory=dict)

    # Time range
    seasons: List[str] = field(default_factory=lambda: ["2023", "2024"])

    # Data requirements
    require_stats: bool = False  # Require detailed match stats
    require_odds: bool = False   # Require odds data
    min_matches_per_league: int = 100

    # Rate limiting
    requests_per_minute: int = 30
    delay_between_leagues: float = 2.0  # seconds

    # Storage
    save_raw_data: bool = False
    output_format: str = "json"  # json, csv, parquet

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0


# Default leagues for each sport
DEFAULT_LEAGUES = {
    Sport.FOOTBALL: [
        "PL",         # Premier League
        "LaLiga",     # La Liga
        "SerieA",     # Serie A
        "Bundesliga", # Bundesliga
        "Ligue1",     # Ligue 1
    ],
    Sport.BASKETBALL: [
        "NBA",
        "EuroLeague",
    ],
    Sport.TENNIS: [
        "ATP",
        "WTA",
    ],
}
