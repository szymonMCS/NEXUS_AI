"""
Base class for sports data sources.

Checkpoint: Dataset Integration
Responsibility: Define interface for all sports dataset integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class DatasetQuality(str, Enum):
    """Quality rating for datasets."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class DatasetLicense(str, Enum):
    """License types for datasets."""
    OPEN = "open"
    ACADEMIC = "academic"
    COMMERCIAL = "commercial"
    RESTRICTED = "restricted"


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""
    name: str
    description: str
    source_url: str
    sport: str
    quality: DatasetQuality
    license: DatasetLicense
    formats: List[str] = field(default_factory=list)
    size_mb: Optional[float] = None
    record_count: Optional[int] = None
    update_frequency: str = "unknown"
    last_updated: Optional[datetime] = None
    features: List[str] = field(default_factory=list)
    has_player_tracking: bool = False
    has_play_by_play: bool = False
    has_injury_data: bool = False
    requires_api_key: bool = False
    rate_limit: Optional[str] = None


@dataclass
class RawMatch:
    """Raw match data from any source."""
    match_id: str
    source: str
    sport: str
    home_team: str
    away_team: str
    home_player_id: Optional[str] = None
    away_player_id: Optional[str] = None
    match_date: datetime
    season: Optional[str] = None
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    result: Optional[str] = None
    venue: Optional[str] = None
    attendance: Optional[int] = None
    weather: Optional[Dict[str, Any]] = None
    home_stats: Dict[str, Any] = field(default_factory=dict)
    away_stats: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    odds: Dict[str, float] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=datetime.utcnow)


class SportsDataSource(ABC):
    """Abstract base class for sports data sources."""
    
    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return dataset metadata."""
        pass
    
    @abstractmethod
    async def fetch_matches(
        self,
        start_date: datetime,
        end_date: datetime,
        league: Optional[str] = None,
    ) -> List[RawMatch]:
        """Fetch matches from the dataset."""
        pass
    
    @abstractmethod
    async def fetch_player_stats(
        self,
        player_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch player statistics."""
        pass
    
    @abstractmethod
    async def fetch_team_stats(
        self,
        team_id: str,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch team statistics."""
        pass
    
    def validate_data(self, match: RawMatch) -> bool:
        """Validate raw match data."""
        required = [match.match_id, match.home_team, match.away_team, match.match_date]
        return all(required)
    
    def preprocess(self, match: RawMatch) -> RawMatch:
        """Preprocess raw match data. Override for custom preprocessing."""
        return match
