# evaluator/freshness_checker.py
"""
Data freshness evaluation for NEXUS AI.
Checks recency of data with intelligent date parsing.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class FreshnessLevel(Enum):
    """Freshness classification."""
    LIVE = "live"           # < 5 minutes
    VERY_FRESH = "very_fresh"   # < 1 hour
    FRESH = "fresh"         # < 6 hours
    RECENT = "recent"       # < 24 hours
    STALE = "stale"         # < 48 hours
    OUTDATED = "outdated"   # > 48 hours


@dataclass
class FreshnessResult:
    """Result of freshness evaluation."""
    freshness_level: FreshnessLevel
    freshness_score: float  # 0-1
    age_seconds: float
    age_human_readable: str
    timestamp: Optional[datetime]
    source: str
    is_acceptable: bool
    details: Dict[str, Any]


class FreshnessChecker:
    """
    Evaluates data freshness with intelligent date parsing.

    Features:
    - Parse various date formats
    - Relative time parsing ("5 minutes ago", "yesterday")
    - Timezone-aware comparisons
    - Configurable freshness thresholds
    """

    # Default thresholds in seconds
    DEFAULT_THRESHOLDS = {
        FreshnessLevel.LIVE: 300,         # 5 minutes
        FreshnessLevel.VERY_FRESH: 3600,  # 1 hour
        FreshnessLevel.FRESH: 21600,      # 6 hours
        FreshnessLevel.RECENT: 86400,     # 24 hours
        FreshnessLevel.STALE: 172800,     # 48 hours
    }

    # Minimum acceptable freshness by data type
    MIN_ACCEPTABLE = {
        "odds": FreshnessLevel.VERY_FRESH,
        "news": FreshnessLevel.RECENT,
        "stats": FreshnessLevel.STALE,
        "rankings": FreshnessLevel.STALE,
        "lineups": FreshnessLevel.VERY_FRESH,
        "live_score": FreshnessLevel.LIVE,
        "default": FreshnessLevel.RECENT,
    }

    # Common date patterns
    DATE_PATTERNS = [
        # ISO formats
        (r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)", "%Y-%m-%dT%H:%M:%S"),
        (r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", "%Y-%m-%d %H:%M:%S"),
        (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),

        # European formats
        (r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2})", "%d/%m/%Y %H:%M"),
        (r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})", "%d.%m.%Y %H:%M"),
        (r"(\d{2}/\d{2}/\d{4})", "%d/%m/%Y"),
        (r"(\d{2}\.\d{2}\.\d{4})", "%d.%m.%Y"),

        # US formats
        (r"(\d{2}/\d{2}/\d{4})", "%m/%d/%Y"),

        # Month name formats
        (r"(\w+ \d{1,2}, \d{4} \d{2}:\d{2})", "%B %d, %Y %H:%M"),
        (r"(\w+ \d{1,2}, \d{4})", "%B %d, %Y"),
        (r"(\d{1,2} \w+ \d{4})", "%d %B %Y"),
    ]

    # Relative time patterns
    RELATIVE_PATTERNS = [
        (r"(\d+)\s*(?:seconds?|sec|s)\s*ago", lambda m: timedelta(seconds=int(m))),
        (r"(\d+)\s*(?:minutes?|mins?|m)\s*ago", lambda m: timedelta(minutes=int(m))),
        (r"(\d+)\s*(?:hours?|hrs?|h)\s*ago", lambda m: timedelta(hours=int(m))),
        (r"(\d+)\s*(?:days?|d)\s*ago", lambda m: timedelta(days=int(m))),
        (r"(\d+)\s*(?:weeks?|w)\s*ago", lambda m: timedelta(weeks=int(m))),
        (r"just\s*now", lambda _: timedelta(seconds=30)),
        (r"yesterday", lambda _: timedelta(days=1)),
        (r"today", lambda _: timedelta(hours=0)),
    ]

    def __init__(
        self,
        custom_thresholds: Optional[Dict[FreshnessLevel, int]] = None,
        custom_min_acceptable: Optional[Dict[str, FreshnessLevel]] = None
    ):
        """
        Initialize the checker.

        Args:
            custom_thresholds: Override default thresholds (in seconds)
            custom_min_acceptable: Override minimum acceptable levels by data type
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

        self.min_acceptable = {**self.MIN_ACCEPTABLE}
        if custom_min_acceptable:
            self.min_acceptable.update(custom_min_acceptable)

    def check_freshness(
        self,
        timestamp_str: Union[str, datetime, int, float],
        source: str = "default",
        data_type: str = "default"
    ) -> FreshnessResult:
        """
        Check freshness of data.

        Args:
            timestamp_str: Timestamp in various formats (string, datetime, unix timestamp)
            source: Source identifier for logging
            data_type: Type of data (odds, news, stats, etc.)

        Returns:
            FreshnessResult with evaluation
        """
        # Parse timestamp
        timestamp = self._parse_timestamp(timestamp_str)

        if timestamp is None:
            return self._unknown_freshness(source, data_type)

        # Calculate age
        now = datetime.now()
        if timestamp.tzinfo is not None:
            # Make comparison timezone-aware
            from datetime import timezone
            now = datetime.now(timezone.utc)
            timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp

        age = now - timestamp
        age_seconds = max(0, age.total_seconds())

        # Determine freshness level
        freshness_level = self._get_freshness_level(age_seconds)

        # Calculate score (1.0 = live, 0.0 = very outdated)
        freshness_score = self._calculate_score(age_seconds)

        # Check if acceptable for data type
        min_level = self.min_acceptable.get(data_type, self.min_acceptable["default"])
        is_acceptable = self._is_acceptable(freshness_level, min_level)

        return FreshnessResult(
            freshness_level=freshness_level,
            freshness_score=round(freshness_score, 4),
            age_seconds=age_seconds,
            age_human_readable=self._format_age(age_seconds),
            timestamp=timestamp,
            source=source,
            is_acceptable=is_acceptable,
            details={
                "data_type": data_type,
                "min_acceptable_level": min_level.value,
                "threshold_seconds": self.thresholds.get(freshness_level, float("inf")),
            }
        )

    def check_batch_freshness(
        self,
        items: List[Dict[str, Any]],
        timestamp_key: str = "timestamp",
        data_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Check freshness for multiple items.

        Args:
            items: List of dicts containing timestamps
            timestamp_key: Key for timestamp field
            data_type: Type of data

        Returns:
            Aggregate freshness analysis
        """
        if not items:
            return {
                "item_count": 0,
                "avg_freshness_score": 0.0,
                "freshest": None,
                "oldest": None,
                "acceptable_count": 0,
                "stale_count": 0,
                "by_level": {},
            }

        results = []
        for i, item in enumerate(items):
            ts = item.get(timestamp_key)
            source = item.get("source", f"item_{i}")
            result = self.check_freshness(ts, source, data_type)
            results.append(result)

        # Aggregate statistics
        acceptable = [r for r in results if r.is_acceptable]
        scores = [r.freshness_score for r in results]
        ages = [r.age_seconds for r in results if r.age_seconds is not None]

        # Group by level
        by_level = {}
        for result in results:
            level = result.freshness_level.value
            by_level[level] = by_level.get(level, 0) + 1

        # Find freshest and oldest
        sorted_results = sorted(results, key=lambda r: r.age_seconds or float("inf"))

        return {
            "item_count": len(items),
            "avg_freshness_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "avg_age_seconds": round(sum(ages) / len(ages), 1) if ages else None,
            "avg_age_readable": self._format_age(sum(ages) / len(ages)) if ages else "Unknown",
            "freshest": {
                "source": sorted_results[0].source,
                "age": sorted_results[0].age_human_readable,
                "level": sorted_results[0].freshness_level.value,
            } if sorted_results else None,
            "oldest": {
                "source": sorted_results[-1].source,
                "age": sorted_results[-1].age_human_readable,
                "level": sorted_results[-1].freshness_level.value,
            } if sorted_results else None,
            "acceptable_count": len(acceptable),
            "acceptable_ratio": round(len(acceptable) / len(results), 2) if results else 0,
            "stale_count": by_level.get("stale", 0) + by_level.get("outdated", 0),
            "by_level": by_level,
        }

    def _parse_timestamp(
        self,
        timestamp: Union[str, datetime, int, float, None]
    ) -> Optional[datetime]:
        """Parse various timestamp formats into datetime."""
        if timestamp is None:
            return None

        # Already datetime
        if isinstance(timestamp, datetime):
            return timestamp

        # Unix timestamp
        if isinstance(timestamp, (int, float)):
            try:
                # Handle milliseconds
                if timestamp > 10000000000:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                logger.warning(f"Invalid unix timestamp: {timestamp}")
                return None

        # String parsing
        if isinstance(timestamp, str):
            timestamp = timestamp.strip()

            # Try relative patterns first
            relative_dt = self._parse_relative(timestamp)
            if relative_dt:
                return relative_dt

            # Try absolute patterns
            absolute_dt = self._parse_absolute(timestamp)
            if absolute_dt:
                return absolute_dt

            logger.warning(f"Could not parse timestamp: {timestamp}")
            return None

        return None

    def _parse_relative(self, text: str) -> Optional[datetime]:
        """Parse relative time expressions."""
        text_lower = text.lower()

        for pattern, delta_func in self.RELATIVE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    if match.groups():
                        delta = delta_func(match.group(1))
                    else:
                        delta = delta_func(None)
                    return datetime.now() - delta
                except (ValueError, TypeError):
                    continue

        return None

    def _parse_absolute(self, text: str) -> Optional[datetime]:
        """Parse absolute date/time formats."""
        # Clean timezone suffix for simpler parsing
        text_clean = re.sub(r'Z$', '', text)
        text_clean = re.sub(r'[+-]\d{2}:?\d{2}$', '', text_clean)

        for pattern, date_format in self.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    # Handle microseconds
                    dt_str = match.group(1)
                    dt_str = re.sub(r'\.\d+', '', dt_str)  # Remove microseconds
                    dt_str = re.sub(r'Z$', '', dt_str)
                    dt_str = re.sub(r'[+-]\d{2}:?\d{2}$', '', dt_str)

                    return datetime.strptime(dt_str, date_format)
                except ValueError:
                    continue

        return None

    def _get_freshness_level(self, age_seconds: float) -> FreshnessLevel:
        """Determine freshness level from age."""
        for level, threshold in sorted(
            self.thresholds.items(),
            key=lambda x: x[1]
        ):
            if age_seconds <= threshold:
                return level

        return FreshnessLevel.OUTDATED

    def _calculate_score(self, age_seconds: float) -> float:
        """Calculate freshness score (1.0 = fresh, 0.0 = very old)."""
        # Use exponential decay
        # Score = 0.5 at ~6 hours, ~0.1 at 24 hours
        half_life = 21600  # 6 hours
        score = 0.5 ** (age_seconds / half_life)
        return max(0, min(1, score))

    def _is_acceptable(
        self,
        current: FreshnessLevel,
        minimum: FreshnessLevel
    ) -> bool:
        """Check if current freshness meets minimum requirement."""
        level_order = [
            FreshnessLevel.LIVE,
            FreshnessLevel.VERY_FRESH,
            FreshnessLevel.FRESH,
            FreshnessLevel.RECENT,
            FreshnessLevel.STALE,
            FreshnessLevel.OUTDATED,
        ]

        current_index = level_order.index(current)
        min_index = level_order.index(minimum)

        return current_index <= min_index

    def _format_age(self, age_seconds: float) -> str:
        """Format age in human-readable form."""
        if age_seconds < 60:
            return f"{int(age_seconds)} seconds ago"
        elif age_seconds < 3600:
            minutes = int(age_seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif age_seconds < 86400:
            hours = int(age_seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(age_seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

    def _unknown_freshness(self, source: str, data_type: str) -> FreshnessResult:
        """Return result for unparseable timestamp."""
        return FreshnessResult(
            freshness_level=FreshnessLevel.OUTDATED,
            freshness_score=0.0,
            age_seconds=float("inf"),
            age_human_readable="Unknown",
            timestamp=None,
            source=source,
            is_acceptable=False,
            details={
                "data_type": data_type,
                "error": "Could not parse timestamp",
            }
        )


def check_data_freshness(
    timestamp: Union[str, datetime, int, float],
    data_type: str = "default"
) -> Dict[str, Any]:
    """
    Convenience function to check data freshness.

    Args:
        timestamp: Timestamp in various formats
        data_type: Type of data (odds, news, stats, etc.)

    Returns:
        Freshness evaluation as dict
    """
    checker = FreshnessChecker()
    result = checker.check_freshness(timestamp, "default", data_type)

    return {
        "freshness_level": result.freshness_level.value,
        "freshness_score": result.freshness_score,
        "age_seconds": result.age_seconds,
        "age_readable": result.age_human_readable,
        "timestamp": result.timestamp.isoformat() if result.timestamp else None,
        "is_acceptable": result.is_acceptable,
        "data_type": data_type,
    }
