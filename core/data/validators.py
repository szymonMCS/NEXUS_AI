"""
Data validators for NEXUS ML.

Checkpoint: 0.7
Responsibility: Validate data quality and detect edge cases.
Principle: NO HALLUCINATION - reject insufficient data instead of guessing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import math

from core.data.schemas import (
    DataQuality,
    MatchData,
    TeamMatchStats,
    HistoricalMatch,
    OddsData,
)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float  # 0.0 - 1.0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class DataValidator:
    """
    Validates match data for ML prediction.

    Key principle: Better to REJECT insufficient data than to HALLUCINATE.
    """

    # Thresholds
    MIN_COMPLETENESS = 0.6
    MIN_SOURCES = 1
    MAX_FRESHNESS_HOURS = 24
    MIN_H2H_MATCHES = 2
    MIN_FORM_MATCHES = 3
    MAX_GOALS_AVG = 10.0  # Sanity check
    MIN_ODDS = 1.01
    MAX_ODDS = 100.0

    def validate_match_data(self, match: MatchData) -> ValidationResult:
        """
        Validate complete match data.

        Returns ValidationResult with errors, warnings, and quality score.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Basic validation
        if not match.match_id:
            errors.append("Missing match_id")

        if not match.home_team or not match.away_team:
            errors.append("Missing team data")

        if not match.league:
            warnings.append("Missing league information")

        # Time validation
        if match.start_time:
            if match.start_time < datetime.utcnow() - timedelta(hours=1):
                warnings.append("Match may have already started")
        else:
            errors.append("Missing start_time")

        # Stats validation
        if match.home_stats:
            stats_result = self.validate_team_stats(match.home_stats, "home")
            errors.extend(stats_result.errors)
            warnings.extend(stats_result.warnings)
        else:
            warnings.append("Missing home team stats")

        if match.away_stats:
            stats_result = self.validate_team_stats(match.away_stats, "away")
            errors.extend(stats_result.errors)
            warnings.extend(stats_result.warnings)
        else:
            warnings.append("Missing away team stats")

        # H2H validation
        if match.h2h_history:
            h2h_result = self.validate_h2h_history(match.h2h_history)
            errors.extend(h2h_result.errors)
            warnings.extend(h2h_result.warnings)
        else:
            warnings.append("No H2H history available")

        # Odds validation
        if match.odds:
            odds_result = self.validate_odds(match.odds)
            errors.extend(odds_result.errors)
            warnings.extend(odds_result.warnings)

        # Data quality validation
        quality_result = self.validate_data_quality(match.data_quality)
        errors.extend(quality_result.errors)
        warnings.extend(quality_result.warnings)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(match, errors, warnings)

        return ValidationResult(
            is_valid=len(errors) == 0 and quality_score >= self.MIN_COMPLETENESS,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
        )

    def validate_team_stats(
        self, stats: TeamMatchStats, team_type: str
    ) -> ValidationResult:
        """Validate team statistics."""
        errors: List[str] = []
        warnings: List[str] = []
        prefix = f"{team_type}_team"

        # Goals average sanity check
        if stats.goals_scored_avg < 0:
            errors.append(f"{prefix}: negative goals_scored_avg")
        elif stats.goals_scored_avg > self.MAX_GOALS_AVG:
            warnings.append(f"{prefix}: unusually high goals_scored_avg ({stats.goals_scored_avg})")

        if stats.goals_conceded_avg < 0:
            errors.append(f"{prefix}: negative goals_conceded_avg")
        elif stats.goals_conceded_avg > self.MAX_GOALS_AVG:
            warnings.append(f"{prefix}: unusually high goals_conceded_avg")

        # Form validation
        if stats.form_points < 0 or stats.form_points > 1:
            errors.append(f"{prefix}: form_points must be 0.0-1.0, got {stats.form_points}")

        # Rest days sanity
        if stats.rest_days < 0:
            errors.append(f"{prefix}: negative rest_days")
        elif stats.rest_days > 30:
            warnings.append(f"{prefix}: unusually long rest period ({stats.rest_days} days)")

        # NaN/Inf check
        for field_name, value in [
            ("goals_scored_avg", stats.goals_scored_avg),
            ("goals_conceded_avg", stats.goals_conceded_avg),
            ("form_points", stats.form_points),
        ]:
            if math.isnan(value) or math.isinf(value):
                errors.append(f"{prefix}: {field_name} is NaN or Inf")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=1.0 if len(errors) == 0 else 0.0,
        )

    def validate_h2h_history(
        self, h2h: List[HistoricalMatch]
    ) -> ValidationResult:
        """Validate head-to-head history."""
        errors: List[str] = []
        warnings: List[str] = []

        if len(h2h) < self.MIN_H2H_MATCHES:
            warnings.append(f"Only {len(h2h)} H2H matches (recommended: {self.MIN_H2H_MATCHES}+)")

        for i, match in enumerate(h2h):
            # Goals sanity
            if match.home_goals < 0 or match.away_goals < 0:
                errors.append(f"H2H match {i}: negative goals")

            if match.home_goals > 15 or match.away_goals > 15:
                warnings.append(f"H2H match {i}: unusually high score ({match.home_goals}-{match.away_goals})")

            # Date validation
            if match.date > datetime.utcnow():
                errors.append(f"H2H match {i}: future date")

            # Very old matches
            if match.date < datetime.utcnow() - timedelta(days=365 * 3):
                warnings.append(f"H2H match {i}: older than 3 years")

        # Check for duplicates
        match_ids = [m.match_id for m in h2h]
        if len(match_ids) != len(set(match_ids)):
            warnings.append("Duplicate matches in H2H history")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=min(1.0, len(h2h) / 5),  # 5 matches = full score
        )

    def validate_odds(self, odds: OddsData) -> ValidationResult:
        """Validate betting odds."""
        errors: List[str] = []
        warnings: List[str] = []

        # Basic odds validation
        for name, value in [
            ("home_win", odds.home_win),
            ("away_win", odds.away_win),
        ]:
            if value is not None:
                if value < self.MIN_ODDS:
                    errors.append(f"Odds {name} too low: {value}")
                elif value > self.MAX_ODDS:
                    warnings.append(f"Odds {name} unusually high: {value}")

        # Over/under validation
        if odds.over_25 is not None and odds.under_25 is not None:
            # Check implied probability sums to ~100% (with margin)
            implied_over = 1 / odds.over_25
            implied_under = 1 / odds.under_25
            total = implied_over + implied_under

            if total < 0.95 or total > 1.15:
                warnings.append(f"Over/under implied probability unusual: {total:.2%}")

        # Handicap validation
        if odds.handicap_line is not None:
            if abs(odds.handicap_line) > 5:
                warnings.append(f"Unusual handicap line: {odds.handicap_line}")

        # Freshness
        age_hours = (datetime.utcnow() - odds.timestamp).total_seconds() / 3600
        if age_hours > self.MAX_FRESHNESS_HOURS:
            warnings.append(f"Odds are {age_hours:.1f} hours old")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=1.0 if len(errors) == 0 and len(warnings) == 0 else 0.8,
        )

    def validate_data_quality(self, quality: DataQuality) -> ValidationResult:
        """Validate data quality metrics."""
        errors: List[str] = []
        warnings: List[str] = []

        if quality.completeness < 0 or quality.completeness > 1:
            errors.append(f"Invalid completeness: {quality.completeness}")

        if quality.sources_count < self.MIN_SOURCES:
            warnings.append(f"Only {quality.sources_count} data source(s)")

        if quality.freshness_hours > self.MAX_FRESHNESS_HOURS:
            warnings.append(f"Data is {quality.freshness_hours} hours old")

        if not quality.is_sufficient:
            warnings.append("Data quality marked as insufficient")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality.completeness,
        )

    def validate_for_prediction(self, match: MatchData) -> Tuple[bool, str]:
        """
        Quick check if match data is sufficient for ML prediction.

        Returns (can_predict, reason).
        Use this before running expensive ML models.
        """
        result = self.validate_match_data(match)

        if result.has_errors:
            return False, f"Validation errors: {'; '.join(result.errors)}"

        if result.quality_score < self.MIN_COMPLETENESS:
            return False, f"Quality too low: {result.quality_score:.2f} < {self.MIN_COMPLETENESS}"

        if not match.data_quality.is_sufficient:
            return False, "Data quality marked as insufficient"

        return True, "OK"

    def _calculate_quality_score(
        self,
        match: MatchData,
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate overall quality score for match data."""
        if errors:
            return 0.0

        score = 0.0
        max_score = 0.0

        # Base completeness from data_quality
        score += match.data_quality.completeness * 0.3
        max_score += 0.3

        # Team stats
        if match.home_stats:
            score += 0.15
        max_score += 0.15

        if match.away_stats:
            score += 0.15
        max_score += 0.15

        # H2H
        if match.h2h_history and len(match.h2h_history) >= self.MIN_H2H_MATCHES:
            score += 0.2
        max_score += 0.2

        # Odds
        if match.odds:
            score += 0.1
        max_score += 0.1

        # Form data
        if match.data_quality.has_form:
            score += 0.1
        max_score += 0.1

        # Penalty for warnings
        warning_penalty = len(warnings) * 0.02
        score = max(0, score - warning_penalty)

        return min(1.0, score / max_score) if max_score > 0 else 0.0


# Singleton instance for convenience
default_validator = DataValidator()


def validate_match(match: MatchData) -> ValidationResult:
    """Convenience function for quick validation."""
    return default_validator.validate_match_data(match)


def can_predict(match: MatchData) -> Tuple[bool, str]:
    """Check if match data is sufficient for prediction."""
    return default_validator.validate_for_prediction(match)
