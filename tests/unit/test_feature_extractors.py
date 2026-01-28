"""
Tests for feature extractors.

Checkpoint: 1.8
Tests for GoalsFeatureExtractor, HandicapFeatureExtractor, FormFeatureExtractor.
"""

import pytest
from datetime import datetime

from core.data.enums import Sport
from core.data.schemas import (
    MatchData,
    TeamData,
    TeamMatchStats,
    HistoricalMatch,
    DataQuality,
    OddsData,
)
from core.ml.features import (
    GoalsFeatureExtractor,
    HandicapFeatureExtractor,
    FormFeatureExtractor,
)


@pytest.fixture
def sample_home_stats():
    return TeamMatchStats(
        goals_scored_avg=1.8,
        goals_conceded_avg=1.0,
        home_goals_avg=2.0,
        away_goals_avg=1.5,
        form_points=0.7,
        rest_days=5,
    )


@pytest.fixture
def sample_away_stats():
    return TeamMatchStats(
        goals_scored_avg=1.3,
        goals_conceded_avg=1.5,
        home_goals_avg=1.5,
        away_goals_avg=1.0,
        form_points=0.5,
        rest_days=3,
    )


@pytest.fixture
def sample_match(sample_home_stats, sample_away_stats):
    return MatchData(
        match_id="test-match-1",
        sport=Sport.FOOTBALL,
        home_team=TeamData(team_id="home-1", name="Home FC", elo_rating=1600),
        away_team=TeamData(team_id="away-1", name="Away FC", elo_rating=1500),
        league="Test League",
        start_time=datetime.utcnow(),
        home_stats=sample_home_stats,
        away_stats=sample_away_stats,
        data_quality=DataQuality(
            completeness=0.8,
            freshness_hours=1,
            sources_count=2,
        ),
    )


@pytest.fixture
def sample_h2h():
    return [
        HistoricalMatch(
            match_id="h2h-1",
            date=datetime(2024, 1, 15),
            home_team_id="home-1",
            away_team_id="away-1",
            home_goals=2,
            away_goals=1,
        ),
        HistoricalMatch(
            match_id="h2h-2",
            date=datetime(2023, 9, 10),
            home_team_id="away-1",
            away_team_id="home-1",
            home_goals=1,
            away_goals=1,
        ),
        HistoricalMatch(
            match_id="h2h-3",
            date=datetime(2023, 3, 5),
            home_team_id="home-1",
            away_team_id="away-1",
            home_goals=3,
            away_goals=2,
        ),
    ]


class TestGoalsFeatureExtractor:
    """Tests for GoalsFeatureExtractor."""

    def test_name(self):
        extractor = GoalsFeatureExtractor()
        assert extractor.name == "goals"

    def test_required_fields(self):
        extractor = GoalsFeatureExtractor()
        assert "home_stats" in extractor.required_fields
        assert "away_stats" in extractor.required_fields

    def test_get_feature_names(self):
        extractor = GoalsFeatureExtractor()
        names = extractor.get_feature_names()
        assert len(names) > 0
        assert "home_goals_scored_avg" in names
        assert "total_expected_goals" in names

    def test_extract_basic(self, sample_match):
        extractor = GoalsFeatureExtractor()
        features = extractor.extract(sample_match)

        assert features["home_goals_scored_avg"] == 1.8
        assert features["away_goals_scored_avg"] == 1.3
        assert features["total_expected_goals"] > 0

    def test_extract_attack_defense_strength(self, sample_match):
        extractor = GoalsFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home team has higher attack
        assert features["home_attack_strength"] > 1.0
        # Away team has weaker defense (higher value = more goals conceded)
        assert features["away_defense_strength"] > features["home_defense_strength"]

    def test_extract_with_h2h(self, sample_match, sample_h2h):
        sample_match.h2h_history = sample_h2h

        extractor = GoalsFeatureExtractor()
        features = extractor.extract(sample_match)

        assert features["h2h_matches_count"] == 3
        # Average total goals: (3 + 2 + 5) / 3 = 3.33
        assert features["h2h_avg_total_goals"] > 2.5
        # 2 out of 3 games were over 2.5
        assert features["h2h_over25_ratio"] == pytest.approx(2/3, rel=0.01)

    def test_extract_safe_missing_data(self):
        extractor = GoalsFeatureExtractor()

        # Match without stats
        match = MatchData(
            match_id="test",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id="h", name="Home"),
            away_team=TeamData(team_id="a", name="Away"),
            league="Test",
            start_time=datetime.utcnow(),
        )

        features = extractor.extract_safe(match)
        assert all(v == 0.0 for v in features.values())

    def test_can_extract(self, sample_match):
        extractor = GoalsFeatureExtractor()
        assert extractor.can_extract(sample_match) is True

        # Without stats
        sample_match.home_stats = None
        assert extractor.can_extract(sample_match) is False

    def test_validate_features(self, sample_match):
        extractor = GoalsFeatureExtractor()
        features = extractor.extract(sample_match)
        assert extractor.validate_features(features) is True

    def test_poisson_probability(self):
        extractor = GoalsFeatureExtractor()
        # P(X=1 | lambda=2) should be around 0.27
        prob = extractor.calculate_poisson_probability(2.0, 1)
        assert 0.2 < prob < 0.3

    def test_over_under_probability(self):
        extractor = GoalsFeatureExtractor()
        probs = extractor.calculate_over_under_prob(1.5, 1.5, threshold=2.5)

        assert "over" in probs
        assert "under" in probs
        assert abs(probs["over"] + probs["under"] - 1.0) < 0.001


class TestHandicapFeatureExtractor:
    """Tests for HandicapFeatureExtractor."""

    def test_name(self):
        extractor = HandicapFeatureExtractor()
        assert extractor.name == "handicap"

    def test_get_feature_names(self):
        extractor = HandicapFeatureExtractor()
        names = extractor.get_feature_names()
        assert "attack_diff" in names
        assert "expected_margin" in names
        assert "form_diff" in names

    def test_extract_strength_diffs(self, sample_match):
        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home team is stronger
        assert features["attack_diff"] > 0
        assert features["overall_strength_diff"] > 0

    def test_extract_form_features(self, sample_match):
        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        assert features["home_form"] == 0.7
        assert features["away_form"] == 0.5
        assert features["form_diff"] == 0.2

    def test_extract_with_h2h_margin(self, sample_match, sample_h2h):
        sample_match.h2h_history = sample_h2h

        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home team has positive H2H record
        assert features["h2h_home_win_ratio"] > features["h2h_away_win_ratio"]

    def test_extract_elo_features(self, sample_match):
        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home team has higher ELO (1600 vs 1500)
        assert features["elo_diff"] > 0

    def test_extract_with_odds(self, sample_match):
        sample_match.odds = OddsData(
            home_win=1.80,
            draw=3.50,
            away_win=4.20,
        )

        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home implied prob should be higher
        assert features["odds_implied_home_prob"] > features["odds_implied_away_prob"]

    def test_blowout_and_close_game(self, sample_match):
        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        assert 0 <= features["blowout_risk"] <= 1
        assert 0 <= features["close_game_prob"] <= 1

    def test_predict_cover_probability(self, sample_match):
        extractor = HandicapFeatureExtractor()
        features = extractor.extract(sample_match)

        probs = extractor.predict_cover_probability(features, handicap_line=-1.5)
        assert "home_cover" in probs
        assert "away_cover" in probs
        assert abs(probs["home_cover"] + probs["away_cover"] - 1.0) < 0.001


class TestFormFeatureExtractor:
    """Tests for FormFeatureExtractor."""

    def test_name(self):
        extractor = FormFeatureExtractor()
        assert extractor.name == "form"

    def test_get_feature_names(self):
        extractor = FormFeatureExtractor()
        names = extractor.get_feature_names()
        assert "home_form_points" in names
        assert "momentum_diff" in names

    def test_extract_basic_form(self, sample_match):
        extractor = FormFeatureExtractor()
        features = extractor.extract(sample_match)

        assert features["home_form_points"] == 0.7
        assert features["away_form_points"] == 0.5
        assert features["form_points_diff"] == 0.2

    def test_extract_rest_normalized(self, sample_match):
        extractor = FormFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home: 5 days, Away: 3 days
        assert features["home_rest_days_normalized"] == pytest.approx(5/7, rel=0.01)
        assert features["away_rest_days_normalized"] == pytest.approx(3/7, rel=0.01)

    def test_momentum_calculation(self, sample_match):
        extractor = FormFeatureExtractor()
        features = extractor.extract(sample_match)

        # Home has better form and more rest
        assert features["home_momentum"] > features["away_momentum"]
        assert features["momentum_diff"] > 0

    def test_overall_form(self, sample_match):
        extractor = FormFeatureExtractor()
        features = extractor.extract(sample_match)

        assert 0 <= features["home_overall_form"] <= 1
        assert 0 <= features["away_overall_form"] <= 1


class TestExtractorIntegration:
    """Integration tests for all extractors working together."""

    def test_all_extractors_same_match(self, sample_match, sample_h2h):
        sample_match.h2h_history = sample_h2h
        sample_match.odds = OddsData(home_win=1.80, away_win=4.00)

        goals = GoalsFeatureExtractor()
        handicap = HandicapFeatureExtractor()
        form = FormFeatureExtractor()

        goals_features = goals.extract(sample_match)
        handicap_features = handicap.extract(sample_match)
        form_features = form.extract(sample_match)

        # All should extract successfully
        assert len(goals_features) > 0
        assert len(handicap_features) > 0
        assert len(form_features) > 0

        # No NaN values
        assert goals.validate_features(goals_features)
        assert handicap.validate_features(handicap_features)
        assert form.validate_features(form_features)

    def test_feature_count_consistency(self):
        """Ensure feature names match actual features extracted."""
        goals = GoalsFeatureExtractor()
        handicap = HandicapFeatureExtractor()
        form = FormFeatureExtractor()

        match = MatchData(
            match_id="test",
            sport=Sport.FOOTBALL,
            home_team=TeamData(team_id="h", name="Home"),
            away_team=TeamData(team_id="a", name="Away"),
            league="Test",
            start_time=datetime.utcnow(),
            home_stats=TeamMatchStats(goals_scored_avg=1.5, goals_conceded_avg=1.0),
            away_stats=TeamMatchStats(goals_scored_avg=1.2, goals_conceded_avg=1.3),
        )

        for extractor in [goals, handicap, form]:
            names = extractor.get_feature_names()
            features = extractor.extract(match)
            assert len(names) == len(features), f"{extractor.name} has mismatched feature count"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
