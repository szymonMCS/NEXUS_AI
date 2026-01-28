"""
Training example dataclasses.

Checkpoint: 3.4
Responsibility: Structure for training data and examples.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List


@dataclass
class TrainingExample:
    """
    Pojedynczy przykład treningowy dla modeli ML.

    Łączy cechy z rzeczywistym wynikiem meczu.
    """
    # Identyfikacja
    example_id: str
    match_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Cechy (zapisane jako dict dla elastyczności)
    features: Dict[str, float] = field(default_factory=dict)

    # Rzeczywiste wyniki
    actual_home_goals: Optional[int] = None
    actual_away_goals: Optional[int] = None

    # Derived targets
    @property
    def actual_total_goals(self) -> Optional[int]:
        if self.actual_home_goals is not None and self.actual_away_goals is not None:
            return self.actual_home_goals + self.actual_away_goals
        return None

    @property
    def actual_margin(self) -> Optional[int]:
        """Home goals - away goals."""
        if self.actual_home_goals is not None and self.actual_away_goals is not None:
            return self.actual_home_goals - self.actual_away_goals
        return None

    @property
    def actual_winner(self) -> Optional[str]:
        margin = self.actual_margin
        if margin is None:
            return None
        if margin > 0:
            return "home"
        elif margin < 0:
            return "away"
        return "draw"

    @property
    def is_over_25(self) -> Optional[bool]:
        total = self.actual_total_goals
        return total > 2.5 if total is not None else None

    @property
    def is_complete(self) -> bool:
        """Check if example has all required data."""
        return (
            self.actual_home_goals is not None and
            self.actual_away_goals is not None and
            len(self.features) > 0
        )

    def to_dict(self) -> Dict:
        return {
            "example_id": self.example_id,
            "match_id": self.match_id,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "actual_home_goals": self.actual_home_goals,
            "actual_away_goals": self.actual_away_goals,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingExample":
        return cls(
            example_id=data["example_id"],
            match_id=data["match_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            features=data.get("features", {}),
            actual_home_goals=data.get("actual_home_goals"),
            actual_away_goals=data.get("actual_away_goals"),
        )


@dataclass
class TrainingBatch:
    """
    Batch of training examples.
    """
    examples: List[TrainingExample] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    source: str = ""  # Where the data came from

    def add(self, example: TrainingExample) -> None:
        self.examples.append(example)

    def get_complete_examples(self) -> List[TrainingExample]:
        """Return only complete examples."""
        return [e for e in self.examples if e.is_complete]

    def get_features_matrix(self, feature_names: List[str]) -> List[List[float]]:
        """Convert to feature matrix."""
        matrix = []
        for ex in self.get_complete_examples():
            row = [ex.features.get(name, 0.0) for name in feature_names]
            matrix.append(row)
        return matrix

    def get_feature_vectors(self) -> List["FeatureVector"]:
        """Convert examples to FeatureVector objects for model training."""
        from core.ml.features.vector import FeatureVector
        from core.data.schemas.quality import DataQuality

        vectors = []
        for ex in self.get_complete_examples():
            vector = FeatureVector(
                features=ex.features,
                match_id=ex.match_id,
                quality=DataQuality(
                    completeness=0.8,
                    freshness_hours=24,
                    sources_count=1,
                    has_h2h=False,
                    has_form=True,
                    has_odds=False,
                ),
            )
            vectors.append(vector)
        return vectors

    def get_goals_targets(self) -> List[tuple]:
        """Get (home_goals, away_goals) targets."""
        return [
            (ex.actual_home_goals, ex.actual_away_goals)
            for ex in self.get_complete_examples()
        ]

    def get_margin_targets(self) -> List[float]:
        """Get margin targets."""
        return [
            float(ex.actual_margin)
            for ex in self.get_complete_examples()
            if ex.actual_margin is not None
        ]

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    validation_split: float = 0.2
    min_samples: int = 100
    max_samples: Optional[int] = None
    feature_names: List[str] = field(default_factory=list)

    # Online learning
    incremental: bool = False
    learning_rate_decay: float = 0.99

    # Early stopping
    patience: int = 5
    min_improvement: float = 0.001
