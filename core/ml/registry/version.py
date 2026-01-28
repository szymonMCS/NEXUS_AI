"""
Model version dataclass.

Checkpoint: 3.1
Responsibility: Track model versions and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ModelVersion:
    """
    Wersja modelu ML z metadanymi.

    Przechowuje informacje o wytrenowanym modelu:
    - Identyfikator i wersja
    - Metryki wydajności
    - Ścieżka do zapisanego modelu
    - Historia zmian
    """
    name: str
    version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = field(default_factory=dict)
    path: Optional[Path] = None

    # Training info
    training_samples: int = 0
    training_duration_seconds: float = 0.0
    feature_names: List[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None  # For incremental training

    # Status
    is_active: bool = False  # Currently used for predictions
    is_deprecated: bool = False

    @property
    def full_name(self) -> str:
        """Full model identifier."""
        return f"{self.name}:{self.version}"

    @property
    def age_hours(self) -> float:
        """How old is this version in hours."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 3600

    @property
    def is_recent(self) -> bool:
        """Is this version less than 24 hours old."""
        return self.age_hours < 24

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a specific metric."""
        return self.metrics.get(name, default)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "path": str(self.path) if self.path else None,
            "training_samples": self.training_samples,
            "training_duration_seconds": self.training_duration_seconds,
            "feature_names": self.feature_names,
            "description": self.description,
            "tags": self.tags,
            "parent_version": self.parent_version,
            "is_active": self.is_active,
            "is_deprecated": self.is_deprecated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metrics=data.get("metrics", {}),
            path=Path(data["path"]) if data.get("path") else None,
            training_samples=data.get("training_samples", 0),
            training_duration_seconds=data.get("training_duration_seconds", 0.0),
            feature_names=data.get("feature_names", []),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            parent_version=data.get("parent_version"),
            is_active=data.get("is_active", False),
            is_deprecated=data.get("is_deprecated", False),
        )


@dataclass
class VersionComparison:
    """Comparison between two model versions."""
    old_version: ModelVersion
    new_version: ModelVersion
    metric_changes: Dict[str, float] = field(default_factory=dict)  # metric -> delta
    is_improvement: bool = False
    comparison_metric: str = "accuracy"

    def __post_init__(self):
        """Calculate metric changes."""
        for metric in set(self.old_version.metrics.keys()) | set(self.new_version.metrics.keys()):
            old_val = self.old_version.get_metric(metric)
            new_val = self.new_version.get_metric(metric)
            self.metric_changes[metric] = new_val - old_val

        # Determine if improvement (assuming higher is better for comparison_metric)
        if self.comparison_metric in self.metric_changes:
            self.is_improvement = self.metric_changes[self.comparison_metric] > 0
