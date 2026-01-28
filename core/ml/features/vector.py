"""
Feature vector for ML models.

Checkpoint: 1.1
Responsibility: Unified feature representation for all ML models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from core.data.schemas import DataQuality


@dataclass
class FeatureVector:
    """
    Wektor cech dla modeli ML.

    Przechowuje cechy jako słownik nazwa->wartość,
    z metadanymi o jakości danych źródłowych.
    """
    features: Dict[str, float]
    match_id: str
    quality: DataQuality
    extractor_names: List[str] = field(default_factory=list)

    def to_array(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Konwertuj do numpy array.

        Args:
            feature_names: Opcjonalna lista nazw cech (dla zachowania kolejności).
                          Jeśli None, użyje posortowanych kluczy.

        Returns:
            numpy array z wartościami cech
        """
        if feature_names is None:
            feature_names = sorted(self.features.keys())
        return np.array([self.features.get(name, 0.0) for name in feature_names])

    def get_feature_names(self) -> List[str]:
        """Zwróć posortowane nazwy cech."""
        return sorted(self.features.keys())

    def get_feature(self, name: str, default: float = 0.0) -> float:
        """Pobierz wartość cechy po nazwie."""
        return self.features.get(name, default)

    def has_feature(self, name: str) -> bool:
        """Sprawdź czy cecha istnieje."""
        return name in self.features

    def merge(self, other: "FeatureVector") -> "FeatureVector":
        """
        Połącz z innym wektorem cech.

        Args:
            other: Drugi wektor cech (musi mieć ten sam match_id)

        Returns:
            Nowy FeatureVector z połączonymi cechami
        """
        if self.match_id != other.match_id:
            raise ValueError(f"Cannot merge vectors for different matches: {self.match_id} vs {other.match_id}")

        merged_features = {**self.features, **other.features}
        merged_extractors = list(set(self.extractor_names + other.extractor_names))

        # Use lower quality of the two
        merged_quality = DataQuality(
            completeness=min(self.quality.completeness, other.quality.completeness),
            freshness_hours=max(self.quality.freshness_hours, other.quality.freshness_hours),
            sources_count=self.quality.sources_count + other.quality.sources_count,
            has_h2h=self.quality.has_h2h or other.quality.has_h2h,
            has_form=self.quality.has_form or other.quality.has_form,
            has_odds=self.quality.has_odds or other.quality.has_odds,
        )

        return FeatureVector(
            features=merged_features,
            match_id=self.match_id,
            quality=merged_quality,
            extractor_names=merged_extractors,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return f"FeatureVector(match={self.match_id}, features={len(self)}, quality={self.quality.completeness:.2f})"
