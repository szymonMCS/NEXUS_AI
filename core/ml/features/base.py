"""
Base feature extractor interface.

Checkpoint: 1.2
Responsibility: Define contract for all feature extractors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set
import math
import logging

from core.data.schemas import MatchData


logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Bazowy interfejs dla ekstraktorów cech.

    Każdy ekstraktor wyciąga określony zestaw cech z danych meczu.
    Ekstraktory są komponowalne - można je łączyć w pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unikalna nazwa ekstraktora."""
        pass

    @property
    def required_fields(self) -> Set[str]:
        """
        Pola MatchData wymagane przez ten ekstraktor.

        Override w podklasach aby określić zależności.
        """
        return set()

    @abstractmethod
    def extract(self, match: MatchData) -> Dict[str, float]:
        """
        Wyciągnij cechy z danych meczu.

        Args:
            match: Dane meczu

        Returns:
            Dict z nazwami cech i ich wartościami (float)

        Raises:
            ValueError: Jeśli brakuje wymaganych danych
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Zwróć listę nazw cech które ten ekstraktor generuje.

        Kolejność musi być deterministyczna (dla reprodukowalności ML).
        """
        pass

    def can_extract(self, match: MatchData) -> bool:
        """
        Sprawdź czy można wyciągnąć cechy z tego meczu.

        Args:
            match: Dane meczu

        Returns:
            True jeśli wszystkie wymagane dane są dostępne
        """
        for field_name in self.required_fields:
            if not self._has_field(match, field_name):
                return False
        return True

    def extract_safe(self, match: MatchData) -> Dict[str, float]:
        """
        Bezpieczna ekstrakcja - zwraca puste cechy jeśli brakuje danych.

        Args:
            match: Dane meczu

        Returns:
            Dict z cechami lub dict z zerami dla wszystkich cech
        """
        if not self.can_extract(match):
            logger.warning(f"{self.name}: Cannot extract features for match {match.match_id}")
            return {name: 0.0 for name in self.get_feature_names()}

        try:
            features = self.extract(match)
            if not self.validate_features(features):
                logger.warning(f"{self.name}: Invalid features for match {match.match_id}")
                return {name: 0.0 for name in self.get_feature_names()}
            return features
        except Exception as e:
            logger.error(f"{self.name}: Error extracting features: {e}")
            return {name: 0.0 for name in self.get_feature_names()}

    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Sprawdź czy cechy są poprawne (brak NaN/Inf).

        Args:
            features: Dict z cechami

        Returns:
            True jeśli wszystkie wartości są prawidłowe
        """
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                logger.warning(f"{self.name}: Feature {name} is not numeric: {type(value)}")
                return False
            if math.isnan(value) or math.isinf(value):
                logger.warning(f"{self.name}: Feature {name} is NaN or Inf")
                return False
        return True

    def _has_field(self, match: MatchData, field_path: str) -> bool:
        """
        Sprawdź czy pole istnieje i nie jest None.

        Obsługuje ścieżki typu "home_stats.goals_scored_avg".
        """
        parts = field_path.split(".")
        obj = match

        for part in parts:
            if obj is None:
                return False
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return False

        return obj is not None

    def _safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Bezpieczne dzielenie z obsługą dzielenia przez zero."""
        if b == 0:
            return default
        return a / b

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        """Ogranicz wartość do zakresu."""
        return max(min_val, min(max_val, value))
