"""
ML Model interface.

Checkpoint: 2.2
Responsibility: Define contract for all ML models.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from pathlib import Path

from core.ml.features import FeatureVector
from core.ml.models.predictions import ModelInfo


# Generic type for prediction output
T = TypeVar('T')


class MLModelInterface(ABC, Generic[T]):
    """
    Abstrakcyjny interfejs dla modeli ML.

    Każdy model musi implementować:
    - predict() - predykcja na podstawie cech
    - train() - trenowanie na danych historycznych
    - save()/load() - persystencja modelu
    - get_model_info() - metadane modelu
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unikalna nazwa modelu."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Wersja modelu."""
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Czy model jest wytrenowany."""
        pass

    @abstractmethod
    def predict(self, features: FeatureVector) -> T:
        """
        Wykonaj predykcję.

        Args:
            features: Wektor cech dla meczu

        Returns:
            Predykcja (typ zależy od modelu)

        Raises:
            ValueError: Jeśli model nie jest wytrenowany
        """
        pass

    @abstractmethod
    def predict_batch(self, features_list: List[FeatureVector]) -> List[T]:
        """
        Predykcja dla wielu meczów.

        Args:
            features_list: Lista wektorów cech

        Returns:
            Lista predykcji
        """
        pass

    @abstractmethod
    def train(
        self,
        features: List[FeatureVector],
        targets: List[Any],
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Wytrenuj model.

        Args:
            features: Lista wektorów cech treningowych
            targets: Lista wartości docelowych
            validation_split: Część danych do walidacji

        Returns:
            Dict z metrykami treningu (accuracy, loss, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> bool:
        """
        Zapisz model do pliku.

        Args:
            path: Ścieżka do zapisu

        Returns:
            True jeśli sukces
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> bool:
        """
        Wczytaj model z pliku.

        Args:
            path: Ścieżka do pliku

        Returns:
            True jeśli sukces
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Zwróć informacje o modelu."""
        pass

    def validate_features(self, features: FeatureVector) -> bool:
        """
        Sprawdź czy cechy są odpowiednie dla tego modelu.

        Args:
            features: Wektor cech

        Returns:
            True jeśli cechy są poprawne
        """
        required = self.get_required_features()
        for name in required:
            if not features.has_feature(name):
                return False
        return True

    def get_required_features(self) -> List[str]:
        """
        Zwróć listę wymaganych cech.

        Override w podklasach.
        """
        return []

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Zwróć ważność cech (jeśli model to wspiera).

        Returns:
            Dict nazwa_cechy -> ważność, lub None
        """
        return None


class EnsembleModel(MLModelInterface[T]):
    """
    Bazowa klasa dla modeli ensemble.

    Łączy predykcje z wielu modeli.
    """

    def __init__(self):
        self._models: List[MLModelInterface] = []
        self._weights: List[float] = []

    def add_model(self, model: MLModelInterface, weight: float = 1.0) -> None:
        """Dodaj model do ensemble."""
        self._models.append(model)
        self._weights.append(weight)
        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

    def remove_model(self, name: str) -> bool:
        """Usuń model z ensemble."""
        for i, model in enumerate(self._models):
            if model.name == name:
                self._models.pop(i)
                self._weights.pop(i)
                # Renormalize
                if self._weights:
                    total = sum(self._weights)
                    self._weights = [w / total for w in self._weights]
                return True
        return False

    @property
    def is_trained(self) -> bool:
        return all(m.is_trained for m in self._models)

    def get_model_info(self) -> ModelInfo:
        from datetime import datetime
        return ModelInfo(
            name=self.name,
            version=self.version,
            trained_at=datetime.utcnow(),
            training_samples=0,
            metrics={},
            feature_names=[],
        )
