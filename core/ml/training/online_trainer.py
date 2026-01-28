"""
Online trainer for incremental model updates.

Checkpoint: 3.5
Responsibility: Incremental training with degradation detection.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

from core.ml.models.interface import MLModelInterface
from core.ml.registry import ModelRegistry, ModelVersion
from core.ml.training.examples import TrainingExample, TrainingBatch, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of a training run."""
    success: bool
    model_name: str
    version: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    samples_used: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""


@dataclass
class DegradationAlert:
    """Alert for model performance degradation."""
    model_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    drop_percentage: float
    detected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def severity(self) -> str:
        if self.drop_percentage > 20:
            return "critical"
        elif self.drop_percentage > 10:
            return "warning"
        return "info"


class OnlineTrainer:
    """
    Trener online dla aktualizacji modeli ML.

    Funkcje:
    - Zbieranie przykładów treningowych
    - Inkrementalne trenowanie
    - Wykrywanie degradacji wydajności
    - Automatyczne retrenowanie
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            registry: Model registry for version management
            config: Training configuration
        """
        self._registry = registry
        self._config = config or TrainingConfig()

        # Example buffer (per model)
        self._buffers: Dict[str, deque] = {}
        self._buffer_size = 1000

        # Performance tracking
        self._recent_metrics: Dict[str, deque] = {}  # model -> recent metric values
        self._baseline_metrics: Dict[str, Dict[str, float]] = {}

        # Callbacks
        self._on_degradation: Optional[Callable[[DegradationAlert], None]] = None

    def add_example(self, model_name: str, example: TrainingExample) -> None:
        """
        Add a training example to the buffer.

        Args:
            model_name: Target model
            example: Training example
        """
        if model_name not in self._buffers:
            self._buffers[model_name] = deque(maxlen=self._buffer_size)

        self._buffers[model_name].append(example)

        # Check if we should trigger training
        if len(self._buffers[model_name]) >= self._config.min_samples:
            logger.debug(f"Buffer for {model_name} has {len(self._buffers[model_name])} samples")

    def train_incremental(
        self,
        model: MLModelInterface,
        force: bool = False,
    ) -> TrainingResult:
        """
        Train model incrementally on buffered examples.

        Args:
            model: Model to train
            force: Force training even if buffer is small

        Returns:
            TrainingResult
        """
        model_name = model.name
        buffer = self._buffers.get(model_name, deque())

        if not force and len(buffer) < self._config.min_samples:
            return TrainingResult(
                success=False,
                model_name=model_name,
                error_message=f"Not enough samples: {len(buffer)} < {self._config.min_samples}",
            )

        start_time = datetime.utcnow()

        try:
            # Prepare training data
            batch = TrainingBatch(examples=list(buffer))
            complete = batch.get_complete_examples()

            if len(complete) < 10:
                return TrainingResult(
                    success=False,
                    model_name=model_name,
                    error_message=f"Not enough complete examples: {len(complete)}",
                )

            # Get feature vectors
            from core.ml.features import FeaturePipeline, FeatureVector
            from core.data.schemas import DataQuality

            feature_vectors = []
            for ex in complete:
                fv = FeatureVector(
                    features=ex.features,
                    match_id=ex.match_id,
                    quality=DataQuality(completeness=1.0, freshness_hours=0, sources_count=1),
                )
                feature_vectors.append(fv)

            # Get targets (depends on model type)
            if "goals" in model_name.lower():
                targets = batch.get_goals_targets()
            else:
                targets = batch.get_margin_targets()

            # Train
            metrics = model.train(
                feature_vectors,
                targets,
                validation_split=self._config.validation_split,
            )

            # Register new version
            version = self._registry.register(
                model,
                metrics,
                description=f"Incremental training on {len(complete)} samples",
                tags=["incremental"],
            )

            # Update baseline metrics
            self._update_baseline(model_name, metrics)

            # Clear buffer
            self._buffers[model_name].clear()

            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info(f"Incremental training complete for {model_name}: {metrics}")

            return TrainingResult(
                success=True,
                model_name=model_name,
                version=version.version,
                metrics=metrics,
                samples_used=len(complete),
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            return TrainingResult(
                success=False,
                model_name=model_name,
                error_message=str(e),
            )

    def check_degradation(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        threshold_pct: float = 10.0,
    ) -> List[DegradationAlert]:
        """
        Check for performance degradation.

        Args:
            model_name: Model to check
            current_metrics: Current performance metrics
            threshold_pct: Percentage drop to trigger alert

        Returns:
            List of degradation alerts
        """
        alerts = []
        baseline = self._baseline_metrics.get(model_name, {})

        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline:
                continue

            baseline_value = baseline[metric_name]
            if baseline_value == 0:
                continue

            # Calculate drop (assuming higher is better)
            drop_pct = ((baseline_value - current_value) / baseline_value) * 100

            if drop_pct > threshold_pct:
                alert = DegradationAlert(
                    model_name=model_name,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    drop_percentage=drop_pct,
                )
                alerts.append(alert)

                logger.warning(
                    f"Degradation detected for {model_name}.{metric_name}: "
                    f"{baseline_value:.3f} -> {current_value:.3f} ({drop_pct:.1f}% drop)"
                )

                # Trigger callback
                if self._on_degradation:
                    self._on_degradation(alert)

        return alerts

    def set_degradation_callback(
        self,
        callback: Callable[[DegradationAlert], None],
    ) -> None:
        """Set callback for degradation alerts."""
        self._on_degradation = callback

    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer sizes."""
        return {name: len(buffer) for name, buffer in self._buffers.items()}

    def should_retrain(
        self,
        model_name: str,
        min_examples: Optional[int] = None,
        max_age_hours: float = 24.0,
    ) -> bool:
        """
        Check if model should be retrained.

        Args:
            model_name: Model to check
            min_examples: Minimum examples in buffer
            max_age_hours: Maximum age of current version

        Returns:
            True if retraining is recommended
        """
        min_examples = min_examples or self._config.min_samples

        # Check buffer size
        buffer_size = len(self._buffers.get(model_name, []))
        if buffer_size >= min_examples:
            return True

        # Check model age
        active = self._registry.get_active_version(model_name)
        if active and active.age_hours > max_age_hours:
            return True

        return False

    def auto_retrain(
        self,
        model: MLModelInterface,
        min_examples: Optional[int] = None,
        max_age_hours: float = 24.0,
    ) -> Optional[TrainingResult]:
        """
        Automatically retrain if conditions are met.

        Args:
            model: Model to potentially retrain
            min_examples: Minimum examples required
            max_age_hours: Maximum model age

        Returns:
            TrainingResult if training occurred, None otherwise
        """
        if self.should_retrain(model.name, min_examples, max_age_hours):
            return self.train_incremental(model, force=True)
        return None

    def _update_baseline(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Update baseline metrics with exponential moving average."""
        if model_name not in self._baseline_metrics:
            self._baseline_metrics[model_name] = metrics.copy()
            return

        alpha = 0.3  # Weight for new metrics
        for key, value in metrics.items():
            if key in self._baseline_metrics[model_name]:
                old = self._baseline_metrics[model_name][key]
                self._baseline_metrics[model_name][key] = alpha * value + (1 - alpha) * old
            else:
                self._baseline_metrics[model_name][key] = value
