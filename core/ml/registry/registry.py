"""
Model registry for version management.

Checkpoint: 3.2
Responsibility: Manage model versions, rollback, and deployment.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type
from threading import Lock

from core.ml.models.interface import MLModelInterface
from core.ml.registry.version import ModelVersion, VersionComparison


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Rejestr modeli ML z wersjonowaniem.

    Funkcje:
    - Rejestracja nowych wersji modeli
    - Śledzenie aktywnych wersji
    - Rollback do poprzednich wersji
    - Porównywanie wydajności
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize registry.

        Args:
            storage_path: Directory for storing models and metadata
        """
        self._storage_path = storage_path or Path("models")
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._versions: Dict[str, List[ModelVersion]] = {}  # model_name -> versions
        self._active_versions: Dict[str, str] = {}  # model_name -> active_version
        self._lock = Lock()

        # Load existing registry
        self._load_registry()

    def register(
        self,
        model: MLModelInterface,
        metrics: Dict[str, float],
        description: str = "",
        tags: Optional[List[str]] = None,
        auto_activate: bool = True,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: Trained model instance
            metrics: Training/validation metrics
            description: Version description
            tags: Optional tags for filtering
            auto_activate: Automatically make this the active version

        Returns:
            Created ModelVersion
        """
        with self._lock:
            model_name = model.name

            # Generate version number
            existing = self._versions.get(model_name, [])
            version_num = len(existing) + 1
            version_str = f"v{version_num}.0.0"

            # Create version directory
            version_path = self._storage_path / model_name / version_str
            version_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model_file = version_path / "model.json"
            model.save(model_file)

            # Create version record
            version = ModelVersion(
                name=model_name,
                version=version_str,
                created_at=datetime.utcnow(),
                metrics=metrics,
                path=version_path,
                training_samples=model.get_model_info().training_samples,
                feature_names=model.get_model_info().feature_names,
                description=description,
                tags=tags or [],
                parent_version=self._active_versions.get(model_name),
            )

            # Store
            if model_name not in self._versions:
                self._versions[model_name] = []
            self._versions[model_name].append(version)

            # Auto-activate if requested
            if auto_activate:
                self._activate_version(model_name, version_str)

            # Save registry
            self._save_registry()

            logger.info(f"Registered {version.full_name} with metrics: {metrics}")
            return version

    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model."""
        with self._lock:
            active_ver = self._active_versions.get(model_name)
            if not active_ver:
                return None

            for v in self._versions.get(model_name, []):
                if v.version == active_ver:
                    return v
            return None

    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific version."""
        with self._lock:
            for v in self._versions.get(model_name, []):
                if v.version == version:
                    return v
            return None

    def list_versions(
        self,
        model_name: str,
        include_deprecated: bool = False,
    ) -> List[ModelVersion]:
        """List all versions for a model."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            if not include_deprecated:
                versions = [v for v in versions if not v.is_deprecated]
            return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        with self._lock:
            return list(self._versions.keys())

    def activate(self, model_name: str, version: str) -> bool:
        """
        Activate a specific version.

        Args:
            model_name: Model name
            version: Version to activate

        Returns:
            True if successful
        """
        with self._lock:
            # Verify version exists
            v = None
            for ver in self._versions.get(model_name, []):
                if ver.version == version:
                    v = ver
                    break

            if not v:
                logger.warning(f"Version {model_name}:{version} not found")
                return False

            if v.is_deprecated:
                logger.warning(f"Cannot activate deprecated version {v.full_name}")
                return False

            self._activate_version(model_name, version)
            self._save_registry()

            logger.info(f"Activated {v.full_name}")
            return True

    def rollback(self, model_name: str, steps: int = 1) -> Optional[ModelVersion]:
        """
        Rollback to a previous version.

        Args:
            model_name: Model name
            steps: How many versions to go back

        Returns:
            New active version, or None if failed
        """
        with self._lock:
            versions = self.list_versions(model_name)
            if len(versions) <= steps:
                logger.warning(f"Not enough versions to rollback {steps} steps")
                return None

            # Find current active index
            active_ver = self._active_versions.get(model_name)
            current_idx = 0
            for i, v in enumerate(versions):
                if v.version == active_ver:
                    current_idx = i
                    break

            # Calculate target
            target_idx = current_idx + steps
            if target_idx >= len(versions):
                target_idx = len(versions) - 1

            target = versions[target_idx]
            self._activate_version(model_name, target.version)
            self._save_registry()

            logger.info(f"Rolled back {model_name} to {target.version}")
            return target

    def deprecate(self, model_name: str, version: str) -> bool:
        """Mark a version as deprecated."""
        with self._lock:
            for v in self._versions.get(model_name, []):
                if v.version == version:
                    v.is_deprecated = True
                    self._save_registry()
                    logger.info(f"Deprecated {v.full_name}")
                    return True
            return False

    def compare_versions(
        self,
        model_name: str,
        old_version: str,
        new_version: str,
        comparison_metric: str = "accuracy",
    ) -> Optional[VersionComparison]:
        """Compare two versions."""
        old = self.get_version(model_name, old_version)
        new = self.get_version(model_name, new_version)

        if not old or not new:
            return None

        return VersionComparison(
            old_version=old,
            new_version=new,
            comparison_metric=comparison_metric,
        )

    def load_model(
        self,
        model_name: str,
        model_class: Type[MLModelInterface],
        version: Optional[str] = None,
    ) -> Optional[MLModelInterface]:
        """
        Load a model from the registry.

        Args:
            model_name: Model name
            model_class: Class to instantiate
            version: Specific version (uses active if None)

        Returns:
            Loaded model instance
        """
        if version:
            v = self.get_version(model_name, version)
        else:
            v = self.get_active_version(model_name)

        if not v or not v.path:
            return None

        model_file = v.path / "model.json"
        if not model_file.exists():
            return None

        model = model_class()
        if model.load(model_file):
            return model
        return None

    def get_best_version(
        self,
        model_name: str,
        metric: str,
        higher_is_better: bool = True,
    ) -> Optional[ModelVersion]:
        """Get the version with best metric value."""
        versions = self.list_versions(model_name)
        if not versions:
            return None

        best = None
        best_value = None

        for v in versions:
            value = v.get_metric(metric)
            if value is None:
                continue

            if best_value is None:
                best = v
                best_value = value
            elif higher_is_better and value > best_value:
                best = v
                best_value = value
            elif not higher_is_better and value < best_value:
                best = v
                best_value = value

        return best

    def cleanup_old_versions(
        self,
        model_name: str,
        keep_count: int = 5,
        keep_active: bool = True,
    ) -> int:
        """
        Remove old versions to save space.

        Args:
            model_name: Model name
            keep_count: Number of recent versions to keep
            keep_active: Always keep active version

        Returns:
            Number of versions removed
        """
        with self._lock:
            versions = sorted(
                self._versions.get(model_name, []),
                key=lambda v: v.created_at,
                reverse=True,
            )

            active_ver = self._active_versions.get(model_name)
            removed = 0

            for i, v in enumerate(versions):
                if i < keep_count:
                    continue
                if keep_active and v.version == active_ver:
                    continue

                # Remove files
                if v.path and v.path.exists():
                    import shutil
                    shutil.rmtree(v.path, ignore_errors=True)

                # Remove from list
                self._versions[model_name].remove(v)
                removed += 1

            if removed > 0:
                self._save_registry()
                logger.info(f"Cleaned up {removed} old versions of {model_name}")

            return removed

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _activate_version(self, model_name: str, version: str) -> None:
        """Internal: activate a version."""
        # Deactivate current
        for v in self._versions.get(model_name, []):
            v.is_active = (v.version == version)

        self._active_versions[model_name] = version

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self._storage_path / "registry.json"

        data = {
            "versions": {
                name: [v.to_dict() for v in versions]
                for name, versions in self._versions.items()
            },
            "active_versions": self._active_versions,
        }

        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self._storage_path / "registry.json"

        if not registry_file.exists():
            return

        try:
            with open(registry_file) as f:
                data = json.load(f)

            self._versions = {
                name: [ModelVersion.from_dict(v) for v in versions]
                for name, versions in data.get("versions", {}).items()
            }
            self._active_versions = data.get("active_versions", {})

            logger.info(f"Loaded registry with {len(self._versions)} models")

        except Exception as e:
            logger.error(f"Error loading registry: {e}")
