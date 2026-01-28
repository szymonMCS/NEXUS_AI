"""
Feature Selection and Dimensionality Reduction.

Based on research:
- "Research and performance analysis of random forest-based feature selection algorithm 
   in sports effectiveness evaluation" (Accuracy 0.819)
- "Predicting football match outcomes: a multilayer perceptron neural network model" 
   (86.7% accuracy with PCA)

Implements:
1. PCA (Principal Component Analysis) - dimensionality reduction
2. Random Forest Feature Selection - importance-based selection
3. Artificial Raindrop Algorithm (ARA) - optimization
4. Combined selection pipeline
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance score."""
    feature_name: str
    importance_score: float
    rank: int
    selected: bool


@dataclass
class SelectionResult:
    """Result of feature selection."""
    selected_features: List[str]
    n_original: int
    n_selected: int
    explained_variance: Optional[float] = None
    method: str = ""
    importance_scores: Dict[str, float] = None


class PCAFeatureReducer:
    """
    PCA-based dimensionality reduction.
    
    Research shows PCA improves prediction accuracy by 10-15%
    by reducing noise and collinearity.
    """
    
    def __init__(self, variance_threshold: float = 0.95, max_components: int = 20):
        """
        Initialize PCA reducer.
        
        Args:
            variance_threshold: Minimum cumulative explained variance
            max_components: Maximum number of components to keep
        """
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.n_components: int = 0
        
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'PCAFeatureReducer':
        """
        Fit PCA on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features
        """
        self.feature_names = feature_names
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA with all components first
        self.pca = PCA(n_components=min(X.shape[1], self.max_components))
        self.pca.fit(X_scaled)
        
        # Determine number of components
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Refit with optimal number
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        logger.info(f"PCA fitted: {len(feature_names)} → {self.n_components} components "
                   f"({sum(self.pca.explained_variance_ratio_):.1%} variance)")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA."""
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def get_component_features(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top contributing original features for each component.
        
        Returns:
            Dictionary mapping component index to list of (feature, weight) tuples
        """
        if self.pca is None:
            return {}
        
        components = {}
        for i in range(self.n_components):
            # Get loadings for this component
            loadings = self.pca.components_[i]
            # Sort by absolute value
            indices = np.argsort(np.abs(loadings))[::-1]
            # Top 5 contributors
            top_features = [
                (self.feature_names[idx], loadings[idx])
                for idx in indices[:5]
            ]
            components[i] = top_features
        
        return components


class RandomForestFeatureSelector:
    """
    Random Forest-based feature selection.
    
    Based on research showing RF + ARA achieves 0.819 accuracy.
    Uses feature importance scores to select optimal feature subset.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        selection_threshold: float = 0.01,
        max_features: Optional[int] = None,
    ):
        """
        Initialize RF selector.
        
        Args:
            n_estimators: Number of trees in forest
            selection_threshold: Minimum importance to keep feature
            max_features: Maximum number of features to select
        """
        self.n_estimators = n_estimators
        self.selection_threshold = selection_threshold
        self.max_features = max_features
        self.rf = None
        self.feature_importance: Dict[str, float] = {}
        self.feature_names: List[str] = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task: str = "classification"
    ) -> 'RandomForestFeatureSelector':
        """
        Fit Random Forest and compute feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Feature names
            task: 'classification' or 'regression'
        """
        self.feature_names = feature_names
        
        # Train Random Forest
        if task == "classification":
            self.rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        
        self.rf.fit(X, y)
        
        # Store importance scores
        importance = self.rf.feature_importances_
        self.feature_importance = {
            name: score
            for name, score in zip(feature_names, importance)
        }
        
        logger.info(f"RF Feature Selection fitted on {len(feature_names)} features")
        
        return self
    
    def select(self) -> SelectionResult:
        """
        Select features based on importance scores.
        
        Returns:
            SelectionResult with selected features
        """
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter by threshold
        selected = [
            name for name, score in sorted_features
            if score >= self.selection_threshold
        ]
        
        # Apply max_features limit
        if self.max_features and len(selected) > self.max_features:
            selected = [name for name, _ in sorted_features[:self.max_features]]
        
        logger.info(f"Selected {len(selected)}/{len(self.feature_names)} features "
                   f"(threshold: {self.selection_threshold})")
        
        return SelectionResult(
            selected_features=selected,
            n_original=len(self.feature_names),
            n_selected=len(selected),
            method="random_forest",
            importance_scores=self.feature_importance
        )
    
    def transform(self, X: np.ndarray, selected_features: List[str]) -> np.ndarray:
        """Transform data keeping only selected features."""
        indices = [self.feature_names.index(f) for f in selected_features]
        return X[:, indices]


class ArtificialRaindropOptimizer:
    """
    Artificial Raindrop Algorithm (ARA) for feature optimization.
    
    Based on research combining RF with ARA for optimal feature selection.
    Simulates raindrop movement to find optimal feature subset.
    """
    
    def __init__(
        self,
        n_drops: int = 30,
        max_iter: int = 50,
        evaporation_rate: float = 0.1,
    ):
        """
        Initialize ARA optimizer.
        
        Args:
            n_drops: Number of raindrops (particles)
            max_iter: Maximum iterations
            evaporation_rate: Rate of evaporation (exploration vs exploitation)
        """
        self.n_drops = n_drops
        self.max_iter = max_iter
        self.evaporation_rate = evaporation_rate
        self.best_solution: Optional[np.ndarray] = None
        self.best_score: float = 0.0
        
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        evaluator: callable
    ) -> SelectionResult:
        """
        Optimize feature selection using ARA.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Feature names
            evaluator: Function to evaluate feature subset (returns score)
        
        Returns:
            SelectionResult with optimal feature set
        """
        n_features = X.shape[1]
        
        # Initialize raindrops (random binary vectors)
        drops = np.random.randint(0, 2, size=(self.n_drops, n_features))
        velocities = np.random.randn(self.n_drops, n_features) * 0.1
        
        # Ensure at least one feature selected
        drops = np.maximum(drops, 1)
        
        scores = np.zeros(self.n_drops)
        
        for iteration in range(self.max_iter):
            # Evaluate each drop
            for i in range(self.n_drops):
                selected_indices = np.where(drops[i] > 0.5)[0]
                if len(selected_indices) == 0:
                    scores[i] = 0
                    continue
                
                X_subset = X[:, selected_indices]
                scores[i] = evaluator(X_subset, y)
                
                # Update best solution
                if scores[i] > self.best_score:
                    self.best_score = scores[i]
                    self.best_solution = drops[i].copy()
            
            # Update velocities and positions
            for i in range(self.n_drops):
                # Move toward best solution
                attraction = (self.best_solution - drops[i]) * np.random.random(n_features)
                velocities[i] = velocities[i] * (1 - self.evaporation_rate) + attraction
                
                # Update position
                drops[i] = drops[i] + velocities[i]
                drops[i] = np.clip(drops[i], 0, 1)
                
                # Random reset (evaporation)
                if np.random.random() < self.evaporation_rate:
                    drops[i] = np.random.randint(0, 2, size=n_features)
        
        # Get final selected features
        if self.best_solution is not None:
            selected_indices = np.where(self.best_solution > 0.5)[0]
            selected_features = [feature_names[i] for i in selected_indices]
        else:
            selected_features = feature_names[:5]  # Default to top 5
        
        logger.info(f"ARA optimization complete. Selected {len(selected_features)} features, "
                   f"best score: {self.best_score:.4f}")
        
        return SelectionResult(
            selected_features=selected_features,
            n_original=n_features,
            n_selected=len(selected_features),
            method="artificial_raindrop_algorithm",
        )


class SportsFeatureSelector:
    """
    Combined feature selection pipeline for sports prediction.
    
    Implements the best practices from research:
    1. PCA for dimensionality reduction
    2. Random Forest for importance scoring
    3. ARA for optimization
    """
    
    def __init__(
        self,
        use_pca: bool = True,
        use_rf: bool = True,
        use_ara: bool = False,  # ARA is slower, optional
        pca_variance: float = 0.95,
        rf_threshold: float = 0.01,
    ):
        """
        Initialize combined selector.
        
        Args:
            use_pca: Whether to use PCA
            use_rf: Whether to use Random Forest selection
            use_ara: Whether to use ARA optimization
            pca_variance: PCA variance threshold
            rf_threshold: RF importance threshold
        """
        self.use_pca = use_pca
        self.use_rf = use_rf
        self.use_ara = use_ara
        
        self.pca_reducer = PCAFeatureReducer(variance_threshold=pca_variance)
        self.rf_selector = RandomForestFeatureSelector(selection_threshold=rf_threshold)
        self.ara_optimizer = ArtificialRaindropOptimizer()
        
        self.selection_history: List[SelectionResult] = []
        
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task: str = "classification"
    ) -> Tuple[np.ndarray, SelectionResult]:
        """
        Fit selector and transform data.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Feature names
            task: 'classification' or 'regression'
        
        Returns:
            Tuple of (transformed X, SelectionResult)
        """
        current_features = feature_names.copy()
        
        # Step 1: Random Forest Selection
        if self.use_rf:
            logger.info("Step 1: Random Forest feature selection...")
            self.rf_selector.fit(X, y, current_features, task)
            rf_result = self.rf_selector.select()
            
            # Transform
            selected_indices = [current_features.index(f) for f in rf_result.selected_features]
            X = X[:, selected_indices]
            current_features = rf_result.selected_features
            self.selection_history.append(rf_result)
        
        # Step 2: ARA Optimization (if enabled)
        if self.use_ara:
            logger.info("Step 2: ARA optimization...")
            
            def evaluator(X_sub, y_sub):
                from sklearn.model_selection import cross_val_score
                from sklearn.ensemble import RandomForestClassifier
                
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                scores = cross_val_score(clf, X_sub, y_sub, cv=3, scoring='accuracy')
                return scores.mean()
            
            ara_result = self.ara_optimizer.optimize(X, y, current_features, evaluator)
            
            selected_indices = [current_features.index(f) for f in ara_result.selected_features]
            X = X[:, selected_indices]
            current_features = ara_result.selected_features
            self.selection_history.append(ara_result)
        
        # Step 3: PCA Reduction
        if self.use_pca:
            logger.info("Step 3: PCA dimensionality reduction...")
            X = self.pca_reducer.fit_transform(X, current_features)
            
            pca_result = SelectionResult(
                selected_features=[f"PC{i+1}" for i in range(X.shape[1])],
                n_original=len(current_features),
                n_selected=X.shape[1],
                explained_variance=sum(self.pca_reducer.pca.explained_variance_ratio_),
                method="pca"
            )
            self.selection_history.append(pca_result)
        
        final_result = SelectionResult(
            selected_features=current_features if not self.use_pca else pca_result.selected_features,
            n_original=len(feature_names),
            n_selected=X.shape[1],
            method="combined_pipeline"
        )
        
        logger.info(f"Feature selection complete: {len(feature_names)} → {X.shape[1]} features")
        
        return X, final_result
    
    def get_selection_report(self) -> str:
        """Generate text report of selection process."""
        lines = [
            "=" * 60,
            "Feature Selection Report",
            "=" * 60,
            ""
        ]
        
        for i, result in enumerate(self.selection_history, 1):
            lines.append(f"Step {i}: {result.method}")
            lines.append(f"  {result.n_original} → {result.n_selected} features")
            if result.explained_variance:
                lines.append(f"  Explained variance: {result.explained_variance:.1%}")
            lines.append("")
        
        return '\n'.join(lines)
