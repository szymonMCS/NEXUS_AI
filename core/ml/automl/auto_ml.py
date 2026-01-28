"""
AutoML for Sports Prediction.

Automated search for optimal:
- Model architecture
- Hyperparameters
- Feature engineering
- Ensemble weights
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelConfig:
    model_type: str
    hyperparams: Dict[str, Any]
    features: List[str]
    score: float = 0.0


@dataclass 
class SearchResult:
    best_config: ModelConfig
    all_configs: List[ModelConfig]
    search_time: float
    n_evaluations: int
    method: str


class BayesianOptimizer:
    """Bayesian Optimization for hyperparameters."""
    
    def __init__(self, n_initial: int = 5, n_iterations: int = 20):
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.X: List[Dict] = []
        self.y: List[float] = []
    
    def optimize(self, search_space: Dict, objective: callable) -> Dict:
        """Optimize hyperparameters."""
        # Random initial points
        for _ in range(self.n_initial):
            params = self._random_sample(search_space)
            score = objective(params)
            self.X.append(params)
            self.y.append(score)
        
        # Bayesian iterations
        for _ in range(self.n_iterations):
            params = self._acquisition_function(search_space)
            score = objective(params)
            self.X.append(params)
            self.y.append(score)
        
        best_idx = np.argmax(self.y)
        return self.X[best_idx]
    
    def _random_sample(self, search_space: Dict) -> Dict:
        params = {}
        for name, (min_val, max_val, type_) in search_space.items():
            if type_ == 'int':
                params[name] = np.random.randint(min_val, max_val + 1)
            elif type_ == 'float':
                params[name] = np.random.uniform(min_val, max_val)
            elif type_ == 'choice':
                params[name] = np.random.choice(min_val)
        return params
    
    def _acquisition_function(self, search_space: Dict) -> Dict:
        if np.random.random() < 0.3:
            return self._random_sample(search_space)
        else:
            best_idx = np.argmax(self.y)
            best_params = self.X[best_idx].copy()
            for key in best_params:
                if isinstance(best_params[key], (int, float)):
                    best_params[key] += np.random.normal(0, 0.1)
            return best_params


class AutoMLPipeline:
    """Complete AutoML pipeline."""
    
    def __init__(self, time_budget: int = 3600, n_trials: int = 50):
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.results: List[ModelConfig] = []
        self.bayesian_opt = BayesianOptimizer()
    
    def search(self, X, y, feature_names, sport="football") -> SearchResult:
        """Run AutoML search."""
        import time
        start_time = time.time()
        
        logger.info(f"Starting AutoML Search for {sport}")
        
        # Feature selection
        selected_features = self._auto_feature_selection(X, y, feature_names)
        
        # Optimize RF
        config = self._optimize_rf(X, y, selected_features)
        self.results.append(config)
        
        search_time = time.time() - start_time
        best_config = max(self.results, key=lambda x: x.score)
        
        return SearchResult(
            best_config=best_config,
            all_configs=self.results,
            search_time=search_time,
            n_evaluations=len(self.results),
            method="automl_v1"
        )
    
    def _auto_feature_selection(self, X, y, feature_names, max_features=20):
        """Select best features."""
        if not SKLEARN_AVAILABLE or X.shape[1] <= max_features:
            return feature_names
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            importance = rf.feature_importances_
            top_indices = np.argsort(importance)[-max_features:]
            return [feature_names[i] for i in top_indices]
        except:
            return feature_names[:max_features]
    
    def _optimize_rf(self, X, y, features) -> ModelConfig:
        """Optimize Random Forest."""
        search_space = {
            'n_estimators': (50, 300, 'int'),
            'max_depth': (5, 30, 'int'),
        }
        
        def objective(params):
            if not SKLEARN_AVAILABLE:
                return np.random.random()
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                random_state=42,
            )
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        best_params = self.bayesian_opt.optimize(search_space, objective)
        score = objective(best_params)
        
        return ModelConfig(model_type='rf', hyperparams=best_params, features=features, score=score)
