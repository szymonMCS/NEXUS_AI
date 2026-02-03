"""
Scikit-learn based models for NEXUS AI.
Random Forest and Gradient Boosting implementations.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

from core.ml.models.base import BasePredictor, PredictionResult, ModelMetadata

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. ML models will use statistical fallback.")


class RandomForestPredictor(BasePredictor):
    """
    Random Forest predictor for sports outcomes.
    Good baseline model with feature importance.
    """
    
    def __init__(self, 
                 name: str = "RandomForest",
                 version: str = "1.0", 
                 sport: str = "general",
                 n_estimators: int = 200,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 **kwargs):
        super().__init__(name, version, sport)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        else:
            self.model = None
            
        self.update_metadata(
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            }
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, **kwargs):
        """
        Train Random Forest model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=away, 1=home, 2=draw for football)
            feature_names: Optional list of feature names
        """
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not available, model not trained")
            return
        
        if feature_names:
            self.feature_names = feature_names
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Update metadata
        self.update_metadata(
            training_samples=len(y),
            features=self.feature_names
        )
        
        # Cross-validation score
        try:
            scores = cross_val_score(self.model, X, y, cv=3)
            self.update_metadata(accuracy=float(scores.mean()))
        except Exception:
            pass
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to array."""
        if self.feature_names:
            values = [features.get(f, 0.0) for f in self.feature_names]
        else:
            values = list(features.values())
        return np.array(values).reshape(1, -1)
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict match outcome."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return PredictionResult(
                home_win_prob=0.5,
                away_win_prob=0.5,
                confidence=0.0,
                model_name=self.name
            )
        
        X = self._features_to_array(features)
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        
        # Handle different number of classes
        if len(probs) == 2:
            # Binary classification (home/away)
            away_prob, home_prob = probs[0], probs[1]
            draw_prob = None
        elif len(probs) == 3:
            # Three outcomes (away/draw/home)
            away_prob, draw_prob, home_prob = probs[0], probs[1], probs[2]
        else:
            home_prob = probs[-1] if len(probs) > 1 else 0.5
            away_prob = 1 - home_prob
            draw_prob = None
        
        # Confidence based on probability spread
        confidence = abs(home_prob - away_prob) + 0.3
        confidence = min(1.0, confidence)
        
        return PredictionResult(
            home_win_prob=float(home_prob),
            away_win_prob=float(away_prob),
            draw_prob=float(draw_prob) if draw_prob else None,
            confidence=float(confidence),
            model_name=self.name,
            version=self.version,
            features_used=list(features.keys())
        )
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return [PredictionResult(0.5, 0.5, confidence=0.0) for _ in features_list]
        
        X = np.array([self._features_to_array(f)[0] for f in features_list])
        if self.scaler:
            X = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X)
        
        results = []
        for i, prob in enumerate(probs):
            if len(prob) == 2:
                away_prob, home_prob = prob[0], prob[1]
                draw_prob = None
            elif len(prob) == 3:
                away_prob, draw_prob, home_prob = prob[0], prob[1], prob[2]
            else:
                home_prob = prob[-1] if len(prob) > 1 else 0.5
                away_prob = 1 - home_prob
                draw_prob = None
            
            confidence = abs(home_prob - away_prob) + 0.3
            
            results.append(PredictionResult(
                home_win_prob=float(home_prob),
                away_win_prob=float(away_prob),
                draw_prob=float(draw_prob) if draw_prob else None,
                confidence=float(min(1.0, confidence)),
                model_name=self.name,
                version=self.version,
                features_used=list(features_list[i].keys())
            ))
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return {}
        
        importances = self.model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}


class GradientBoostingPredictor(BasePredictor):
    """
    Gradient Boosting predictor (XGBoost-like).
    Good for capturing non-linear relationships.
    """
    
    def __init__(self,
                 name: str = "GradientBoosting",
                 version: str = "1.0",
                 sport: str = "general",
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 **kwargs):
        super().__init__(name, version, sport)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                **kwargs
            )
        else:
            self.model = None
            
        self.update_metadata(
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, **kwargs):
        """Train Gradient Boosting model."""
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not available, model not trained")
            return
        
        if feature_names:
            self.feature_names = feature_names
        
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        self.update_metadata(
            training_samples=len(y),
            features=self.feature_names
        )
        
        try:
            scores = cross_val_score(self.model, X, y, cv=3)
            self.update_metadata(accuracy=float(scores.mean()))
        except Exception:
            pass
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to array."""
        if self.feature_names:
            values = [features.get(f, 0.0) for f in self.feature_names]
        else:
            values = list(features.values())
        return np.array(values).reshape(1, -1)
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict match outcome."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return PredictionResult(0.5, 0.5, confidence=0.0, model_name=self.name)
        
        X = self._features_to_array(features)
        if self.scaler:
            X = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X)[0]
        
        if len(probs) == 2:
            away_prob, home_prob = probs[0], probs[1]
            draw_prob = None
        elif len(probs) == 3:
            away_prob, draw_prob, home_prob = probs[0], probs[1], probs[2]
        else:
            home_prob = probs[-1] if len(probs) > 1 else 0.5
            away_prob = 1 - home_prob
            draw_prob = None
        
        confidence = abs(home_prob - away_prob) + 0.3
        
        return PredictionResult(
            home_win_prob=float(home_prob),
            away_win_prob=float(away_prob),
            draw_prob=float(draw_prob) if draw_prob else None,
            confidence=float(min(1.0, confidence)),
            model_name=self.name,
            version=self.version,
            features_used=list(features.keys())
        )
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return [PredictionResult(0.5, 0.5, confidence=0.0) for _ in features_list]
        
        X = np.array([self._features_to_array(f)[0] for f in features_list])
        if self.scaler:
            X = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X)
        
        results = []
        for i, prob in enumerate(probs):
            if len(prob) == 2:
                away_prob, home_prob = prob[0], prob[1]
                draw_prob = None
            elif len(prob) == 3:
                away_prob, draw_prob, home_prob = prob[0], prob[1], prob[2]
            else:
                home_prob = prob[-1] if len(prob) > 1 else 0.5
                away_prob = 1 - home_prob
                draw_prob = None
            
            confidence = abs(home_prob - away_prob) + 0.3
            
            results.append(PredictionResult(
                home_win_prob=float(home_prob),
                away_win_prob=float(away_prob),
                draw_prob=float(draw_prob) if draw_prob else None,
                confidence=float(min(1.0, confidence)),
                model_name=self.name,
                version=self.version,
                features_used=list(features_list[i].keys())
            ))
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return {}
        
        importances = self.model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
