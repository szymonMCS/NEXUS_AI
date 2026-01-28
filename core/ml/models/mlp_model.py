"""
Multi-Layer Perceptron (MLP) Neural Network for Sports Prediction.

Based on research:
- "Predicting football match outcomes: a multilayer perceptron neural network model"
  Data: FIFA World Cup technical statistics
  Method: MLP + PCA (22 technical indicators)
  Results: 86.7% accuracy

Implements deep neural network with PCA preprocessing for sports outcome prediction.
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available, MLP model will use fallback")

from core.ml.models.interface import MLModelInterface
from core.ml.models.predictions import PredictionResult, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class MLPParameters:
    """Parameters for MLP Neural Network."""
    # Architecture
    hidden_layer_sizes: tuple = (128, 64, 32)  # 3 hidden layers
    activation: str = 'relu'
    solver: str = 'adam'
    alpha: float = 0.0001  # L2 regularization
    
    # Training
    learning_rate: str = 'adaptive'
    learning_rate_init: float = 0.001
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    
    # Other
    batch_size: str = 'auto'
    random_state: int = 42
    verbose: bool = False


class MLPNeuralNetworkModel(MLModelInterface[PredictionResult]):
    """
    Multi-Layer Perceptron Neural Network for sports prediction.
    
    Architecture based on research achieving 86.7% accuracy:
    - Input: 22 technical indicators (after PCA)
    - Hidden: 128 → 64 → 32 neurons
    - Output: 3 classes (home win, draw, away win) or 1 regression value
    
    Features:
    - PCA preprocessing for dimensionality reduction
    - Batch normalization via StandardScaler
    - Early stopping to prevent overfitting
    - Adaptive learning rate
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        params: Optional[MLPParameters] = None,
        task: str = "classification",
        use_pca: bool = True,
        pca_components: int = 22,  # As in research
    ):
        """
        Initialize MLP Neural Network.
        
        Args:
            params: MLP parameters
            task: 'classification' or 'regression'
            use_pca: Whether to use PCA preprocessing
            pca_components: Number of PCA components (research used 22)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for MLP model")
        
        self._params = params or MLPParameters()
        self.task = task
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task == "classification" else None
        
        # PCA
        self.pca = None
        if use_pca:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=pca_components)
        
        self._trained = False
        self._training_samples = 0
        self._trained_at: Optional[datetime] = None
        self._metrics: Dict[str, float] = {}
        self._feature_names: List[str] = []
        
        # Training history
        self.loss_curve_: List[float] = []
        
    @property
    def name(self) -> str:
        return f"mlp_neural_network_{self.task}"
    
    @property
    def version(self) -> str:
        return self.VERSION
    
    @property
    def is_trained(self) -> bool:
        return self._trained
    
    def _preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features: scale and optionally apply PCA.
        
        Args:
            features: Raw features
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed features
        """
        # Scale features
        if fit:
            X = self.scaler.fit_transform(features)
        else:
            X = self.scaler.transform(features)
        
        # Apply PCA
        if self.use_pca and self.pca is not None:
            if fit:
                X = self.pca.fit_transform(X)
                explained_var = sum(self.pca.explained_variance_ratio_)
                logger.info(f"PCA explained variance: {explained_var:.1%}")
            else:
                X = self.pca.transform(X)
        
        return X
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        """
        Generate prediction using MLP Neural Network.
        
        Args:
            features: Feature vector or matrix
            
        Returns:
            PredictionResult with probabilities
        """
        if not self._trained or self.model is None:
            logger.warning("MLP model not trained, using default predictions")
            return self._default_prediction()
        
        # Ensure 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Preprocess
        X = self._preprocess_features(features, fit=False)
        
        if self.task == "classification":
            return self._predict_classification(X)
        else:
            return self._predict_regression(X)
    
    def _predict_classification(self, X: np.ndarray) -> PredictionResult:
        """Classification prediction."""
        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        
        # Get confidence (max probability)
        confidence = float(np.max(probs))
        
        # Calculate prediction entropy (uncertainty measure)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        uncertainty = entropy / max_entropy  # Normalized entropy
        
        # Adjust confidence by uncertainty
        adjusted_confidence = confidence * (1 - uncertainty)
        
        # Build result
        if len(probs) == 3:
            return PredictionResult(
                home_win_prob=float(probs[2]),
                draw_prob=float(probs[1]),
                away_win_prob=float(probs[0]),
                confidence=float(adjusted_confidence),
                model_version=self.version,
                reasoning=f"MLP predicts: H={probs[2]:.1%}, D={probs[1]:.1%}, A={probs[0]:.1%}"
            )
        elif len(probs) == 2:
            return PredictionResult(
                home_win_prob=float(probs[1]),
                away_win_prob=float(probs[0]),
                draw_prob=0.0,
                confidence=float(adjusted_confidence),
                model_version=self.version,
                reasoning=f"MLP predicts: H={probs[1]:.1%}, A={probs[0]:.1%}"
            )
        else:
            return self._default_prediction()
    
    def _predict_regression(self, X: np.ndarray) -> PredictionResult:
        """Regression prediction."""
        prediction = self.model.predict(X)[0]
        
        # For regression, use R² as confidence proxy
        if 'r2' in self._metrics:
            confidence = max(0.5, self._metrics['r2'])
        else:
            confidence = 0.6
        
        return PredictionResult(
            home_win_prob=0.5,
            draw_prob=0.0,
            away_win_prob=0.5,
            confidence=float(confidence),
            expected_goals=float(prediction),
            model_version=self.version,
            reasoning=f"MLP regression prediction: {prediction:.2f}"
        )
    
    def _default_prediction(self) -> PredictionResult:
        """Default prediction when model not trained."""
        return PredictionResult(
            home_win_prob=0.4,
            draw_prob=0.2,
            away_win_prob=0.4,
            confidence=0.3,
            model_version=self.version,
            reasoning="MLP not trained - using default"
        )
    
    def train(
        self,
        features: List[np.ndarray],
        targets: List[Any],
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train MLP Neural Network.
        
        Args:
            features: List of feature vectors
            targets: List of target values
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        logger.info(f"Training MLP on {len(X)} samples with architecture {self._params.hidden_layer_sizes}")
        
        # Encode labels for classification
        if self.task == "classification" and self.label_encoder is not None:
            y = self.label_encoder.fit_transform(y)
        
        # Preprocess features
        X_processed = self._preprocess_features(X, fit=True)
        
        # Split data
        split_idx = int(len(X_processed) * (1 - validation_split))
        indices = np.random.permutation(len(X_processed))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train MLP
        if self.task == "classification":
            self.model = MLPClassifier(
                hidden_layer_sizes=self._params.hidden_layer_sizes,
                activation=self._params.activation,
                solver=self._params.solver,
                alpha=self._params.alpha,
                learning_rate=self._params.learning_rate,
                learning_rate_init=self._params.learning_rate_init,
                max_iter=self._params.max_iter,
                early_stopping=self._params.early_stopping,
                validation_fraction=self._params.validation_fraction,
                n_iter_no_change=self._params.n_iter_no_change,
                batch_size=self._params.batch_size,
                random_state=self._params.random_state,
                verbose=self._params.verbose,
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=self._params.hidden_layer_sizes,
                activation=self._params.activation,
                solver=self._params.solver,
                alpha=self._params.alpha,
                learning_rate=self._params.learning_rate,
                learning_rate_init=self._params.learning_rate_init,
                max_iter=self._params.max_iter,
                early_stopping=self._params.early_stopping,
                validation_fraction=self._params.validation_fraction,
                n_iter_no_change=self._params.n_iter_no_change,
                batch_size=self._params.batch_size,
                random_state=self._params.random_state,
                verbose=self._params.verbose,
            )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Store loss curve
        if hasattr(self.model, 'loss_curve_'):
            self.loss_curve_ = self.model.loss_curve_
        
        # Evaluate
        metrics = self._evaluate(X_val, y_val)
        
        # Store training info
        self._trained = True
        self._training_samples = len(features)
        self._trained_at = datetime.utcnow()
        self._metrics = metrics
        
        logger.info(f"MLP training complete. Final loss: {self.model.loss_:.4f}")
        logger.info(f"Validation accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
        
        return metrics
    
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate model on validation set."""
        metrics = {}
        
        if self.task == "classification":
            y_pred = self.model.predict(X_val)
            y_prob = self.model.predict_proba(X_val)
            
            metrics['accuracy'] = accuracy_score(y_val, y_pred)
            metrics['f1'] = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            metrics['log_loss'] = log_loss(y_val, y_prob)
            
            # Training info
            metrics['n_iterations'] = self.model.n_iter_
            metrics['final_loss'] = self.model.loss_
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            y_pred = self.model.predict(X_val)
            metrics['mse'] = mean_squared_error(y_val, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_val, y_pred)
            metrics['r2'] = r2_score(y_val, y_pred)
            metrics['n_iterations'] = self.model.n_iter_
            metrics['final_loss'] = self.model.loss_
        
        return metrics
    
    def predict_batch(self, features_list: List[np.ndarray]) -> List[PredictionResult]:
        """Predict for multiple matches."""
        return [self.predict(f) for f in features_list]
    
    def save(self, path: Path) -> bool:
        """Save model to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "model": self.model,
                "scaler": self.scaler,
                "pca": self.pca,
                "label_encoder": self.label_encoder,
                "params": self._params,
                "task": self.task,
                "use_pca": self.use_pca,
                "pca_components": self.pca_components,
                "trained": self._trained,
                "training_samples": self._training_samples,
                "trained_at": self._trained_at.isoformat() if self._trained_at else None,
                "metrics": self._metrics,
                "loss_curve": self.loss_curve_,
                "version": self.version,
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"MLP model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving MLP model: {e}")
            return False
    
    def load(self, path: Path) -> bool:
        """Load model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.pca = data["pca"]
            self.label_encoder = data["label_encoder"]
            self._params = data["params"]
            self.task = data["task"]
            self.use_pca = data["use_pca"]
            self.pca_components = data["pca_components"]
            self._trained = data["trained"]
            self._training_samples = data["training_samples"]
            self._trained_at = datetime.fromisoformat(data["trained_at"]) if data["trained_at"] else None
            self._metrics = data["metrics"]
            self.loss_curve_ = data["loss_curve"]
            
            logger.info(f"MLP model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading MLP model: {e}")
            return False
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name=self.name,
            version=self.version,
            trained_at=self._trained_at or datetime.utcnow(),
            training_samples=self._training_samples,
            metrics=self._metrics,
            feature_names=[],
        )
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return {
            "loss_curve": self.loss_curve_,
            "n_iterations": self.model.n_iter_ if self.model else 0,
            "final_loss": self.model.loss_ if self.model else 0,
        }
