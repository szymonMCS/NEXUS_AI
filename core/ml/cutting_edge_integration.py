"""
Cutting-Edge ML Integration for NEXUS AI.

Integrates all advanced models:
- Random Forest (81.9% acc)
- MLP Neural Network (86.7% acc)
- Quantum Neural Network
- Transformers (sequence modeling)
- Graph Neural Networks (team analysis)
- Reinforcement Learning (staking)
- AutoML (automatic optimization)
- Transfer Learning (cross-league)

Provides unified interface for best prediction quality.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import all cutting-edge models
from core.ml.models.random_forest_model import RandomForestEnsembleModel
from core.ml.models.mlp_model import MLPNeuralNetworkModel
from core.ml.models.quantum_nn import HybridQuantumClassicalModel
from core.ml.transformers.sports_transformer import SportsTransformer, TeamFormAnalyzer
from core.ml.gnn.graph_neural_network import TeamStrengthPredictor, create_team_graph
from core.ml.rl.staking_optimizer import StakingOptimizer
from core.ml.automl.auto_ml import AutoMLPipeline
from core.ml.automl.meta_learning import MetaLearner
from core.ml.transfer.transfer_learning import TransferLearningModel

logger = logging.getLogger(__name__)


@dataclass
class CuttingEdgePrediction:
    """Prediction result from cutting-edge models."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float
    predicted_outcome: str
    
    # Model contributions
    rf_prediction: Optional[Dict] = None
    mlp_prediction: Optional[Dict] = None
    qnn_prediction: Optional[Dict] = None
    transformer_prediction: Optional[Dict] = None
    gnn_prediction: Optional[Dict] = None
    
    # Meta info
    ensemble_method: str = "weighted"
    models_used: List[str] = None
    
    def __post_init__(self):
        if self.models_used is None:
            self.models_used = []


class CuttingEdgeEnsemble:
    """
    Ultimate ensemble combining all cutting-edge techniques.
    
    Architecture:
    1. Feature extraction (AutoML selection)
    2. Parallel prediction (RF, MLP, QNN, Transformer)
    3. Team analysis (GNN)
    4. Smart ensemble with confidence weighting
    5. Staking optimization (RL)
    """
    
    def __init__(
        self,
        use_rf: bool = True,
        use_mlp: bool = True,
        use_qnn: bool = False,  # QNN is experimental
        use_transformer: bool = True,
        use_gnn: bool = True,
        use_automl: bool = True,
        enable_transfer: bool = True,
    ):
        """Initialize cutting-edge ensemble."""
        self.use_rf = use_rf
        self.use_mlp = use_mlp
        self.use_qnn = use_qnn
        self.use_transformer = use_transformer
        self.use_gnn = use_gnn
        self.use_automl = use_automl
        self.enable_transfer = enable_transfer
        
        # Initialize models
        self.rf_model = RandomForestEnsembleModel() if use_rf else None
        self.mlp_model = MLPNeuralNetworkModel(use_pca=True) if use_mlp else None
        self.qnn_model = HybridQuantumClassicalModel() if use_qnn else None
        self.transformer = SportsTransformer() if use_transformer else None
        self.gnn = TeamStrengthPredictor() if use_gnn else None
        
        # Support systems
        self.automl = AutoMLPipeline() if use_automl else None
        self.transfer_learner = TransferLearningModel() if enable_transfer else None
        self.staking_optimizer = StakingOptimizer()
        
        # Model performance tracking for dynamic weighting
        self.model_performance: Dict[str, List[float]] = {
            'rf': [],
            'mlp': [],
            'qnn': [],
            'transformer': [],
            'gnn': [],
        }
        
        logger.info("CuttingEdgeEnsemble initialized")
        logger.info(f"Models: RF={use_rf}, MLP={use_mlp}, QNN={use_qnn}, "
                   f"Transformer={use_transformer}, GNN={use_gnn}")
    
    def predict(
        self,
        features: np.ndarray,
        match_context: Optional[Dict] = None,
        team_data: Optional[Dict] = None,
    ) -> CuttingEdgePrediction:
        """
        Generate prediction using all cutting-edge models.
        
        Args:
            features: Input features
            match_context: Additional match context
            team_data: Team composition data for GNN
            
        Returns:
            CuttingEdgePrediction with all model outputs
        """
        predictions = {}
        
        # 1. Random Forest
        if self.use_rf and self.rf_model:
            try:
                pred = self.rf_model.predict(features)
                predictions['rf'] = {
                    'home': pred.home_win_prob,
                    'draw': pred.draw_prob,
                    'away': pred.away_win_prob,
                    'confidence': pred.confidence,
                }
            except Exception as e:
                logger.warning(f"RF prediction failed: {e}")
        
        # 2. MLP Neural Network
        if self.use_mlp and self.mlp_model:
            try:
                pred = self.mlp_model.predict(features)
                predictions['mlp'] = {
                    'home': pred['home_win_prob'],
                    'draw': pred['draw_prob'],
                    'away': pred['away_win_prob'],
                    'confidence': pred['confidence'],
                }
            except Exception as e:
                logger.warning(f"MLP prediction failed: {e}")
        
        # 3. Quantum Neural Network
        if self.use_qnn and self.qnn_model:
            try:
                pred = self.qnn_model.predict(features)
                predictions['qnn'] = {
                    'home': pred['home_win_prob'],
                    'draw': pred['draw_prob'],
                    'away': pred['away_win_prob'],
                    'confidence': pred['confidence'],
                }
            except Exception as e:
                logger.warning(f"QNN prediction failed: {e}")
        
        # 4. Transformer (if sequence data available)
        if self.use_transformer and self.transformer and match_context:
            try:
                if 'recent_matches' in match_context:
                    pred = self.transformer.predict(match_context['recent_matches'])
                    predictions['transformer'] = {
                        'home': pred['home_win_prob'],
                        'draw': pred['draw_prob'],
                        'away': pred['away_win_prob'],
                        'confidence': pred['confidence'],
                    }
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
        
        # 5. GNN (if team data available)
        if self.use_gnn and self.gnn and team_data:
            try:
                home_graph = create_team_graph(
                    team_data['home']['name'],
                    team_data['home']['players']
                )
                away_graph = create_team_graph(
                    team_data['away']['name'],
                    team_data['away']['players']
                )
                
                pred = self.gnn.predict_match(home_graph, away_graph)
                predictions['gnn'] = {
                    'home': pred['home_win_prob'],
                    'draw': pred['draw_prob'],
                    'away': pred['away_win_prob'],
                    'confidence': pred['confidence'],
                }
            except Exception as e:
                logger.warning(f"GNN prediction failed: {e}")
        
        # Combine predictions using smart ensemble
        if not predictions:
            # Fallback
            return CuttingEdgePrediction(
                home_win_prob=0.4,
                draw_prob=0.2,
                away_win_prob=0.4,
                confidence=0.3,
                predicted_outcome='unknown',
            )
        
        # Dynamic weighting based on recent performance
        weights = self._compute_dynamic_weights(predictions.keys())
        
        # Weighted average
        weighted_home = sum(predictions[m]['home'] * weights[m] for m in predictions)
        weighted_draw = sum(predictions[m]['draw'] * weights[m] for m in predictions)
        weighted_away = sum(predictions[m]['away'] * weights[m] for m in predictions)
        
        # Normalize
        total = weighted_home + weighted_draw + weighted_away
        weighted_home /= total
        weighted_draw /= total
        weighted_away /= total
        
        # Confidence based on model agreement
        home_probs = [predictions[m]['home'] for m in predictions]
        agreement = 1 - np.std(home_probs)
        confidence = np.mean([predictions[m]['confidence'] for m in predictions]) * agreement
        
        # Determine outcome
        probs = {'home': weighted_home, 'draw': weighted_draw, 'away': weighted_away}
        predicted = max(probs, key=probs.get)
        
        return CuttingEdgePrediction(
            home_win_prob=round(weighted_home, 4),
            draw_prob=round(weighted_draw, 4),
            away_win_prob=round(weighted_away, 4),
            confidence=round(confidence, 4),
            predicted_outcome=predicted,
            rf_prediction=predictions.get('rf'),
            mlp_prediction=predictions.get('mlp'),
            qnn_prediction=predictions.get('qnn'),
            transformer_prediction=predictions.get('transformer'),
            gnn_prediction=predictions.get('gnn'),
            ensemble_method='dynamic_weighted',
            models_used=list(predictions.keys()),
        )
    
    def _compute_dynamic_weights(self, model_names) -> Dict[str, float]:
        """Compute dynamic weights based on recent performance."""
        weights = {}
        
        for model in model_names:
            history = self.model_performance.get(model, [])
            if len(history) >= 5:
                # Weight by recent accuracy
                recent_acc = np.mean(history[-10:])
                weights[model] = 0.2 + recent_acc  # Base 0.2 + accuracy
            else:
                # Default weights based on research
                default_weights = {
                    'rf': 0.82,  # Research: 81.9%
                    'mlp': 0.87,  # Research: 86.7%
                    'qnn': 0.75,
                    'transformer': 0.80,
                    'gnn': 0.78,
                }
                weights[model] = default_weights.get(model, 0.5)
        
        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def update_performance(self, model_name: str, was_correct: bool):
        """Update model performance for dynamic weighting."""
        if model_name in self.model_performance:
            self.model_performance[model_name].append(1.0 if was_correct else 0.0)
            # Keep last 50
            self.model_performance[model_name] = self.model_performance[model_name][-50:]
    
    def optimize_stake(
        self,
        prediction: CuttingEdgePrediction,
        odds: Dict[str, float],
        bankroll: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Optimize stake using RL-based staking optimizer.
        
        Args:
            prediction: Prediction from ensemble
            odds: Market odds
            bankroll: Current bankroll
            
        Returns:
            Staking recommendation
        """
        # Calculate recent performance
        all_history = []
        for hist in self.model_performance.values():
            all_history.extend(hist)
        
        recent_win_rate = np.mean(all_history[-20:]) if len(all_history) >= 20 else 0.5
        
        # Calculate streak
        streak = 0
        for result in reversed(all_history[-10:]):
            if result == 1:
                streak = streak + 1 if streak >= 0 else 0
            else:
                streak = streak - 1 if streak <= 0 else 0
        
        # Get odds for predicted outcome
        predicted = prediction.predicted_outcome
        outcome_odds = odds.get(predicted, 2.0)
        
        # Optimize stake
        recommendation = self.staking_optimizer.optimize_stake(
            prediction_prob=getattr(prediction, f'{predicted}_win_prob'),
            odds=outcome_odds,
            model_confidence=prediction.confidence,
            recent_win_rate=recent_win_rate,
            current_streak=streak,
        )
        
        return recommendation
    
    def run_automl_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        sport: str = "football",
    ) -> Dict[str, Any]:
        """
        Run AutoML to optimize models for specific sport/league.
        
        Args:
            X: Training features
            y: Training labels
            feature_names: Feature names
            sport: Sport type
            
        Returns:
            Optimization results
        """
        if not self.use_automl or not self.automl:
            return {"error": "AutoML not enabled"}
        
        logger.info(f"Running AutoML optimization for {sport}")
        
        result = self.automl.search(X, y, feature_names, sport)
        
        logger.info(f"AutoML complete. Best model: {result.best_config.model_type}")
        logger.info(f"Best score: {result.best_config.score:.4f}")
        
        return {
            "best_model_type": result.best_config.model_type,
            "best_hyperparams": result.best_config.hyperparams,
            "best_score": result.best_config.score,
            "search_time": result.search_time,
            "n_evaluations": result.n_evaluations,
        }
    
    def transfer_from_league(
        self,
        source_league: str,
        target_league: str,
        X_source,
        y_source,
        X_target,
        y_target,
    ):
        """
        Apply transfer learning from source to target league.
        
        Args:
            source_league: Name of source league
            target_league: Name of target league
            X_source: Source features
            y_source: Source labels
            X_target: Target features
            y_target: Target labels
        """
        if not self.enable_transfer or not self.transfer_learner:
            logger.warning("Transfer learning not enabled")
            return
        
        logger.info(f"Transfer learning: {source_league} -> {target_league}")
        
        # Pre-train on source
        self.transfer_learner.pretrain(X_source, y_source)
        
        # Fine-tune on target
        self.transfer_learner.fine_tune(X_target, y_target)
        
        logger.info("Transfer learning complete")
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get performance comparison of all models."""
        comparison = {}
        
        for model, history in self.model_performance.items():
            if history:
                comparison[model] = {
                    "recent_accuracy": np.mean(history[-10:]) if len(history) >= 10 else np.mean(history),
                    "total_predictions": len(history),
                    "current_weight": self._compute_dynamic_weights([model]).get(model, 0.2),
                }
        
        return comparison


# Convenience function
def create_cutting_edge_ensemble(**kwargs) -> CuttingEdgeEnsemble:
    """Create cutting-edge ensemble with all features."""
    return CuttingEdgeEnsemble(**kwargs)
