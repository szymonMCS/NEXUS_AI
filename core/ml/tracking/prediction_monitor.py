"""
Prediction Quality Monitoring System.

Tracks and evaluates prediction accuracy over time.
Provides feedback for model retraining.

Based on sport_datasets_AI_report.md - Model evaluation techniques:
- Accuracy tracking
- ROI monitoring
- Calibration analysis
- Backtesting framework
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class BetResult(str, Enum):
    """Result of a bet."""
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"


@dataclass
class TrackedPrediction:
    """Single tracked prediction."""
    prediction_id: str
    match_id: str
    sport: str
    model_version: str
    model_type: str  # goals, handicap, hybrid
    
    # Prediction details
    predicted_outcome: str
    predicted_prob: float
    confidence: float
    
    # Market info
    market_odds: float
    edge: float
    recommended_stake: float
    
    # Actual result
    actual_outcome: Optional[str] = None
    actual_score: Optional[tuple] = None
    bet_result: BetResult = BetResult.PENDING
    profit_loss: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        data['bet_result'] = self.bet_result.value
        return data


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_type: str
    model_version: str
    sport: str
    
    # Sample size
    total_predictions: int = 0
    resolved_predictions: int = 0
    
    # Accuracy metrics
    correct_predictions: int = 0
    accuracy: float = 0.0
    brier_score: float = 0.0
    
    # Betting metrics
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    
    # Financial metrics
    total_stake: float = 0.0
    total_return: float = 0.0
    roi: float = 0.0
    profit: float = 0.0
    
    # By confidence level
    high_conf_accuracy: float = 0.0  # > 70%
    med_conf_accuracy: float = 0.0   # 50-70%
    low_conf_accuracy: float = 0.0   # < 50%
    
    # Time period
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['from_date'] = self.from_date.isoformat() if self.from_date else None
        data['to_date'] = self.to_date.isoformat() if self.to_date else None
        return data


class PredictionMonitor:
    """
    Monitors prediction quality and model performance.
    
    Usage:
        monitor = PredictionMonitor()
        
        # Track a prediction
        monitor.track_prediction(prediction)
        
        # Resolve when result known
        monitor.resolve_prediction(prediction_id, actual_outcome, profit)
        
        # Get performance report
        performance = monitor.get_performance(model_type="goals", days=30)
    """
    
    def __init__(self, storage_path: str = "data/tracking"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._predictions: Dict[str, TrackedPrediction] = {}
        self._load_predictions()
    
    def _get_storage_file(self) -> Path:
        """Get storage file for current date."""
        today = datetime.now().strftime("%Y-%m")
        return self.storage_path / f"predictions_{today}.jsonl"
    
    def _load_predictions(self):
        """Load recent predictions from storage."""
        # Load last 3 months
        for i in range(3):
            date = datetime.now() - timedelta(days=i*30)
            file_path = self.storage_path / f"predictions_{date.strftime('%Y-%m')}.jsonl"
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                pred = TrackedPrediction(**data)
                                self._predictions[pred.prediction_id] = pred
                            except:
                                continue
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(self._predictions)} tracked predictions")
    
    def track_prediction(self, prediction: TrackedPrediction) -> bool:
        """Track a new prediction."""
        self._predictions[prediction.prediction_id] = prediction
        
        # Save to storage
        try:
            storage_file = self._get_storage_file()
            with open(storage_file, 'a') as f:
                f.write(json.dumps(prediction.to_dict()) + '\n')
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def resolve_prediction(
        self,
        prediction_id: str,
        actual_outcome: str,
        profit_loss: float = 0.0,
        actual_score: Optional[tuple] = None,
    ) -> bool:
        """Resolve a prediction with actual result."""
        pred = self._predictions.get(prediction_id)
        if not pred:
            logger.warning(f"Prediction not found: {prediction_id}")
            return False
        
        pred.actual_outcome = actual_outcome
        pred.actual_score = actual_score
        pred.profit_loss = profit_loss
        pred.resolved_at = datetime.utcnow()
        
        # Determine bet result
        if profit_loss > 0:
            pred.bet_result = BetResult.WIN
        elif profit_loss < 0:
            pred.bet_result = BetResult.LOSS
        elif profit_loss == 0:
            pred.bet_result = BetResult.PUSH
        
        # Update storage
        return self._update_storage(pred)
    
    def _update_storage(self, prediction: TrackedPrediction) -> bool:
        """Update prediction in storage."""
        try:
            storage_file = self._get_storage_file()
            
            # Read all predictions
            lines = []
            if storage_file.exists():
                with open(storage_file, 'r') as f:
                    lines = f.readlines()
            
            # Update the prediction
            updated = False
            for i, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    if data['prediction_id'] == prediction.prediction_id:
                        lines[i] = json.dumps(prediction.to_dict()) + '\n'
                        updated = True
                        break
                except:
                    continue
            
            # Write back
            if updated:
                with open(storage_file, 'w') as f:
                    f.writelines(lines)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update storage: {e}")
            return False
    
    def get_performance(
        self,
        model_type: Optional[str] = None,
        sport: Optional[str] = None,
        model_version: Optional[str] = None,
        days: int = 30,
    ) -> ModelPerformance:
        """Get performance metrics for specified filters."""
        
        # Filter predictions
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filtered = [
            p for p in self._predictions.values()
            if p.created_at >= cutoff_date
            and (not model_type or p.model_type == model_type)
            and (not sport or p.sport == sport)
            and (not model_version or p.model_version == model_version)
            and p.bet_result != BetResult.PENDING
        ]
        
        if not filtered:
            return ModelPerformance(
                model_type=model_type or "all",
                model_version=model_version or "all",
                sport=sport or "all",
                from_date=cutoff_date,
                to_date=datetime.utcnow(),
            )
        
        # Calculate metrics
        total = len(filtered)
        resolved = [p for p in filtered if p.bet_result != BetResult.PENDING]
        
        # Accuracy
        correct = sum(1 for p in resolved if p.predicted_outcome == p.actual_outcome)
        accuracy = correct / len(resolved) if resolved else 0.0
        
        # Brier score
        brier_scores = []
        for p in resolved:
            if p.predicted_outcome == p.actual_outcome:
                brier_scores.append((1 - p.predicted_prob) ** 2)
            else:
                brier_scores.append((0 - p.predicted_prob) ** 2)
        brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0
        
        # Betting metrics
        bets = [p for p in resolved if p.recommended_stake > 0]
        wins = sum(1 for p in bets if p.bet_result == BetResult.WIN)
        losses = sum(1 for p in bets if p.bet_result == BetResult.LOSS)
        pushes = sum(1 for p in bets if p.bet_result == BetResult.PUSH)
        
        total_stake = sum(p.recommended_stake for p in bets)
        total_return = sum(p.profit_loss + p.recommended_stake for p in bets if p.bet_result != BetResult.PENDING)
        profit = sum(p.profit_loss for p in bets)
        roi = (profit / total_stake) if total_stake > 0 else 0.0
        
        # By confidence
        high_conf = [p for p in resolved if p.confidence >= 0.7]
        med_conf = [p for p in resolved if 0.5 <= p.confidence < 0.7]
        low_conf = [p for p in resolved if p.confidence < 0.5]
        
        high_acc = sum(1 for p in high_conf if p.predicted_outcome == p.actual_outcome) / len(high_conf) if high_conf else 0.0
        med_acc = sum(1 for p in med_conf if p.predicted_outcome == p.actual_outcome) / len(med_conf) if med_conf else 0.0
        low_acc = sum(1 for p in low_conf if p.predicted_outcome == p.actual_outcome) / len(low_conf) if low_conf else 0.0
        
        return ModelPerformance(
            model_type=model_type or "all",
            model_version=model_version or "all",
            sport=sport or "all",
            total_predictions=total,
            resolved_predictions=len(resolved),
            correct_predictions=correct,
            accuracy=accuracy,
            brier_score=brier,
            total_bets=len(bets),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            total_stake=total_stake,
            total_return=total_return,
            roi=roi,
            profit=profit,
            high_conf_accuracy=high_acc,
            med_conf_accuracy=med_acc,
            low_conf_accuracy=low_acc,
            from_date=cutoff_date,
            to_date=datetime.utcnow(),
        )
    
    def get_retraining_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for model retraining."""
        recommendations = []
        
        # Check each sport and model type
        for sport in ["football", "basketball", "tennis", "hockey", "baseball", "handball"]:
            for model_type in ["goals", "handicap", "hybrid"]:
                perf = self.get_performance(model_type=model_type, sport=sport, days=30)
                
                if perf.total_predictions < 10:
                    continue
                
                needs_retrain = False
                reasons = []
                
                if perf.accuracy < 0.5 and perf.resolved_predictions >= 20:
                    needs_retrain = True
                    reasons.append(f"Low accuracy: {perf.accuracy:.1%}")
                
                if perf.roi < -0.1 and perf.total_bets >= 10:
                    needs_retrain = True
                    reasons.append(f"Negative ROI: {perf.roi:.1%}")
                
                if perf.brier_score > 0.25:
                    needs_retrain = True
                    reasons.append(f"Poor calibration: {perf.brier_score:.3f}")
                
                if needs_retrain:
                    recommendations.append({
                        "sport": sport,
                        "model_type": model_type,
                        "reasons": reasons,
                        "current_performance": perf.to_dict(),
                    })
        
        return {
            "needs_retraining": len(recommendations) > 0,
            "recommendations": recommendations,
            "checked_at": datetime.utcnow().isoformat(),
        }
    
    def generate_report(self, days: int = 30) -> str:
        """Generate a text report of model performance."""
        lines = [
            "=" * 70,
            "NEXUS AI - Prediction Performance Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Period: Last {days} days",
            "=" * 70,
            "",
        ]
        
        # Overall performance
        overall = self.get_performance(days=days)
        lines.extend([
            "OVERALL PERFORMANCE:",
            f"  Total Predictions: {overall.total_predictions}",
            f"  Resolved: {overall.resolved_predictions}",
            f"  Accuracy: {overall.accuracy:.1%}",
            f"  Brier Score: {overall.brier_score:.4f}",
            "",
            "BETTING PERFORMANCE:",
            f"  Total Bets: {overall.total_bets}",
            f"  Wins: {overall.wins} | Losses: {overall.losses} | Pushes: {overall.pushes}",
            f"  Win Rate: {overall.win_rate:.1%}",
            f"  Total Stake: ${overall.total_stake:.2f}",
            f"  Profit/Loss: ${overall.profit:.2f}",
            f"  ROI: {overall.roi:.1%}",
            "",
        ])
        
        # By sport
        lines.append("PERFORMANCE BY SPORT:")
        for sport in ["football", "basketball", "tennis", "hockey", "baseball", "handball"]:
            perf = self.get_performance(sport=sport, days=days)
            if perf.total_predictions > 0:
                lines.append(f"  {sport.upper():12} | Acc: {perf.accuracy:.1%} | ROI: {perf.roi:+.1%} | N: {perf.total_predictions}")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        return '\n'.join(lines)
