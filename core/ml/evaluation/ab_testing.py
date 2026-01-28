"""
A/B Testing Framework for Model Comparison.

Comprehensive testing system comparing old vs new models
with statistical significance testing.
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of A/B test between two models."""
    test_id: str
    model_a_name: str
    model_b_name: str
    n_a: int
    n_b: int
    accuracy_a: float
    accuracy_b: float
    roi_a: float
    roi_b: float
    p_value_accuracy: float
    p_value_roi: float
    winner: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value_accuracy < alpha or self.p_value_roi < alpha


@dataclass
class PredictionRecord:
    """Single prediction record for analysis."""
    record_id: str
    match_id: str
    model_name: str
    test_group: str
    predicted_outcome: str
    predicted_prob: float
    confidence: float
    actual_outcome: Optional[str] = None
    was_correct: Optional[bool] = None
    profit_loss: float = 0.0
    sport: str = ""
    league: str = ""
    odds_taken: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return data


class ABTestingFramework:
    """Comprehensive A/B Testing Framework for ML models."""
    
    def __init__(self, storage_path: str = "data/ab_tests"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, PredictionRecord] = {}
        self._load_data()
        logger.info("A/B Testing Framework initialized")
    
    def _load_data(self):
        """Load existing predictions and tests."""
        cutoff = datetime.utcnow() - timedelta(days=30)
        
        for file_path in self.storage_path.glob("predictions_*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            record = PredictionRecord(**data)
                            if record.created_at >= cutoff:
                                self.predictions[record.record_id] = record
                        except:
                            continue
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.predictions)} prediction records")
    
    def start_test(
        self,
        model_a_name: str,
        model_b_name: str,
        test_name: Optional[str] = None,
        target_samples: int = 100,
        min_samples_for_significance: int = 30,
    ) -> str:
        """Start new A/B test."""
        test_id = str(uuid.uuid4())[:8]
        
        test_config = {
            "test_id": test_id,
            "test_name": test_name or f"{model_a_name}_vs_{model_b_name}",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "target_samples": target_samples,
            "min_samples": min_samples_for_significance,
            "started_at": datetime.utcnow().isoformat(),
            "status": "active",
            "samples_a": 0,
            "samples_b": 0,
        }
        
        self.active_tests[test_id] = test_config
        
        logger.info(f"Started A/B test {test_id}: {model_a_name} vs {model_b_name}")
        logger.info(f"  Target samples: {target_samples}")
        
        return test_id
    
    def assign_group(self, test_id: str) -> str:
        """Randomly assign prediction to test group."""
        if test_id not in self.active_tests:
            return 'A'
        
        test = self.active_tests[test_id]
        total = test["samples_a"] + test["samples_b"]
        
        if total == 0:
            group = 'A' if np.random.random() < 0.5 else 'B'
        else:
            prop_a = test["samples_a"] / total if total > 0 else 0.5
            group = 'B' if prop_a > 0.5 else 'A'
        
        if group == 'A':
            test["samples_a"] += 1
        else:
            test["samples_b"] += 1
        
        return group
    
    def record_prediction(
        self,
        test_id: str,
        model_name: str,
        group: str,
        match_id: str,
        predicted_outcome: str,
        predicted_prob: float,
        confidence: float,
        sport: str = "",
        league: str = "",
        odds: float = 0.0,
    ) -> str:
        """Record a prediction for A/B testing."""
        record_id = str(uuid.uuid4())
        
        record = PredictionRecord(
            record_id=record_id,
            match_id=match_id,
            model_name=model_name,
            test_group=group,
            predicted_outcome=predicted_outcome,
            predicted_prob=predicted_prob,
            confidence=confidence,
            sport=sport,
            league=league,
            odds_taken=odds,
        )
        
        self.predictions[record_id] = record
        self._save_prediction(record)
        
        return record_id
    
    def _save_prediction(self, record: PredictionRecord):
        """Save prediction to disk."""
        try:
            month_str = record.created_at.strftime("%Y-%m")
            file_path = self.storage_path / f"predictions_{month_str}.jsonl"
            
            with open(file_path, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
    
    def resolve_prediction(
        self,
        record_id: str,
        actual_outcome: str,
        profit_loss: float = 0.0,
    ) -> bool:
        """Resolve prediction with actual result."""
        if record_id not in self.predictions:
            return False
        
        record = self.predictions[record_id]
        record.actual_outcome = actual_outcome
        record.profit_loss = profit_loss
        record.resolved_at = datetime.utcnow()
        record.was_correct = (record.predicted_outcome == actual_outcome)
        
        self._update_prediction_file(record)
        return True
    
    def _update_prediction_file(self, record: PredictionRecord):
        """Update prediction in file storage."""
        try:
            month_str = record.created_at.strftime("%Y-%m")
            file_path = self.storage_path / f"predictions_{month_str}.jsonl"
            
            if not file_path.exists():
                return
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    if data['record_id'] == record.record_id:
                        lines[i] = json.dumps(record.to_dict()) + '\n'
                        updated = True
                        break
                except:
                    continue
            
            if updated:
                with open(file_path, 'w') as f:
                    f.writelines(lines)
        except Exception as e:
            logger.error(f"Error updating prediction file: {e}")
    
    def analyze_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze A/B test results."""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        model_a = test["model_a"]
        model_b = test["model_b"]
        
        test_predictions = [
            p for p in self.predictions.values()
            if p.model_name in [model_a, model_b] and p.was_correct is not None
        ]
        
        group_a = [p for p in test_predictions if p.test_group == 'A']
        group_b = [p for p in test_predictions if p.test_group == 'B']
        
        n_a = len(group_a)
        n_b = len(group_b)
        
        if n_a < test["min_samples"] or n_b < test["min_samples"]:
            logger.info(f"Insufficient data: A={n_a}, B={n_b}")
            return None
        
        correct_a = sum(1 for p in group_a if p.was_correct)
        correct_b = sum(1 for p in group_b if p.was_correct)
        
        acc_a = correct_a / n_a if n_a > 0 else 0
        acc_b = correct_b / n_b if n_b > 0 else 0
        
        roi_a = sum(p.profit_loss for p in group_a)
        roi_b = sum(p.profit_loss for p in group_b)
        
        # Statistical tests
        if SCIPY_AVAILABLE:
            successes_a = [1 if p.was_correct else 0 for p in group_a]
            successes_b = [1 if p.was_correct else 0 for p in group_b]
            _, p_value_acc = stats.ttest_ind(successes_a, successes_b)
            
            profits_a = [p.profit_loss for p in group_a]
            profits_b = [p.profit_loss for p in group_b]
            _, p_value_roi = stats.ttest_ind(profits_a, profits_b)
        else:
            p_value_acc = 0.5
            p_value_roi = 0.5
        
        alpha = 0.05
        if p_value_acc < alpha and acc_b > acc_a:
            winner = 'B'
            confidence = 1 - p_value_acc
        elif p_value_acc < alpha and acc_a > acc_b:
            winner = 'A'
            confidence = 1 - p_value_acc
        else:
            winner = 'tie'
            confidence = 1 - p_value_acc
        
        result = ABTestResult(
            test_id=test_id,
            model_a_name=model_a,
            model_b_name=model_b,
            n_a=n_a,
            n_b=n_b,
            accuracy_a=acc_a,
            accuracy_b=acc_b,
            roi_a=roi_a,
            roi_b=roi_b,
            p_value_accuracy=p_value_acc,
            p_value_roi=p_value_roi,
            winner=winner,
            confidence=confidence,
        )
        
        self._save_test_result(result)
        return result
    
    def _save_test_result(self, result: ABTestResult):
        """Save test result to disk."""
        try:
            file_path = self.storage_path / "test_results.jsonl"
            with open(file_path, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving test result: {e}")
    
    def get_test_report(self, test_id: str) -> str:
        """Generate text report for test."""
        result = self.analyze_test(test_id)
        
        if result is None:
            return "Insufficient data for analysis"
        
        lines = [
            "=" * 70,
            f"A/B Test Report: {result.model_a_name} vs {result.model_b_name}",
            f"Test ID: {test_id}",
            f"Analysis Date: {result.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
            f"Samples: A={result.n_a}, B={result.n_b}",
            f"Accuracy: A={result.accuracy_a:.1%}, B={result.accuracy_b:.1%}",
            f"Difference: {result.accuracy_b - result.accuracy_a:+.1%}",
            f"P-value: {result.p_value_accuracy:.4f}",
            "",
            f"ROI: A=${result.roi_a:+.2f}, B=${result.roi_b:+.2f}",
            "",
            f"Winner: {result.winner} ({result.confidence:.1%} confidence)",
            f"Significant: {'Yes' if result.is_significant else 'No'}",
            "=" * 70,
        ]
        
        return '\n'.join(lines)


def run_quick_comparison(
    model_a_predictions: List[bool],
    model_b_predictions: List[bool],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict[str, Any]:
    """Quick statistical comparison of two models."""
    n_a = len(model_a_predictions)
    n_b = len(model_b_predictions)
    
    acc_a = np.mean(model_a_predictions)
    acc_b = np.mean(model_b_predictions)
    
    if SCIPY_AVAILABLE:
        _, p_value = stats.ttest_ind(
            [1 if x else 0 for x in model_a_predictions],
            [1 if x else 0 for x in model_b_predictions]
        )
    else:
        p_value = 0.5
    
    return {
        "model_a": {"name": model_a_name, "accuracy": acc_a, "n": n_a},
        "model_b": {"name": model_b_name, "accuracy": acc_b, "n": n_b},
        "difference": acc_b - acc_a,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "winner": model_b_name if acc_b > acc_a and p_value < 0.05 else 
                  model_a_name if acc_a > acc_b and p_value < 0.05 else "tie",
    }
