"""
Football Evaluator Agent - evaluates football data quality.
"""

import json
import statistics
from typing import Any, Dict, List, Optional
from datetime import datetime
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class FootballEvaluatorAgent(BaseAgent):
    """Evaluator for football/soccer data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("FootballEvaluatorAgent", config)
        self.sport = "football"
        self.required_fields = [
            'match_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals',
            'home_shots', 'away_shots', 'home_corners', 'away_corners'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute evaluation task."""
        action = task.get('action', 'evaluate')
        
        if action == 'evaluate':
            return await self._evaluate_dataset(task)
        elif action == 'summarize':
            return await self._summarize_evaluation(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _evaluate_dataset(self, task: Dict[str, Any]) -> TaskResult:
        """Evaluate dataset quality."""
        data = task.get('data', {})
        dataset_path = task.get('dataset_path', '')
        
        records = data.get('records', [])
        if not records:
            records = self._load_dataset(dataset_path)
        
        if not records:
            return TaskResult(success=False, error="No records to evaluate")
        
        logger.info(f"[{self.name}] Evaluating {len(records)} {self.sport} records")
        
        metrics = self._calculate_metrics(records)
        quality_score = self._calculate_quality_score(metrics)
        
        report = {
            'sport': self.sport,
            'evaluated_at': datetime.now().isoformat(),
            'record_count': len(records),
            'quality_score': quality_score,
            'quality_grade': self._grade_quality(quality_score),
            'metrics': metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        import os
        eval_dir = os.path.join(os.getcwd(), 'datasets', 'sports_data', 'evaluated')
        os.makedirs(eval_dir, exist_ok=True)
        report_path = os.path.join(eval_dir, f"{self.sport}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"[{self.name}] Quality score: {quality_score:.2f}/100 ({report['quality_grade']})")
        
        return TaskResult(
            success=True,
            data=report,
            records_processed=len(records),
            metadata={'quality_score': quality_score, 'report_path': report_path}
        )
    
    async def _summarize_evaluation(self, task: Dict[str, Any]) -> TaskResult:
        """Summarize evaluation results."""
        data = task.get('data', {})
        
        summary = {
            'sport': self.sport,
            'total_records': data.get('storage', {}).get('record_count', 0),
            'format': data.get('storage', {}).get('format', 'unknown'),
            'sources': data.get('acquisition', {}).get('raw_count', 0),
            'filepath': data.get('storage', {}).get('filepath', '')
        }
        
        return TaskResult(success=True, data=summary)
    
    def _load_dataset(self, path: str) -> List[Dict]:
        """Load dataset from file."""
        if not path:
            return []
        
        try:
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return data.get('records', [])
            elif path.endswith('.csv'):
                import csv
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
        
        return []
    
    def _calculate_metrics(self, records: List[Dict]) -> Dict:
        """Calculate football-specific metrics."""
        completeness = self._calculate_completeness(records, self.required_fields)
        consistency = self._calculate_football_consistency(records)
        
        # Goal statistics
        total_goals = [r.get('home_goals', 0) + r.get('away_goals', 0) for r in records if r.get('home_goals') is not None]
        
        # xG availability
        xg_available = sum(1 for r in records if r.get('home_xg') is not None)
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'avg_total_goals': statistics.mean(total_goals) if total_goals else 0,
            'xg_coverage': xg_available / len(records) if records else 0,
            'home_win_rate': sum(1 for r in records if r.get('home_goals', 0) > r.get('away_goals', 0)) / len(records) if records else 0,
            'draw_rate': sum(1 for r in records if r.get('home_goals', 0) == r.get('away_goals', 0)) / len(records) if records else 0
        }
    
    def _calculate_completeness(self, records: List[Dict], required_fields: List[str]) -> Dict:
        """Calculate field completeness."""
        if not records:
            return {'overall': 0, 'field_rates': {}}
        
        field_rates = {}
        for field in required_fields:
            present = sum(1 for r in records if field in r and r[field] is not None and r[field] != '')
            field_rates[field] = present / len(records)
        
        overall = sum(field_rates.values()) / len(field_rates) if field_rates else 0
        
        return {'overall': overall, 'field_rates': field_rates}
    
    def _calculate_football_consistency(self, records: List[Dict]) -> Dict:
        """Check football data consistency."""
        issues = []
        
        for i, r in enumerate(records):
            home_goals = r.get('home_goals', 0)
            away_goals = r.get('away_goals', 0)
            
            # Check for unrealistic scores
            if home_goals < 0 or away_goals < 0:
                issues.append(f"Record {i}: Negative goals")
            if home_goals > 15 or away_goals > 15:
                issues.append(f"Record {i}: Unusually high score")
            
            # Check shots vs goals
            home_shots = r.get('home_shots', 0)
            if home_shots > 0 and home_goals > home_shots:
                issues.append(f"Record {i}: More goals than shots")
            
            # Check possession sums to ~100
            home_poss = r.get('home_possession', 0)
            away_poss = r.get('away_possession', 0)
            if home_poss > 0 and away_poss > 0 and abs(home_poss + away_poss - 100) > 5:
                issues.append(f"Record {i}: Possession doesn't sum to 100")
        
        return {'rate': 1 - min(len(issues) / len(records), 1) if records else 0, 'issues': issues[:10]}
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score."""
        completeness = metrics.get('completeness', {}).get('overall', 0)
        consistency = metrics.get('consistency', {}).get('rate', 0)
        
        score = (completeness * 0.6 + consistency * 0.4) * 100
        return round(score, 2)
    
    def _grade_quality(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Acceptable)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Unusable)'
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        completeness = metrics.get('completeness', {}).get('overall', 0)
        if completeness < 0.8:
            missing = [f for f, r in metrics.get('completeness', {}).get('field_rates', {}).items() if r < 0.8]
            recommendations.append(f"Improve completeness for fields: {missing[:5]}")
        
        xg_coverage = metrics.get('xg_coverage', 0)
        if xg_coverage < 0.5:
            recommendations.append("Add xG (expected goals) data for better predictions")
        
        return recommendations if recommendations else ['No major issues found']
