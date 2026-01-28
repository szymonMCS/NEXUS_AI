"""
Evaluator Agents - Evaluate data quality for each sport.
Each evaluator checks sport-specific data quality metrics.
"""

import json
import statistics
from typing import Any, Dict, List, Optional
from datetime import datetime
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class BaseEvaluatorAgent(BaseAgent):
    """Base class for sport evaluators."""
    
    def __init__(self, name: str, sport: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.sport = sport
        
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
            # Try to load from file
            records = self._load_dataset(dataset_path)
        
        if not records:
            return TaskResult(success=False, error="No records to evaluate")
        
        logger.info(f"[{self.name}] Evaluating {len(records)} {self.sport} records")
        
        # Calculate metrics
        metrics = self._calculate_metrics(records)
        
        # Quality score
        quality_score = self._calculate_quality_score(metrics)
        
        # Generate report
        report = {
            'sport': self.sport,
            'evaluated_at': datetime.now().isoformat(),
            'record_count': len(records),
            'quality_score': quality_score,
            'quality_grade': self._grade_quality(quality_score),
            'metrics': metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        # Save report
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
    
    def _calculate_consistency(self, records: List[Dict]) -> Dict:
        """Calculate data consistency."""
        # Check for logical inconsistencies
        issues = []
        
        for i, r in enumerate(records):
            # Add sport-specific checks in subclasses
            pass
        
        consistency_rate = 1 - (len(issues) / len(records)) if records else 0
        
        return {'rate': consistency_rate, 'issues_found': len(issues), 'sample_issues': issues[:5]}
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score."""
        completeness = metrics.get('completeness', {}).get('overall', 0)
        consistency = metrics.get('consistency', {}).get('rate', 0)
        
        # Weighted average
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
        
        consistency = metrics.get('consistency', {}).get('rate', 0)
        if consistency < 0.9:
            recommendations.append("Address data consistency issues")
        
        return recommendations if recommendations else ['No major issues found']


class BasketballEvaluatorAgent(BaseEvaluatorAgent):
    """Evaluator for basketball data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("BasketballEvaluatorAgent", "basketball", config)
        self.required_fields = [
            'game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score',
            'home_fg_made', 'home_fg_attempts', 'home_rebounds', 'home_assists',
            'away_fg_made', 'away_fg_attempts', 'away_rebounds', 'away_assists'
        ]
    
    def _calculate_metrics(self, records: List[Dict]) -> Dict:
        """Calculate basketball-specific metrics."""
        completeness = self._calculate_completeness(records, self.required_fields)
        consistency = self._calculate_basketball_consistency(records)
        
        # Statistical distribution
        scores = [r.get('home_score', 0) + r.get('away_score', 0) for r in records if r.get('home_score') and r.get('away_score')]
        stats_dist = {
            'avg_total_score': statistics.mean(scores) if scores else 0,
            'score_std': statistics.stdev(scores) if len(scores) > 1 else 0,
            'score_range': (min(scores), max(scores)) if scores else (0, 0)
        }
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'statistical_distribution': stats_dist,
            'home_win_rate': sum(1 for r in records if r.get('home_score', 0) > r.get('away_score', 0)) / len(records) if records else 0
        }
    
    def _calculate_basketball_consistency(self, records: List[Dict]) -> Dict:
        """Check basketball data consistency."""
        issues = []
        
        for i, r in enumerate(records):
            home_score = r.get('home_score', 0)
            away_score = r.get('away_score', 0)
            
            # Check for unrealistic scores
            if home_score < 0 or away_score < 0:
                issues.append(f"Record {i}: Negative score")
            if home_score > 200 or away_score > 200:
                issues.append(f"Record {i}: Unrealistic high score")
            
            # Check quarter totals
            home_q_total = sum(r.get(f'home_q{i}', 0) for i in range(1, 5))
            away_q_total = sum(r.get(f'away_q{i}', 0) for i in range(1, 5))
            
            if home_q_total > home_score + 5 or home_q_total < home_score - 5:
                issues.append(f"Record {i}: Quarter total mismatch for home team")
        
        return {'rate': 1 - min(len(issues) / len(records), 1) if records else 0, 'issues': issues[:10]}


class VolleyballEvaluatorAgent(BaseEvaluatorAgent):
    """Evaluator for volleyball data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VolleyballEvaluatorAgent", "volleyball", config)
        self.required_fields = [
            'match_id', 'date', 'home_team', 'away_team', 'home_sets_won', 'away_sets_won',
            'home_total_points', 'away_total_points'
        ]
    
    def _calculate_metrics(self, records: List[Dict]) -> Dict:
        """Calculate volleyball-specific metrics."""
        completeness = self._calculate_completeness(records, self.required_fields)
        consistency = self._calculate_volleyball_consistency(records)
        
        # Set distribution
        set_counts = [r.get('home_sets_won', 0) + r.get('away_sets_won', 0) for r in records]
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'avg_sets_per_match': statistics.mean(set_counts) if set_counts else 0,
            'five_set_matches': sum(1 for s in set_counts if s == 5) / len(set_counts) if set_counts else 0
        }
    
    def _calculate_volleyball_consistency(self, records: List[Dict]) -> Dict:
        """Check volleyball data consistency."""
        issues = []
        
        for i, r in enumerate(records):
            home_sets = r.get('home_sets_won', 0)
            away_sets = r.get('away_sets_won', 0)
            
            # Check set scores
            if not (3 <= home_sets + away_sets <= 5):
                issues.append(f"Record {i}: Invalid total sets {home_sets + away_sets}")
            
            if home_sets == away_sets:
                issues.append(f"Record {i}: Tied sets (impossible)")
            
            if home_sets < 0 or away_sets < 0 or home_sets > 3 or away_sets > 3:
                issues.append(f"Record {i}: Invalid individual set count")
        
        return {'rate': 1 - min(len(issues) / len(records), 1) if records else 0, 'issues': issues[:10]}


class HandballEvaluatorAgent(BaseEvaluatorAgent):
    """Evaluator for handball data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("HandballEvaluatorAgent", "handball", config)
        self.required_fields = [
            'match_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score',
            'home_shots', 'away_shots', 'home_saves', 'away_saves'
        ]
    
    def _calculate_metrics(self, records: List[Dict]) -> Dict:
        """Calculate handball-specific metrics."""
        completeness = self._calculate_completeness(records, self.required_fields)
        consistency = self._calculate_handball_consistency(records)
        
        # Score distribution
        scores = [r.get('home_score', 0) + r.get('away_score', 0) for r in records]
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'avg_total_goals': statistics.mean(scores) if scores else 0,
            'high_scoring_matches': sum(1 for s in scores if s > 60) / len(scores) if scores else 0
        }
    
    def _calculate_handball_consistency(self, records: List[Dict]) -> Dict:
        """Check handball data consistency."""
        issues = []
        
        for i, r in enumerate(records):
            home = r.get('home_score', 0)
            away = r.get('away_score', 0)
            
            # Check for unrealistic scores
            if home < 0 or away < 0:
                issues.append(f"Record {i}: Negative score")
            if home > 50 or away > 50:
                issues.append(f"Record {i}: Unusually high score")
            if home == away:
                issues.append(f"Record {i}: Draw (rare in handball)")
            
            # Check shot statistics
            home_shots = r.get('home_shots', 0)
            if home_shots > 0 and home > home_shots:
                issues.append(f"Record {i}: More goals than shots")
        
        return {'rate': 1 - min(len(issues) / len(records), 1) if records else 0, 'issues': issues[:10]}


class TennisEvaluatorAgent(BaseEvaluatorAgent):
    """Evaluator for tennis data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TennisEvaluatorAgent", "tennis", config)
        self.required_fields = [
            'match_id', 'date', 'player1_name', 'player2_name', 
            'player1_sets_won', 'player2_sets_won', 'surface'
        ]
    
    def _calculate_metrics(self, records: List[Dict]) -> Dict:
        """Calculate tennis-specific metrics."""
        completeness = self._calculate_completeness(records, self.required_fields)
        consistency = self._calculate_tennis_consistency(records)
        
        # Surface distribution
        surfaces = {}
        for r in records:
            s = r.get('surface', 'Unknown')
            surfaces[s] = surfaces.get(s, 0) + 1
        
        # Upset rate (lower ranked player wins)
        upsets = 0
        total = 0
        for r in records:
            r1 = r.get('player1_rank', 999)
            r2 = r.get('player2_rank', 999)
            p1_won = r.get('player1_sets_won', 0) > r.get('player2_sets_won', 0)
            
            if r1 != 999 and r2 != 999:
                total += 1
                if (r1 > r2 and p1_won) or (r2 > r1 and not p1_won):
                    upsets += 1
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'surface_distribution': surfaces,
            'upset_rate': upsets / total if total > 0 else 0,
            'avg_match_duration': statistics.mean([r.get('match_duration_minutes', 120) for r in records]) if records else 0
        }
    
    def _calculate_tennis_consistency(self, records: List[Dict]) -> Dict:
        """Check tennis data consistency."""
        issues = []
        
        for i, r in enumerate(records):
            p1_sets = r.get('player1_sets_won', 0)
            p2_sets = r.get('player2_sets_won', 0)
            
            # Check set counts
            if not (2 <= p1_sets + p2_sets <= 5):
                issues.append(f"Record {i}: Invalid total sets")
            
            if p1_sets == p2_sets:
                issues.append(f"Record {i}: Tied sets")
            
            if p1_sets < 0 or p2_sets < 0 or p1_sets > 3 or p2_sets > 3:
                issues.append(f"Record {i}: Invalid individual set count")
            
            # Men's matches need 3 sets, women's need 2
            if p1_sets + p2_sets == 5:
                # Best of 5 (men's Grand Slam)
                if p1_sets != 3 and p2_sets != 3:
                    issues.append(f"Record {i}: Invalid 5-set match result")
            elif p1_sets + p2_sets == 3:
                # Best of 3
                if p1_sets != 2 and p2_sets != 2:
                    issues.append(f"Record {i}: Invalid 3-set match result")
        
        return {'rate': 1 - min(len(issues) / len(records), 1) if records else 0, 'issues': issues[:10]}
