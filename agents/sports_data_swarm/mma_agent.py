"""
MMA Agent - specialized for MMA/UFC data collection.

Data requirements:
- Fight results (method, round, time)
- Fighter statistics (strikes, takedowns, submissions)
- Physical attributes (height, reach, weight)
- Fight history and records
- Betting odds
- Event information
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class MMAAgent(BaseAgent):
    """
    MMA Agent - specialized for UFC/MMA data collection.
    
    Data requirements:
    - Fight results and methods
    - Fighter statistics
    - Physical attributes
    - Fight history
    - Betting odds
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("MMAAgent", config)
        self.sport = "mma"
        self.organizations = [
            'UFC', 'Bellator', 'ONE Championship', 'PFL', 'KSW',
            'Rizin', 'Invicta FC', 'Cage Warriors', 'ACB', 'M-1 Global'
        ]
        self.data_sources = [
            'ufc.com', 'sherdog.com', 'tapology.com', 'espn.com/mma',
            'flashscore.com/mma', 'mmajunkie.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute MMA-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for MMA."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        fights_per_org = target // len(self.organizations)
        
        strategy = {
            'sport': self.sport,
            'target_records': target,
            'date_range': date_range,
            'organizations': self.organizations,
            'distribution': {org: fights_per_org for org in self.organizations},
            'data_sources': self.data_sources,
            'search_queries': self._generate_search_queries(date_range),
            'schema': {
                'required_fields': [
                    'fight_id', 'date', 'event', 'organization', 'weight_class',
                    'fighter1_name', 'fighter2_name', 'fighter1_result', 'fighter2_result',
                    'method', 'round', 'time', 'referee',
                    'fighter1_strikes_landed', 'fighter1_strikes_attempted',
                    'fighter2_strikes_landed', 'fighter2_strikes_attempted',
                    'fighter1_takedowns_landed', 'fighter1_takedowns_attempted',
                    'fighter2_takedowns_landed', 'fighter2_takedowns_attempted'
                ],
                'optional_fields': [
                    'fighter1_height', 'fighter1_reach', 'fighter1_weight',
                    'fighter2_height', 'fighter2_reach', 'fighter2_weight',
                    'fighter1_age', 'fighter2_age', 'fighter1_stance', 'fighter2_stance',
                    'fighter1_record', 'fighter2_record', 'title_fight',
                    'fighter1_sub_attempts', 'fighter2_sub_attempts',
                    'fighter1_ctrl_time', 'fighter2_ctrl_time',
                    'betting_odds_f1', 'betting_odds_f2'
                ]
            },
            'web_sources': [
                {'url': 'https://www.ufc.com/', 'priority': 1},
                {'url': 'https://www.sherdog.com/', 'priority': 1},
                {'url': 'https://www.tapology.com/', 'priority': 1},
                {'url': 'https://www.flashscore.com/mma/', 'priority': 2},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.organizations)} organizations")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for MMA data."""
        queries = [
            "UFC fight results statistics historical data",
            "MMA fight outcomes betting odds results",
            "UFC fighter statistics physical attributes data"
        ]
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate MMA data completeness."""
        data = task.get('data', [])
        required = ['fight_id', 'date', 'fighter1_name', 'fighter2_name', 'method', 'round']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)
