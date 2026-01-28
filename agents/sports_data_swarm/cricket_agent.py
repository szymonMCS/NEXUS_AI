"""
Cricket Agent - specialized for cricket (Test, ODI, T20) data collection.

Data requirements:
- Match results (runs, wickets, overs)
- Batting statistics (runs, balls, fours, sixes, strike rate)
- Bowling statistics (overs, maidens, runs, wickets, economy)
- Team statistics (run rate, extras, partnerships)
- Toss information
- Venue details
- Weather conditions
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class CricketAgent(BaseAgent):
    """
    Cricket Agent - specialized for cricket data collection.
    
    Formats covered:
    - Test Cricket
    - ODI (One Day International)
    - T20 International
    - T20 Leagues (IPL, BBL, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CricketAgent", config)
        self.sport = "cricket"
        self.formats = [
            'Test', 'ODI', 'T20I', 'IPL', 'BBL', 'PSL', 'CPL',
            'The Hundred', 'County Championship', 'Sheffield Shield'
        ]
        self.data_sources = [
            'espncricinfo.com', 'cricbuzz.com', 'flashscore.com/cricket',
            'icc-cricket.com', 'cricketworld.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute cricket-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for cricket."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        matches_per_format = target // len(self.formats)
        
        strategy = {
            'sport': self.sport,
            'target_records': target,
            'date_range': date_range,
            'formats': self.formats,
            'distribution': {fmt: matches_per_format for fmt in self.formats},
            'data_sources': self.data_sources,
            'search_queries': self._generate_search_queries(date_range),
            'schema': {
                'required_fields': [
                    'match_id', 'date', 'format', 'competition', 'venue',
                    'team1', 'team2', 'toss_winner', 'toss_decision',
                    'team1_runs', 'team1_wickets', 'team1_overs',
                    'team2_runs', 'team2_wickets', 'team2_overs',
                    'match_result', 'winner'
                ],
                'optional_fields': [
                    'team1_run_rate', 'team2_run_rate',
                    'team1_extras', 'team2_extras',
                    'team1_fours', 'team1_sixes', 'team2_fours', 'team2_sixes',
                    'highest_team_score', 'lowest_team_score',
                    'top_scorer_runs', 'top_wickets',
                    'first_innings_lead', 'follow_on_enforced',
                    'dl_method_applied', 'match_reduced_overs',
                    'betting_odds_t1', 'betting_odds_t2', 'match_handicap'
                ]
            },
            'web_sources': [
                {'url': 'https://www.espncricinfo.com/', 'priority': 1},
                {'url': 'https://www.cricbuzz.com/', 'priority': 1},
                {'url': 'https://www.flashscore.com/cricket/', 'priority': 1},
                {'url': 'https://www.icc-cricket.com/', 'priority': 2},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.formats)} formats")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for cricket data."""
        queries = [
            "cricket match results statistics historical data",
            "IPL T20 cricket betting data results",
            "Test cricket match results data"
        ]
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate cricket data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'team1', 'team2', 'team1_runs', 'team2_runs']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)
