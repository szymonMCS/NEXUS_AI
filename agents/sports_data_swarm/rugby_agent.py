"""
Rugby Agent - specialized for rugby union/league data collection.

Data requirements:
- Match results (scores, tries, conversions)
- Player statistics (tries, assists, tackles, meters gained)
- Team statistics (possession, territory, scrums, lineouts)
- Discipline (cards, penalties)
- Tournament information
- Home/away splits
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class RugbyAgent(BaseAgent):
    """
    Rugby Agent - specialized for rugby union and league data collection.
    
    Competitions covered:
    - Rugby World Cup
    - Six Nations
    - Rugby Championship
    - Premiership Rugby
    - Top 14
    - Super Rugby
    - NRL (League)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("RugbyAgent", config)
        self.sport = "rugby"
        self.competitions = [
            'Rugby World Cup', 'Six Nations', 'Rugby Championship',
            'Premiership Rugby', 'Top 14', 'Super Rugby', 'URC',
            'NRL', 'Super League', 'Challenge Cup'
        ]
        self.data_sources = [
            'world.rugby', 'sixnations.rugby', 'espn.com/rugby',
            'flashscore.com/rugby', 'nrl.com', 'superrugby.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute rugby-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for rugby."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        matches_per_comp = target // len(self.competitions)
        
        strategy = {
            'sport': self.sport,
            'target_records': target,
            'date_range': date_range,
            'competitions': self.competitions,
            'distribution': {comp: matches_per_comp for comp in self.competitions},
            'data_sources': self.data_sources,
            'search_queries': self._generate_search_queries(date_range),
            'schema': {
                'required_fields': [
                    'match_id', 'date', 'competition', 'home_team', 'away_team',
                    'home_score', 'away_score', 'home_tries', 'away_tries',
                    'home_conversions', 'away_conversions',
                    'home_penalties', 'away_penalties',
                    'home_drop_goals', 'away_drop_goals',
                    'venue', 'attendance'
                ],
                'optional_fields': [
                    'home_possession', 'away_possession', 'home_territory', 'away_territory',
                    'home_scrums_won', 'home_scrums_lost', 'away_scrums_won', 'away_scrums_lost',
                    'home_lineouts_won', 'home_lineouts_lost', 'away_lineouts_won', 'away_lineouts_lost',
                    'home_tackles_made', 'home_tackles_missed', 'away_tackles_made', 'away_tackles_missed',
                    'home_turnovers', 'away_turnovers', 'home_kicks', 'away_kicks',
                    'home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards',
                    'handicap_line', 'total_points_line', 'moneyline_home', 'moneyline_away'
                ]
            },
            'web_sources': [
                {'url': 'https://www.world.rugby/', 'priority': 1},
                {'url': 'https://www.sixnations.rugby/', 'priority': 1},
                {'url': 'https://www.flashscore.com/rugby/', 'priority': 1},
                {'url': 'https://www.nrl.com/', 'priority': 1},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.competitions)} competitions")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for rugby data."""
        queries = [
            "rugby match results statistics historical data",
            "Six Nations results betting data",
            "NRL rugby league results statistics"
        ]
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate rugby data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)
