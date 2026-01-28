"""
Hockey Agent - specialized for ice hockey (NHL) data collection.

Data requirements:
- Game results (goals, periods, shots)
- Goaltender statistics (saves, save %, GAA)
- Skater statistics (goals, assists, shots, hits, blocks)
- Power play / penalty kill stats
- Faceoff percentages
- Home/away splits
- Advanced metrics (Corsi, Fenwick, xG)
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class HockeyAgent(BaseAgent):
    """
    Hockey Agent - specialized for NHL and ice hockey data collection.
    
    Data requirements:
    - Game results (goals, shots, saves)
    - Goaltender statistics
    - Skater statistics
    - Advanced metrics (Corsi, Fenwick)
    - Special teams (PP, PK)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("HockeyAgent", config)
        self.sport = "hockey"
        self.leagues = [
            'NHL', 'AHL', 'KHL Russia', 'SHL Sweden', 'Liiga Finland',
            'Czech Extraliga', 'NLA Switzerland', 'DEL Germany',
            'ECHL', 'NCAA Hockey', 'PWHL', 'Olympics'
        ]
        self.data_sources = [
            'hockey-reference.com', 'nhl.com', 'eliteprospects.com',
            'flashscore.com/hockey', 'espn.com/nhl', 'naturalstattrick.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute hockey-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for hockey."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        games_per_league = target // len(self.leagues)
        
        strategy = {
            'sport': self.sport,
            'target_records': target,
            'date_range': date_range,
            'leagues': self.leagues,
            'distribution': {league: games_per_league for league in self.leagues},
            'data_sources': self.data_sources,
            'search_queries': self._generate_search_queries(date_range),
            'schema': {
                'required_fields': [
                    'game_id', 'date', 'league', 'season', 'home_team', 'away_team',
                    'home_goals', 'away_goals', 'home_shots', 'away_shots',
                    'home_saves', 'away_saves', 'home_pp_goals', 'away_pp_goals',
                    'home_pp_opps', 'away_pp_opps', 'home_pim', 'away_pim',
                    'home_faceoff_wins', 'away_faceoff_wins',
                    'overtime', 'shootout', 'venue'
                ],
                'optional_fields': [
                    'home_corsi', 'away_corsi', 'home_fenwick', 'away_fenwick',
                    'home_xg', 'away_xg', 'home_hits', 'away_hits',
                    'home_blocks', 'away_blocks', 'home_takeaways', 'away_takeaways',
                    'home_giveaways', 'away_giveaways',
                    'home_goalie_saves', 'home_goalie_shots', 'away_goalie_saves', 'away_goalie_shots',
                    'puck_line', 'total_goals_line', 'moneyline_home', 'moneyline_away'
                ]
            },
            'web_sources': [
                {'url': 'https://www.hockey-reference.com/', 'priority': 1},
                {'url': 'https://www.nhl.com/', 'priority': 1},
                {'url': 'https://www.flashscore.com/hockey/', 'priority': 1},
                {'url': 'https://www.naturalstattrick.com/', 'priority': 2},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.leagues)} leagues")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for hockey data."""
        queries = []
        for league in ['NHL', 'KHL', 'SHL']:
            queries.append(f"{league} hockey game results statistics historical data")
            queries.append(f"{league} hockey betting odds results")
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate hockey data completeness."""
        data = task.get('data', [])
        required = ['game_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)
