"""
Table Tennis Agent - specialized for table tennis data collection.

Data requirements:
- Match results (games, points)
- Player rankings
- Tournament information (ITTF, WTT)
- Head-to-head records
- Serve statistics
- Rally length
- Playing style (attack, defense)
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class TableTennisAgent(BaseAgent):
    """
    Table Tennis Agent - specialized for table tennis/ping pong data collection.
    
    Tours covered:
    - ITTF World Tour
    - WTT (World Table Tennis)
    - Olympic Games
    - Pro Tour
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TableTennisAgent", config)
        self.sport = "table_tennis"
        self.tours = [
            'ITTF World Tour', 'WTT', 'Olympics', 'World Championships',
            'World Cup', 'ITTF Challenge Series', 'European Championships',
            'Asian Championships'
        ]
        self.data_sources = [
            'ittf.com', 'worldtabletennis.com', 'flashscore.com/table-tennis',
            'tabletennis.guide', 'ratingcentral.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute table tennis-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for table tennis."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        matches_per_tour = target // len(self.tours)
        
        strategy = {
            'sport': self.sport,
            'target_records': target,
            'date_range': date_range,
            'tours': self.tours,
            'distribution': {tour: matches_per_tour for tour in self.tours},
            'data_sources': self.data_sources,
            'search_queries': self._generate_search_queries(date_range),
            'schema': {
                'required_fields': [
                    'match_id', 'date', 'tournament', 'tour', 'event',
                    'player1_name', 'player2_name', 'player1_rank', 'player2_rank',
                    'player1_nationality', 'player2_nationality',
                    'player1_games_won', 'player2_games_won',
                    'match_result'  # e.g., "4-3" for 4 games to 3
                ],
                'optional_fields': [
                    'player1_style', 'player2_style',  # attack, defense, all-round
                    'player1_grip', 'player2_grip',  # shakehand, penhold
                    'player1_age', 'player2_age',
                    'total_points_played', 'average_rally_length',
                    'player1_serve_points_won', 'player2_serve_points_won',
                    'player1_receive_points_won', 'player2_receive_points_won',
                    'deciding_game_played', 'player1_handicap', 'player2_handicap',
                    'betting_odds_p1', 'betting_odds_p2'
                ]
            },
            'web_sources': [
                {'url': 'https://www.ittf.com/', 'priority': 1},
                {'url': 'https://worldtabletennis.com/', 'priority': 1},
                {'url': 'https://www.flashscore.com/table-tennis/', 'priority': 1},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.tours)} tours")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for table tennis data."""
        queries = [
            "table tennis match results ITTF statistics",
            "WTT table tennis tournament results data",
            "table tennis betting odds results"
        ]
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate table tennis data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'player1_name', 'player2_name', 'player1_games_won', 'player2_games_won']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)
