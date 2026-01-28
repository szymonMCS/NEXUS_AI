"""
Baseball Agent - specialized for baseball (MLB) data collection.

Data requirements:
- Game results (runs, innings, hits, errors)
- Pitcher statistics (ERA, WHIP, strikeouts, walks)
- Batter statistics (avg, HR, RBI, OPS)
- Team statistics (runs scored, runs allowed)
- Advanced metrics (WAR, wOBA, FIP)
- Home/away splits
- Weather conditions (wind, temperature)
- Ballpark factors
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class BaseballAgent(BaseAgent):
    """
    Baseball Agent - specialized for MLB and baseball data collection.
    
    Data requirements:
    - Game results (runs, hits, errors)
    - Pitcher and batter statistics
    - Advanced sabermetrics (WAR, wOBA, FIP)
    - Ballpark factors
    - Weather conditions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("BaseballAgent", config)
        self.sport = "baseball"
        self.leagues = [
            'MLB', 'AAA Minor League', 'NPB Japan', 'KBO Korea',
            'CPBL Taiwan', 'Mexican League', 'Dominican Winter League',
            'Venezuelan Winter League', 'Cuban National Series'
        ]
        self.data_sources = [
            'baseball-reference.com', 'fangraphs.com', 'mlb.com',
            'baseballsavant.mlb.com', 'espn.com/mlb', 'flashscore.com/baseball',
            'statmuse.com/mlb'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute baseball-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for baseball."""
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
                    'home_runs', 'away_runs', 'home_hits', 'away_hits',
                    'home_errors', 'away_errors', 'innings_played',
                    'home_starter_pitcher', 'away_starter_pitcher',
                    'home_starter_era', 'away_starter_era',
                    'venue', 'attendance', 'weather_temp', 'weather_condition'
                ],
                'optional_fields': [
                    'home_team_avg', 'away_team_avg', 'home_team_ops', 'away_team_ops',
                    'home_starter_whip', 'away_starter_whip',
                    'home_starter_so', 'away_starter_so',
                    'home_starter_bb', 'away_starter_bb',
                    'wind_speed', 'wind_direction', 'ballpark_factor',
                    'home_moneyline', 'away_moneyline', 'runline', 'total_runs_line',
                    'home_batting_war', 'away_batting_war'
                ]
            },
            'web_sources': [
                {'url': 'https://www.baseball-reference.com/', 'priority': 1},
                {'url': 'https://www.fangraphs.com/', 'priority': 1},
                {'url': 'https://www.mlb.com/', 'priority': 2},
                {'url': 'https://baseballsavant.mlb.com/', 'priority': 2},
                {'url': 'https://www.flashscore.com/baseball/', 'priority': 1},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.leagues)} leagues")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for baseball data."""
        queries = []
        for league in ['MLB', 'NPB', 'KBO']:
            queries.append(f"{league} baseball game results box score historical data")
            queries.append(f"{league} baseball statistics pitcher batter data")
            queries.append(f"{league} baseball betting odds results")
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate baseball data completeness."""
        data = task.get('data', [])
        required = ['game_id', 'date', 'home_team', 'away_team', 'home_runs', 'away_runs']
        
        valid = 0
        issues = []
        
        for record in data:
            missing = [f for f in required if f not in record or record[f] is None]
            if not missing:
                valid += 1
            else:
                issues.append(f"Record missing: {missing}")
        
        return TaskResult(
            success=True,
            data={'valid': valid, 'total': len(data), 'issues': issues[:10]},
            records_processed=valid
        )
