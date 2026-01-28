"""
Football/Soccer Agent - specialized for football data collection.

Data requirements:
- Match results (goals, half-time, possession)
- Team statistics (shots, shots on target, corners, fouls, cards)
- Player statistics (goals, assists, passes, tackles)
- xG (expected goals) data
- Home/away performance
- League standings context
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class FootballAgent(BaseAgent):
    """
    Football/Soccer Agent - specialized for football data collection.
    
    Data requirements:
    - Match results (goals, half-time, possession)
    - Team statistics (shots, shots on target, corners, fouls, cards)
    - xG (expected goals) data
    - Home/away performance
    - League standings context
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("FootballAgent", config)
        self.sport = "football"
        self.leagues = [
            'Premier League England', 'La Liga Spain', 'Serie A Italy', 'Bundesliga Germany',
            'Ligue 1 France', 'Champions League', 'Europa League', 'Europa Conference League',
            'Eredivisie Netherlands', 'Primeira Liga Portugal', 'Super Lig Turkey',
            'Championship England', 'MLS USA', 'Brasileirao Brazil', 'Primera Division Argentina',
            'J1 League Japan', 'K League 1 South Korea', 'A-League Australia',
            'World Cup', 'Euro Championship', 'Copa America', 'AFCON', 'Nations League'
        ]
        self.data_sources = [
            'flashscore.com', 'fbref.com', 'understat.com', 'whoscored.com',
            'transfermarkt.com', 'soccerway.com', 'football-data.co.uk',
            'api-football.com', 'footystats.org'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute football-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for football."""
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
                    'match_id', 'date', 'league', 'season', 'matchday', 'home_team', 'away_team',
                    'home_goals', 'away_goals', 'home_ht_goals', 'away_ht_goals',
                    'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
                    'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
                    'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards',
                    'home_possession', 'away_possession', 'home_passes', 'away_passes',
                    'home_pass_accuracy', 'away_pass_accuracy', 'home_tackles', 'away_tackles',
                    'venue', 'attendance', 'referee'
                ],
                'optional_fields': [
                    'home_xg', 'away_xg', 'home_ppda', 'away_ppda',  # Expected goals, passes per defensive action
                    'home_deep_passes', 'away_deep_passes',  # Passes into dangerous areas
                    'home_points_before', 'away_points_before',  # Table position context
                    'home_form_last5', 'away_form_last5',  # Recent form
                    'odds_home', 'odds_draw', 'odds_away',  # Betting odds
                    'temperature', 'weather'
                ]
            },
            'web_sources': [
                {'url': 'https://www.flashscore.com/football/', 'priority': 1},
                {'url': 'https://fbref.com/en/comps/', 'priority': 1},
                {'url': 'https://understat.com/', 'priority': 1},  # xG data
                {'url': 'https://www.whoscored.com/', 'priority': 2},
                {'url': 'https://www.football-data.co.uk/data.php', 'priority': 2},
                {'url': 'https://www.transfermarkt.com/', 'priority': 3},
                {'url': 'https://apiv3.apifootball.com/', 'priority': 2},  # API source
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.leagues)} leagues")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for football data."""
        queries = []
        base_query = f"football match results statistics {date_range.get('start', '')} to {date_range.get('end', '')}"
        
        for league in self.leagues[:8]:  # Top 8 leagues
            queries.append(f"{league} football results xg expected goals")
            queries.append(f"{league} historical match data statistics")
            queries.append(f"{league} shots corners possession data")
            queries.append(f"{league} football database csv download")
        
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate football data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        
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
