"""
Sport-specific agents - specialized agents for each sport.
Each agent knows the best data sources and required data fields for its sport.
"""

from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class BasketballAgent(BaseAgent):
    """
    Basketball Agent - specialized for basketball data collection.
    
    Data requirements:
    - Game results (points, quarters, overtime)
    - Team statistics (FG%, 3P%, FT%, rebounds, assists, turnovers)
    - Player statistics
    - Home/away performance
    - Head-to-head history
    - Injuries
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("BasketballAgent", config)
        self.sport = "basketball"
        self.leagues = [
            'NBA', 'EuroLeague', 'EuroCup', 'ACB Spain', 'Legabasket Italy',
            'BBL Germany', 'Pro A France', 'Liga ABA', 'VTB United',
            'Polish Basketball League', 'NCAA'
        ]
        self.data_sources = [
            'flashscore.com', 'basketball.realgm.com', 'basketball-reference.com',
            'euroleague.net', 'espn.com/nba', 'nba.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute basketball-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for basketball."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        # Calculate games needed per league
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
                    'home_score', 'away_score', 'home_q1', 'home_q2', 'home_q3', 'home_q4',
                    'away_q1', 'away_q2', 'away_q3', 'away_q4', 'overtime',
                    'home_fg_made', 'home_fg_attempts', 'home_3p_made', 'home_3p_attempts',
                    'home_ft_made', 'home_ft_attempts', 'home_rebounds', 'home_assists',
                    'home_steals', 'home_blocks', 'home_turnovers', 'home_fouls',
                    'away_fg_made', 'away_fg_attempts', 'away_3p_made', 'away_3p_attempts',
                    'away_ft_made', 'away_ft_attempts', 'away_rebounds', 'away_assists',
                    'away_steals', 'away_blocks', 'away_turnovers', 'away_fouls',
                    'venue', 'attendance', 'referees'
                ],
                'optional_fields': [
                    'pace', 'offensive_rating', 'defensive_rating', 'true_shooting_pct',
                    'home_rest_days', 'away_rest_days', 'home_travel_distance', 'away_travel_distance'
                ]
            },
            'web_sources': [
                {'url': 'https://www.flashscore.com/basketball/', 'priority': 1},
                {'url': 'https://basketball.realgm.com/international/leagues', 'priority': 2},
                {'url': 'https://www.basketball-reference.com/', 'priority': 2},
                {'url': 'https://www.euroleaguebasketball.net/euroleague/', 'priority': 1},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.leagues)} leagues")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for basketball data."""
        queries = []
        base_query = f"basketball game results {date_range.get('start', '')} to {date_range.get('end', '')}"
        
        for league in self.leagues[:6]:  # Top 6 leagues
            queries.append(f"{league} basketball box score statistics results")
            queries.append(f"{league} historical game data player stats")
            queries.append(f"{league} games {date_range.get('start', '2020')} {date_range.get('end', '2024')}")
        
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate basketball data completeness."""
        data = task.get('data', [])
        required = ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']
        
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


class VolleyballAgent(BaseAgent):
    """
    Volleyball Agent - specialized for volleyball data collection.
    
    Data requirements:
    - Match results (sets, points per set)
    - Team statistics (attacks, blocks, serves, reception)
    - Player statistics
    - Set-by-set breakdown
    - Home/away performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VolleyballAgent", config)
        self.sport = "volleyball"
        self.leagues = [
            'Lega Pallavolo SuperLega Italy', 'Polish PlusLiga', 'Russian Super League',
            'Turkish Efeler Ligi', 'Brazilian Superliga', 'CEV Champions League',
            'CEV Cup', 'FIVB Club World Championship', 'French Ligue A',
            'German Bundesliga', 'NCAA Volleyball'
        ]
        self.data_sources = [
            'volleyball-madness.com', 'volleyball.world', 'legavolley.it',
            'plusliga.pl', 'flashscore.com/volleyball', 'bvbinfo.com'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute volleyball-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for volleyball."""
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
                    'match_id', 'date', 'league', 'season', 'home_team', 'away_team',
                    'home_sets_won', 'away_sets_won', 'set1_home', 'set1_away',
                    'set2_home', 'set2_away', 'set3_home', 'set3_away',
                    'set4_home', 'set4_away', 'set5_home', 'set5_away',
                    'home_total_points', 'away_total_points',
                    'home_attacks', 'home_blocks', 'home_aces', 'home_errors',
                    'away_attacks', 'away_blocks', 'away_aces', 'away_errors',
                    'home_reception_pct', 'away_reception_pct',
                    'venue', 'attendance', 'duration_minutes'
                ],
                'optional_fields': [
                    'home_attack_pct', 'away_attack_pct', 'home_block_pct', 'away_block_pct',
                    'mvp_player', 'referees', 'weather_conditions'
                ]
            },
            'web_sources': [
                {'url': 'https://www.flashscore.com/volleyball/', 'priority': 1},
                {'url': 'https://www.plusliga.pl/', 'priority': 1},
                {'url': 'https://www.legavolley.it/', 'priority': 1},
                {'url': 'https://en.volleyballworld.com/', 'priority': 2},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.leagues)} leagues")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for volleyball data."""
        queries = []
        for league in self.leagues[:6]:
            queries.append(f"{league} volleyball match results statistics")
            queries.append(f"{league} historical scores set results")
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate volleyball data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'home_team', 'away_team', 'home_sets_won', 'away_sets_won']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)


class HandballAgent(BaseAgent):
    """
    Handball Agent - specialized for handball data collection.
    
    Data requirements:
    - Match results (goals, half-time)
    - Team statistics (shots, saves, turnovers, penalties)
    - Player statistics
    - Goalkeeper performance
    - Home/away performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("HandballAgent", config)
        self.sport = "handball"
        self.leagues = [
            'EHF Champions League', 'EHF European League', 'Bundesliga Germany',
            'Liga ASOBAL Spain', 'LNH France', 'Polish Superliga',
            'Danish Handball League', 'Swedish Handball League', 'EHF Cup',
            'World Championship', 'European Championship', 'Olympic Games'
        ]
        self.data_sources = [
            'flashscore.com/handball', 'eurohandball.com', 'handball-base.com',
            'worldofhandball.com', 'handball.net'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute handball-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for handball."""
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
                    'match_id', 'date', 'league', 'season', 'home_team', 'away_team',
                    'home_score', 'away_score', 'home_ht_score', 'away_ht_score',
                    'home_shots', 'away_shots', 'home_saves', 'away_saves',
                    'home_turnovers', 'away_turnovers', 'home_penalties', 'away_penalties',
                    'home_suspensions', 'away_suspensions', 'home_7m_goals', 'away_7m_goals',
                    'home_7m_attempts', 'away_7m_attempts', 'home_fast_breaks', 'away_fast_breaks',
                    'home_gk_saves', 'away_gk_saves', 'home_gk_shots', 'away_gk_shots',
                    'venue', 'attendance', 'referees'
                ],
                'optional_fields': [
                    'mvp', 'top_scorer_home', 'top_scorer_away', 'fastest_goal',
                    'penalty_minutes', 'team_timeout_home', 'team_timeout_away'
                ]
            },
            'web_sources': [
                {'url': 'https://www.flashscore.com/handball/', 'priority': 1},
                {'url': 'https://www.eurohandball.com/', 'priority': 1},
                {'url': 'https://www.handball-base.com/', 'priority': 2},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.leagues)} leagues")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for handball data."""
        queries = []
        for league in self.leagues[:6]:
            queries.append(f"{league} handball results match statistics")
            queries.append(f"{league} historical game scores goals")
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate handball data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)


class TennisAgent(BaseAgent):
    """
    Tennis Agent - specialized for tennis data collection.
    
    Data requirements:
    - Match results (sets, games, tiebreaks)
    - Player statistics (aces, double faults, winners, unforced errors)
    - Serve statistics (1st/2nd serve %, break points)
    - Court surface, tournament level
    - Head-to-head history
    - Player rankings
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TennisAgent", config)
        self.sport = "tennis"
        self.tournaments = [
            'Grand Slam', 'ATP Masters 1000', 'ATP 500', 'ATP 250',
            'WTA 1000', 'WTA 500', 'WTA 250', 'Davis Cup', 'Billie Jean King Cup',
            'Olympic Games', 'Challenger Tour', 'ITF Tour'
        ]
        self.surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        self.data_sources = [
            'atptour.com', 'wtatennis.com', 'flashscore.com/tennis',
            'tennisabstract.com', 'ultimatetennisstatistics.com',
            'tennis24.com', 'live-tennis.eu', 'stevegtennis.com',
            'ranking-tennis.com', 'coretennis.net', 'tt.tennis-navigator.ru'
        ]
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute tennis-specific task."""
        action = task.get('action', 'plan_collection')
        
        if action == 'plan_collection':
            return await self._plan_collection(task)
        elif action == 'validate_data':
            return await self._validate_data(task)
        else:
            return TaskResult(success=False, error=f"Unknown action: {action}")
    
    async def _plan_collection(self, task: Dict[str, Any]) -> TaskResult:
        """Plan data collection strategy for tennis."""
        target = task.get('target_records', 10000)
        date_range = task.get('date_range', {})
        
        matches_per_tournament = target // len(self.tournaments)
        
        strategy = {
            'sport': self.sport,
            'target_records': target,
            'date_range': date_range,
            'tournaments': self.tournaments,
            'surfaces': self.surfaces,
            'distribution': {t: matches_per_tournament for t in self.tournaments},
            'data_sources': self.data_sources,
            'search_queries': self._generate_search_queries(date_range),
            'schema': {
                'required_fields': [
                    'match_id', 'date', 'tournament', 'tournament_level', 'surface',
                    'round', 'player1_name', 'player2_name', 'player1_rank', 'player2_rank',
                    'player1_seed', 'player2_seed', 'player1_sets_won', 'player2_sets_won',
                    'set1_score', 'set2_score', 'set3_score', 'set4_score', 'set5_score',
                    'player1_aces', 'player2_aces', 'player1_double_faults', 'player2_double_faults',
                    'player1_first_serve_pct', 'player2_first_serve_pct',
                    'player1_first_serve_won', 'player2_first_serve_won',
                    'player1_second_serve_won', 'player2_second_serve_won',
                    'player1_break_points_won', 'player2_break_points_won',
                    'player1_break_points_total', 'player2_break_points_total',
                    'player1_winners', 'player2_winners', 'player1_unforced_errors', 'player2_unforced_errors',
                    'player1_net_points_won', 'player2_net_points_won',
                    'player1_total_points_won', 'player2_total_points_won',
                    'match_duration_minutes', 'umpire', 'venue'
                ],
                'optional_fields': [
                    'player1_age', 'player2_age', 'player1_height', 'player2_height',
                    'player1_hand', 'player2_hand', 'head_to_head_before_match',
                    'player1_odds', 'player2_odds', 'temperature', 'weather'
                ]
            },
            'web_sources': [
                {'url': 'https://www.flashscore.com/tennis/', 'priority': 1},
                {'url': 'https://www.tennis24.com/', 'priority': 1},
                {'url': 'https://www.atptour.com/en/scores/archive', 'priority': 1},
                {'url': 'https://www.wtatennis.com/scores', 'priority': 1},
                {'url': 'http://www.tennisabstract.com/', 'priority': 2},
                {'url': 'https://live-tennis.eu/', 'priority': 2},
                {'url': 'http://www.stevegtennis.com/', 'priority': 3},
                {'url': 'https://www.ranking-tennis.com/', 'priority': 3},
                {'url': 'https://www.coretennis.net/', 'priority': 3},
            ]
        }
        
        logger.info(f"[{self.name}] Planned collection: {target} records across {len(self.tournaments)} tournaments")
        return TaskResult(success=True, data=strategy)
    
    def _generate_search_queries(self, date_range: Dict) -> List[str]:
        """Generate search queries for tennis data."""
        queries = []
        for tournament in self.tournaments[:6]:
            queries.append(f"{tournament} tennis match results statistics ATP WTA")
            queries.append(f"{tournament} historical scores serve stats")
        queries.append("tennis match statistics ATP tour complete data")
        queries.append("WTA tour tennis results match stats database")
        return queries
    
    async def _validate_data(self, task: Dict[str, Any]) -> TaskResult:
        """Validate tennis data completeness."""
        data = task.get('data', [])
        required = ['match_id', 'date', 'player1_name', 'player2_name', 'player1_sets_won', 'player2_sets_won']
        
        valid = sum(1 for r in data if all(f in r and r[f] is not None for f in required))
        return TaskResult(success=True, records_processed=valid)
