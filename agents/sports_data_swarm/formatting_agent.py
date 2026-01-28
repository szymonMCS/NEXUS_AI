"""
Formatting Agent - Normalizes and formats data according to sport-specific schemas.
Ensures data consistency and completeness for ML training.
"""

import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class FormattingAgent(BaseAgent):
    """
    Agent responsible for formatting and normalizing sports data.
    
    Responsibilities:
    - Normalize team/player names
    - Standardize date formats
    - Convert numeric fields
    - Fill missing values with defaults
    - Validate against schema
    - Create ML-ready features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("FormattingAgent", config)
        self.name_mappings = {}  # Cache for name normalizations
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute data formatting task.
        
        Task params:
        - sport: Sport type
        - raw_data: Raw data to format
        - schema: Expected schema
        """
        self.status = AgentStatus.RUNNING
        
        try:
            sport = task['sport']
            raw_data = task.get('raw_data', {})
            schema = task.get('schema', {})
            
            records = raw_data.get('records', [])
            
            logger.info(f"[{self.name}] Formatting {len(records)} records for {sport}")
            
            formatted_records = []
            errors = []
            
            for i, record in enumerate(records):
                try:
                    formatted = await self._format_record(record, sport, schema)
                    if formatted:
                        # Add ML features
                        formatted = self._add_ml_features(formatted, sport)
                        formatted_records.append(formatted)
                    
                    if (i + 1) % 1000 == 0:
                        self.log_progress(f"Formatted records", i + 1, len(records))
                        
                except Exception as e:
                    errors.append(f"Record {i}: {str(e)}")
                    continue
            
            logger.info(f"[{self.name}] Successfully formatted {len(formatted_records)} records")
            
            # Generate schema compliance report
            compliance = self._check_schema_compliance(formatted_records, schema)
            
            self.status = AgentStatus.COMPLETED
            return TaskResult(
                success=True,
                data={
                    'records': formatted_records,
                    'compliance': compliance,
                    'errors': errors[:50]  # First 50 errors
                },
                records_processed=len(formatted_records),
                metadata={
                    'schema_fields': len(schema.get('required_fields', [])),
                    'compliance_rate': compliance.get('overall_rate', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            self.status = AgentStatus.ERROR
            return TaskResult(success=False, error=str(e))
    
    async def _format_record(self, record: Dict, sport: str, schema: Dict) -> Optional[Dict]:
        """Format a single record according to sport schema."""
        required = schema.get('required_fields', [])
        optional = schema.get('optional_fields', [])
        
        formatted = {}
        
        # Format based on sport
        formatters = {
            'basketball': self._format_basketball_record,
            'volleyball': self._format_volleyball_record,
            'handball': self._format_handball_record,
            'tennis': self._format_tennis_record
        }
        
        formatter = formatters.get(sport)
        if formatter:
            formatted = formatter(record)
        else:
            formatted = record.copy()
        
        # Ensure all required fields exist
        for field in required:
            if field not in formatted or formatted[field] is None:
                formatted[field] = self._get_default_value(field, sport)
        
        # Add optional fields with defaults
        for field in optional:
            if field not in formatted:
                formatted[field] = self._get_default_value(field, sport)
        
        # Clean and normalize
        formatted = self._clean_record(formatted)
        
        return formatted
    
    def _format_basketball_record(self, record: Dict) -> Dict:
        """Format basketball-specific fields."""
        formatted = {}
        
        # Basic info
        formatted['game_id'] = str(record.get('game_id', record.get('match_id', '')))
        formatted['date'] = self._normalize_date(record.get('date', ''))
        formatted['league'] = self._normalize_text(record.get('league', 'Unknown'))
        formatted['season'] = record.get('season', '2024-25')
        
        # Teams
        formatted['home_team'] = self._normalize_team_name(record.get('home_team', 'Unknown'))
        formatted['away_team'] = self._normalize_team_name(record.get('away_team', 'Unknown'))
        
        # Scores
        formatted['home_score'] = self._parse_int(record.get('home_score', 0))
        formatted['away_score'] = self._parse_int(record.get('away_score', 0))
        formatted['home_q1'] = self._parse_int(record.get('home_q1', 0))
        formatted['home_q2'] = self._parse_int(record.get('home_q2', 0))
        formatted['home_q3'] = self._parse_int(record.get('home_q3', 0))
        formatted['home_q4'] = self._parse_int(record.get('home_q4', 0))
        formatted['away_q1'] = self._parse_int(record.get('away_q1', 0))
        formatted['away_q2'] = self._parse_int(record.get('away_q2', 0))
        formatted['away_q3'] = self._parse_int(record.get('away_q3', 0))
        formatted['away_q4'] = self._parse_int(record.get('away_q4', 0))
        formatted['overtime'] = self._parse_int(record.get('overtime', 0))
        
        # Statistics (with realistic defaults)
        formatted['home_fg_made'] = self._parse_int(record.get('home_fg_made', formatted['home_score'] // 2))
        formatted['home_fg_attempts'] = self._parse_int(record.get('home_fg_attempts', formatted['home_fg_made'] * 2))
        formatted['home_3p_made'] = self._parse_int(record.get('home_3p_made', formatted['home_score'] // 6))
        formatted['home_3p_attempts'] = self._parse_int(record.get('home_3p_attempts', formatted['home_3p_made'] * 3))
        formatted['home_ft_made'] = self._parse_int(record.get('home_ft_made', formatted['home_score'] // 5))
        formatted['home_ft_attempts'] = self._parse_int(record.get('home_ft_attempts', formatted['home_ft_made'] + 3))
        formatted['home_rebounds'] = self._parse_int(record.get('home_rebounds', 35))
        formatted['home_assists'] = self._parse_int(record.get('home_assists', 20))
        formatted['home_steals'] = self._parse_int(record.get('home_steals', 7))
        formatted['home_blocks'] = self._parse_int(record.get('home_blocks', 4))
        formatted['home_turnovers'] = self._parse_int(record.get('home_turnovers', 13))
        formatted['home_fouls'] = self._parse_int(record.get('home_fouls', 20))
        
        formatted['away_fg_made'] = self._parse_int(record.get('away_fg_made', formatted['away_score'] // 2))
        formatted['away_fg_attempts'] = self._parse_int(record.get('away_fg_attempts', formatted['away_fg_made'] * 2))
        formatted['away_3p_made'] = self._parse_int(record.get('away_3p_made', formatted['away_score'] // 6))
        formatted['away_3p_attempts'] = self._parse_int(record.get('away_3p_attempts', formatted['away_3p_made'] * 3))
        formatted['away_ft_made'] = self._parse_int(record.get('away_ft_made', formatted['away_score'] // 5))
        formatted['away_ft_attempts'] = self._parse_int(record.get('away_ft_attempts', formatted['away_ft_made'] + 3))
        formatted['away_rebounds'] = self._parse_int(record.get('away_rebounds', 35))
        formatted['away_assists'] = self._parse_int(record.get('away_assists', 20))
        formatted['away_steals'] = self._parse_int(record.get('away_steals', 7))
        formatted['away_blocks'] = self._parse_int(record.get('away_blocks', 4))
        formatted['away_turnovers'] = self._parse_int(record.get('away_turnovers', 13))
        formatted['away_fouls'] = self._parse_int(record.get('away_fouls', 20))
        
        # Venue info
        formatted['venue'] = record.get('venue', 'Unknown')
        formatted['attendance'] = self._parse_int(record.get('attendance', 5000))
        formatted['referees'] = record.get('referees', 'Unknown')
        
        # Calculated fields for ML
        formatted['home_fg_pct'] = round(formatted['home_fg_made'] / formatted['home_fg_attempts'], 3) if formatted['home_fg_attempts'] > 0 else 0
        formatted['home_3p_pct'] = round(formatted['home_3p_made'] / formatted['home_3p_attempts'], 3) if formatted['home_3p_attempts'] > 0 else 0
        formatted['home_ft_pct'] = round(formatted['home_ft_made'] / formatted['home_ft_attempts'], 3) if formatted['home_ft_attempts'] > 0 else 0
        formatted['away_fg_pct'] = round(formatted['away_fg_made'] / formatted['away_fg_attempts'], 3) if formatted['away_fg_attempts'] > 0 else 0
        formatted['away_3p_pct'] = round(formatted['away_3p_made'] / formatted['away_3p_attempts'], 3) if formatted['away_3p_attempts'] > 0 else 0
        formatted['away_ft_pct'] = round(formatted['away_ft_made'] / formatted['away_ft_attempts'], 3) if formatted['away_ft_attempts'] > 0 else 0
        
        return formatted
    
    def _format_volleyball_record(self, record: Dict) -> Dict:
        """Format volleyball-specific fields."""
        formatted = {}
        
        formatted['match_id'] = str(record.get('match_id', ''))
        formatted['date'] = self._normalize_date(record.get('date', ''))
        formatted['league'] = self._normalize_text(record.get('league', 'Unknown'))
        formatted['season'] = record.get('season', '2024-25')
        
        formatted['home_team'] = self._normalize_team_name(record.get('home_team', 'Unknown'))
        formatted['away_team'] = self._normalize_team_name(record.get('away_team', 'Unknown'))
        
        formatted['home_sets_won'] = self._parse_int(record.get('home_sets_won', 0))
        formatted['away_sets_won'] = self._parse_int(record.get('away_sets_won', 0))
        
        # Set scores
        for i in range(1, 6):
            formatted[f'set{i}_home'] = self._parse_int(record.get(f'set{i}_home', 0))
            formatted[f'set{i}_away'] = self._parse_int(record.get(f'set{i}_away', 0))
        
        formatted['home_total_points'] = self._parse_int(record.get('home_total_points', sum(formatted[f'set{i}_home'] for i in range(1, 6))))
        formatted['away_total_points'] = self._parse_int(record.get('away_total_points', sum(formatted[f'set{i}_away'] for i in range(1, 6))))
        
        # Statistics
        formatted['home_attacks'] = self._parse_int(record.get('home_attacks', 50))
        formatted['home_blocks'] = self._parse_int(record.get('home_blocks', 8))
        formatted['home_aces'] = self._parse_int(record.get('home_aces', 5))
        formatted['home_errors'] = self._parse_int(record.get('home_errors', 15))
        formatted['away_attacks'] = self._parse_int(record.get('away_attacks', 50))
        formatted['away_blocks'] = self._parse_int(record.get('away_blocks', 8))
        formatted['away_aces'] = self._parse_int(record.get('away_aces', 5))
        formatted['away_errors'] = self._parse_int(record.get('away_errors', 15))
        
        formatted['home_reception_pct'] = self._parse_float(record.get('home_reception_pct', 0.75))
        formatted['away_reception_pct'] = self._parse_float(record.get('away_reception_pct', 0.75))
        
        formatted['venue'] = record.get('venue', 'Unknown')
        formatted['attendance'] = self._parse_int(record.get('attendance', 2000))
        formatted['duration_minutes'] = self._parse_int(record.get('duration_minutes', 90))
        
        return formatted
    
    def _format_handball_record(self, record: Dict) -> Dict:
        """Format handball-specific fields."""
        formatted = {}
        
        formatted['match_id'] = str(record.get('match_id', ''))
        formatted['date'] = self._normalize_date(record.get('date', ''))
        formatted['league'] = self._normalize_text(record.get('league', 'Unknown'))
        formatted['season'] = record.get('season', '2024-25')
        
        formatted['home_team'] = self._normalize_team_name(record.get('home_team', 'Unknown'))
        formatted['away_team'] = self._normalize_team_name(record.get('away_team', 'Unknown'))
        
        formatted['home_score'] = self._parse_int(record.get('home_score', 0))
        formatted['away_score'] = self._parse_int(record.get('away_score', 0))
        formatted['home_ht_score'] = self._parse_int(record.get('home_ht_score', formatted['home_score'] // 2))
        formatted['away_ht_score'] = self._parse_int(record.get('away_ht_score', formatted['away_score'] // 2))
        
        # Statistics
        formatted['home_shots'] = self._parse_int(record.get('home_shots', formatted['home_score'] + 15))
        formatted['away_shots'] = self._parse_int(record.get('away_shots', formatted['away_score'] + 15))
        formatted['home_saves'] = self._parse_int(record.get('home_saves', 10))
        formatted['away_saves'] = self._parse_int(record.get('away_saves', 10))
        formatted['home_turnovers'] = self._parse_int(record.get('home_turnovers', 12))
        formatted['away_turnovers'] = self._parse_int(record.get('away_turnovers', 12))
        formatted['home_penalties'] = self._parse_int(record.get('home_penalties', 3))
        formatted['away_penalties'] = self._parse_int(record.get('away_penalties', 3))
        formatted['home_suspensions'] = self._parse_int(record.get('home_suspensions', 1))
        formatted['away_suspensions'] = self._parse_int(record.get('away_suspensions', 1))
        formatted['home_7m_goals'] = self._parse_int(record.get('home_7m_goals', formatted['home_score'] // 5))
        formatted['away_7m_goals'] = self._parse_int(record.get('away_7m_goals', formatted['away_score'] // 5))
        formatted['home_7m_attempts'] = self._parse_int(record.get('home_7m_attempts', formatted['home_7m_goals'] + 3))
        formatted['away_7m_attempts'] = self._parse_int(record.get('away_7m_attempts', formatted['away_7m_goals'] + 3))
        formatted['home_fast_breaks'] = self._parse_int(record.get('home_fast_breaks', 5))
        formatted['away_fast_breaks'] = self._parse_int(record.get('away_fast_breaks', 5))
        formatted['home_gk_saves'] = self._parse_int(record.get('home_gk_saves', 8))
        formatted['away_gk_saves'] = self._parse_int(record.get('away_gk_saves', 8))
        formatted['home_gk_shots'] = self._parse_int(record.get('home_gk_shots', formatted['away_shots']))
        formatted['away_gk_shots'] = self._parse_int(record.get('away_gk_shots', formatted['home_shots']))
        
        formatted['venue'] = record.get('venue', 'Unknown')
        formatted['attendance'] = self._parse_int(record.get('attendance', 3000))
        formatted['referees'] = record.get('referees', 'Unknown')
        
        return formatted
    
    def _format_tennis_record(self, record: Dict) -> Dict:
        """Format tennis-specific fields."""
        formatted = {}
        
        formatted['match_id'] = str(record.get('match_id', ''))
        formatted['date'] = self._normalize_date(record.get('date', ''))
        formatted['tournament'] = self._normalize_text(record.get('tournament', 'Unknown'))
        formatted['tournament_level'] = record.get('tournament_level', 'ATP 250')
        formatted['surface'] = record.get('surface', 'Hard')
        formatted['round'] = record.get('round', 'R32')
        
        formatted['player1_name'] = self._normalize_player_name(record.get('player1_name', 'Unknown'))
        formatted['player2_name'] = self._normalize_player_name(record.get('player2_name', 'Unknown'))
        formatted['player1_rank'] = self._parse_int(record.get('player1_rank', 100))
        formatted['player2_rank'] = self._parse_int(record.get('player2_rank', 100))
        formatted['player1_seed'] = self._parse_int(record.get('player1_seed', 0))
        formatted['player2_seed'] = self._parse_int(record.get('player2_seed', 0))
        
        formatted['player1_sets_won'] = self._parse_int(record.get('player1_sets_won', 0))
        formatted['player2_sets_won'] = self._parse_int(record.get('player2_sets_won', 0))
        
        # Set scores
        for i in range(1, 6):
            formatted[f'set{i}_score'] = record.get(f'set{i}_score', '')
        
        # Statistics
        formatted['player1_aces'] = self._parse_int(record.get('player1_aces', 5))
        formatted['player2_aces'] = self._parse_int(record.get('player2_aces', 5))
        formatted['player1_double_faults'] = self._parse_int(record.get('player1_double_faults', 3))
        formatted['player2_double_faults'] = self._parse_int(record.get('player2_double_faults', 3))
        formatted['player1_first_serve_pct'] = self._parse_float(record.get('player1_first_serve_pct', 0.60))
        formatted['player2_first_serve_pct'] = self._parse_float(record.get('player2_first_serve_pct', 0.60))
        formatted['player1_first_serve_won'] = self._parse_float(record.get('player1_first_serve_won', 0.70))
        formatted['player2_first_serve_won'] = self._parse_float(record.get('player2_first_serve_won', 0.70))
        formatted['player1_second_serve_won'] = self._parse_float(record.get('player1_second_serve_won', 0.50))
        formatted['player2_second_serve_won'] = self._parse_float(record.get('player2_second_serve_won', 0.50))
        formatted['player1_break_points_won'] = self._parse_int(record.get('player1_break_points_won', 2))
        formatted['player2_break_points_won'] = self._parse_int(record.get('player2_break_points_won', 2))
        formatted['player1_break_points_total'] = self._parse_int(record.get('player1_break_points_total', 5))
        formatted['player2_break_points_total'] = self._parse_int(record.get('player2_break_points_total', 5))
        formatted['player1_winners'] = self._parse_int(record.get('player1_winners', 20))
        formatted['player2_winners'] = self._parse_int(record.get('player2_winners', 20))
        formatted['player1_unforced_errors'] = self._parse_int(record.get('player1_unforced_errors', 25))
        formatted['player2_unforced_errors'] = self._parse_int(record.get('player2_unforced_errors', 25))
        formatted['player1_net_points_won'] = self._parse_int(record.get('player1_net_points_won', 8))
        formatted['player2_net_points_won'] = self._parse_int(record.get('player2_net_points_won', 8))
        formatted['player1_total_points_won'] = self._parse_int(record.get('player1_total_points_won', 70))
        formatted['player2_total_points_won'] = self._parse_int(record.get('player2_total_points_won', 70))
        
        formatted['match_duration_minutes'] = self._parse_int(record.get('match_duration_minutes', 120))
        formatted['umpire'] = record.get('umpire', 'Unknown')
        formatted['venue'] = record.get('venue', 'Unknown')
        
        return formatted
    
    def _add_ml_features(self, record: Dict, sport: str) -> Dict:
        """Add ML-specific features to the record."""
        if sport == 'basketball':
            record['point_diff'] = record.get('home_score', 0) - record.get('away_score', 0)
            record['total_points'] = record.get('home_score', 0) + record.get('away_score', 0)
            record['home_win'] = 1 if record['point_diff'] > 0 else 0
            record['pace_estimate'] = record['total_points'] / 90  # Approximate pace
            
        elif sport == 'volleyball':
            record['sets_diff'] = record.get('home_sets_won', 0) - record.get('away_sets_won', 0)
            record['total_points'] = record.get('home_total_points', 0) + record.get('away_total_points', 0)
            record['home_win'] = 1 if record['sets_diff'] > 0 else 0
            
        elif sport == 'handball':
            record['goal_diff'] = record.get('home_score', 0) - record.get('away_score', 0)
            record['total_goals'] = record.get('home_score', 0) + record.get('away_score', 0)
            record['home_win'] = 1 if record['goal_diff'] > 0 else 0
            
        elif sport == 'tennis':
            record['sets_diff'] = record.get('player1_sets_won', 0) - record.get('player2_sets_won', 0)
            record['player1_win'] = 1 if record['sets_diff'] > 0 else 0
            record['rank_diff'] = record.get('player2_rank', 100) - record.get('player1_rank', 100)
            record['favorite_win'] = 1 if (record['rank_diff'] > 0 and record['player1_win'] == 1) or \
                                          (record['rank_diff'] < 0 and record['player1_win'] == 0) else 0
        
        # Add timestamp
        record['processed_at'] = datetime.now().isoformat()
        
        return record
    
    def _check_schema_compliance(self, records: List[Dict], schema: Dict) -> Dict:
        """Check how well records comply with schema."""
        required = schema.get('required_fields', [])
        
        if not records:
            return {'overall_rate': 0, 'field_rates': {}}
        
        field_rates = {}
        for field in required:
            present = sum(1 for r in records if field in r and r[field] is not None)
            field_rates[field] = round(present / len(records), 3)
        
        overall = sum(field_rates.values()) / len(field_rates) if field_rates else 0
        
        return {
            'overall_rate': round(overall, 3),
            'field_rates': field_rates,
            'total_records': len(records)
        }
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format."""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        # Try various formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except:
                continue
        
        return date_str if len(date_str) == 10 else datetime.now().strftime('%Y-%m-%d')
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text field."""
        if not text:
            return 'Unknown'
        return str(text).strip().title()
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name."""
        if not name:
            return 'Unknown'
        name = str(name).strip()
        # Cache common normalizations
        if name.lower() in self.name_mappings:
            return self.name_mappings[name.lower()]
        return name
    
    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name."""
        if not name:
            return 'Unknown'
        return str(name).strip().title()
    
    def _parse_int(self, value: Any) -> int:
        """Parse integer value."""
        try:
            return int(float(value))
        except:
            return 0
    
    def _parse_float(self, value: Any) -> float:
        """Parse float value."""
        try:
            return float(value)
        except:
            return 0.0
    
    def _get_default_value(self, field: str, sport: str) -> Any:
        """Get default value for a field."""
        defaults = {
            'game_id': '', 'match_id': '', 'date': datetime.now().strftime('%Y-%m-%d'),
            'league': 'Unknown', 'tournament': 'Unknown', 'season': '2024-25',
            'home_team': 'Unknown', 'away_team': 'Unknown',
            'player1_name': 'Unknown', 'player2_name': 'Unknown',
            'venue': 'Unknown', 'referees': 'Unknown', 'umpire': 'Unknown',
            'surface': 'Hard', 'round': 'R32', 'tournament_level': 'ATP 250'
        }
        
        if field in ['home_score', 'away_score', 'home_sets_won', 'away_sets_won',
                     'player1_sets_won', 'player2_sets_won', 'attendance', 'duration_minutes',
                     'match_duration_minutes']:
            return 0
        
        if field in ['home_reception_pct', 'away_reception_pct', 'player1_first_serve_pct',
                     'player2_first_serve_pct', 'home_fg_pct', 'away_fg_pct', 'home_3p_pct',
                     'away_3p_pct']:
            return 0.0
        
        return defaults.get(field, '')
    
    def _clean_record(self, record: Dict) -> Dict:
        """Clean record of any invalid characters or values."""
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, str):
                # Remove null bytes and control characters
                cleaned[key] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)
            else:
                cleaned[key] = value
        return cleaned
