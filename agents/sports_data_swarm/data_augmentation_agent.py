"""
Data Augmentation Agent - Augments and enriches sports data for better ML training.

Techniques used:
1. Gaussian Noise Injection - add small random variations to numeric features
2. SMOTE-like Synthetic Generation - create synthetic samples based on existing ones
3. Feature Engineering - create new derived features
4. Time-based Features - add temporal context
5. Rolling Averages - calculate form indicators
6. Interaction Features - combine existing features
"""

import random
import statistics
import copy
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class DataAugmentationAgent(BaseAgent):
    """
    Agent responsible for augmenting sports data to improve ML model training.
    
    Augmentation techniques:
    - Gaussian noise injection for numerical stability
    - Synthetic sample generation based on statistical distributions
    - Feature engineering (rolling averages, form indicators)
    - Interaction features
    - Time-based features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("DataAugmentationAgent", config)
        self.augmentation_factor = config.get('augmentation_factor', 2.0) if config else 2.0
        self.noise_std = config.get('noise_std', 0.02) if config else 0.02  # 2% noise
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute data augmentation task.
        
        Task params:
        - sport: Sport type
        - data: Original dataset
        - target_multiplier: How many times to multiply the dataset
        - techniques: List of augmentation techniques to apply
        """
        self.status = AgentStatus.RUNNING
        
        try:
            sport = task['sport']
            data = task.get('data', {})
            records = data.get('records', [])
            target_multiplier = task.get('target_multiplier', self.augmentation_factor)
            techniques = task.get('techniques', ['all'])
            
            if not records:
                return TaskResult(success=False, error="No records to augment")
            
            logger.info(f"[{self.name}] Augmenting {len(records)} {sport} records")
            logger.info(f"[{self.name}] Target multiplier: {target_multiplier}x")
            
            augmented_records = records.copy()
            
            # Apply augmentation techniques
            if 'noise' in techniques or 'all' in techniques:
                noise_records = await self._add_gaussian_noise(records, sport)
                augmented_records.extend(noise_records)
                logger.info(f"[{self.name}] Added {len(noise_records)} noise-augmented records")
            
            if 'synthetic' in techniques or 'all' in techniques:
                synthetic_records = await self._generate_synthetic_samples(records, sport, target_multiplier)
                augmented_records.extend(synthetic_records)
                logger.info(f"[{self.name}] Added {len(synthetic_records)} synthetic records")
            
            if 'features' in techniques or 'all' in techniques:
                await self._engineer_features(augmented_records, sport)
                logger.info(f"[{self.name}] Engineered features for all records")
            
            if 'rolling' in techniques or 'all' in techniques:
                await self._add_rolling_averages(augmented_records, sport)
                logger.info(f"[{self.name}] Added rolling averages")
            
            if 'interactions' in techniques or 'all' in techniques:
                await self._add_interaction_features(augmented_records, sport)
                logger.info(f"[{self.name}] Added interaction features")
            
            # Final statistics
            logger.info(f"[{self.name}] Original: {len(records)}, Augmented: {len(augmented_records)}")
            logger.info(f"[{self.name}] Multiplication factor: {len(augmented_records)/len(records):.2f}x")
            
            self.status = AgentStatus.COMPLETED
            return TaskResult(
                success=True,
                data={
                    'original_count': len(records),
                    'augmented_count': len(augmented_records),
                    'multiplier': len(augmented_records) / len(records),
                    'records': augmented_records
                },
                records_processed=len(augmented_records),
                metadata={
                    'techniques_applied': techniques,
                    'sport': sport
                }
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            self.status = AgentStatus.ERROR
            return TaskResult(success=False, error=str(e))
    
    async def _add_gaussian_noise(self, records: List[Dict], sport: str) -> List[Dict]:
        """Add Gaussian noise to numerical features."""
        noisy_records = []
        
        # Define fields to add noise to (by sport)
        noise_fields = {
            'basketball': ['home_score', 'away_score', 'home_fg_pct', 'home_3p_pct', 'away_fg_pct'],
            'volleyball': ['home_total_points', 'away_total_points', 'home_attacks', 'away_attacks'],
            'handball': ['home_score', 'away_score', 'home_shots', 'away_shots'],
            'tennis': ['player1_aces', 'player2_aces', 'player1_first_serve_pct', 'player2_first_serve_pct'],
            'football': ['home_goals', 'away_goals', 'home_shots', 'away_shots', 'home_xg', 'away_xg']
        }
        
        fields = noise_fields.get(sport, [])
        
        for record in records:
            new_record = copy.deepcopy(record)
            new_record['augmented'] = True
            new_record['augmentation_type'] = 'noise'
            
            for field in fields:
                if field in new_record and isinstance(new_record[field], (int, float)):
                    original_value = new_record[field]
                    # Add Gaussian noise proportional to value
                    noise = random.gauss(0, self.noise_std * abs(original_value) if original_value != 0 else self.noise_std)
                    new_record[field] = round(original_value + noise, 2)
            
            noisy_records.append(new_record)
        
        return noisy_records
    
    async def _generate_synthetic_samples(self, records: List[Dict], sport: str, 
                                           multiplier: float) -> List[Dict]:
        """Generate synthetic samples using statistical interpolation."""
        synthetic_records = []
        
        target_count = int(len(records) * (multiplier - 1))
        
        if len(records) < 2:
            return synthetic_records
        
        # Group records by similar characteristics
        groups = self._group_similar_records(records, sport)
        
        generated = 0
        attempts = 0
        max_attempts = target_count * 3
        
        while generated < target_count and attempts < max_attempts:
            attempts += 1
            
            # Pick a group and two random records from it
            group = random.choice(groups) if groups else records
            if len(group) < 2:
                continue
            
            record1, record2 = random.sample(group, 2)
            
            # Create interpolated record
            synthetic = self._interpolate_records(record1, record2, sport)
            
            if synthetic:
                synthetic['synthetic'] = True
                synthetic['augmentation_type'] = 'synthetic'
                synthetic_records.append(synthetic)
                generated += 1
        
        return synthetic_records
    
    def _group_similar_records(self, records: List[Dict], sport: str) -> List[List[Dict]]:
        """Group records by similar characteristics for better interpolation."""
        groups = []
        
        # Define grouping keys by sport
        group_keys = {
            'basketball': ['league', 'home_team'],
            'football': ['league', 'home_team'],
            'tennis': ['tournament', 'surface'],
            'volleyball': ['league', 'home_team'],
            'handball': ['league', 'home_team']
        }
        
        keys = group_keys.get(sport, ['league'])
        
        # Simple grouping
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for record in records:
            key = tuple(record.get(k, 'unknown') for k in keys)
            grouped[key].append(record)
        
        return list(grouped.values())
    
    def _interpolate_records(self, r1: Dict, r2: Dict, sport: str) -> Optional[Dict]:
        """Interpolate between two records to create a synthetic one."""
        synthetic = {}
        
        # Random interpolation factor
        alpha = random.uniform(0.3, 0.7)
        
        for key in r1.keys():
            if key in ['game_id', 'match_id', 'date', 'home_team', 'away_team', 
                      'player1_name', 'player2_name', 'league', 'tournament']:
                # Keep categorical values from one record
                synthetic[key] = r1[key]
            elif isinstance(r1[key], (int, float)) and isinstance(r2.get(key), (int, float)):
                # Interpolate numerical values
                val1 = r1[key]
                val2 = r2.get(key, val1)
                synthetic[key] = round(val1 * alpha + val2 * (1 - alpha), 2)
            elif isinstance(r1[key], str):
                synthetic[key] = r1[key]
            else:
                synthetic[key] = r1[key]
        
        # Generate new ID
        synthetic['game_id'] = f"synth_{random.randint(100000, 999999)}"
        synthetic['match_id'] = synthetic['game_id']
        
        return synthetic
    
    async def _engineer_features(self, records: List[Dict], sport: str):
        """Add engineered features to records."""
        if sport == 'basketball':
            await self._engineer_basketball_features(records)
        elif sport == 'football':
            await self._engineer_football_features(records)
        elif sport == 'tennis':
            await self._engineer_tennis_features(records)
        elif sport == 'volleyball':
            await self._engineer_volleyball_features(records)
        elif sport == 'handball':
            await self._engineer_handball_features(records)
    
    async def _engineer_basketball_features(self, records: List[Dict]):
        """Add basketball-specific engineered features."""
        for r in records:
            # Shooting efficiency
            if r.get('home_fg_attempts', 0) > 0:
                r['home_ts_pct'] = round(r.get('home_score', 0) / (2 * r.get('home_fg_attempts', 1)), 3)
            if r.get('away_fg_attempts', 0) > 0:
                r['away_ts_pct'] = round(r.get('away_score', 0) / (2 * r.get('away_fg_attempts', 1)), 3)
            
            # Rebound rates
            total_rebounds = r.get('home_rebounds', 0) + r.get('away_rebounds', 0)
            if total_rebounds > 0:
                r['home_rebound_rate'] = round(r.get('home_rebounds', 0) / total_rebounds, 3)
                r['away_rebound_rate'] = round(r.get('away_rebounds', 0) / total_rebounds, 3)
            
            # Assist-to-turnover ratio
            if r.get('home_turnovers', 0) > 0:
                r['home_ast_to_ratio'] = round(r.get('home_assists', 0) / r.get('home_turnovers', 1), 2)
            if r.get('away_turnovers', 0) > 0:
                r['away_ast_to_ratio'] = round(r.get('away_assists', 0) / r.get('away_turnovers', 1), 2)
            
            # Four factors
            if r.get('home_fg_attempts', 0) > 0:
                r['home_efg_pct'] = round((r.get('home_fg_made', 0) + 0.5 * r.get('home_3p_made', 0)) / r.get('home_fg_attempts', 1), 3)
            if r.get('away_fg_attempts', 0) > 0:
                r['away_efg_pct'] = round((r.get('away_fg_made', 0) + 0.5 * r.get('away_3p_made', 0)) / r.get('away_fg_attempts', 1), 3)
    
    async def _engineer_football_features(self, records: List[Dict]):
        """Add football-specific engineered features."""
        for r in records:
            # Shot accuracy
            if r.get('home_shots', 0) > 0:
                r['home_shot_accuracy'] = round(r.get('home_shots_on_target', 0) / r.get('home_shots', 1), 3)
            if r.get('away_shots', 0) > 0:
                r['away_shot_accuracy'] = round(r.get('away_shots_on_target', 0) / r.get('away_shots', 1), 3)
            
            # Conversion rate (goals per shot)
            if r.get('home_shots', 0) > 0:
                r['home_conversion_rate'] = round(r.get('home_goals', 0) / r.get('home_shots', 1), 3)
            if r.get('away_shots', 0) > 0:
                r['away_conversion_rate'] = round(r.get('away_goals', 0) / r.get('away_shots', 1), 3)
            
            xG_diff = r.get('home_xg', 0) - r.get('away_xg', 0)
            goals_diff = r.get('home_goals', 0) - r.get('away_goals', 0)
            r['xg_performance_diff'] = round(goals_diff - xG_diff, 2)
            
            # Dominance index
            poss_factor = abs(r.get('home_possession', 50) - 50) / 50
            shot_factor = abs(r.get('home_shots', 0) - r.get('away_shots', 0)) / max(r.get('home_shots', 1) + r.get('away_shots', 1), 1)
            r['dominance_index'] = round((poss_factor + shot_factor) / 2, 3)
    
    async def _engineer_tennis_features(self, records: List[Dict]):
        """Add tennis-specific engineered features."""
        for r in records:
            # Serve efficiency
            if r.get('player1_first_serve_pct', 0) > 0:
                r['player1_serve_efficiency'] = round(
                    r.get('player1_first_serve_won', 0) * r.get('player1_first_serve_pct', 0), 3
                )
            if r.get('player2_first_serve_pct', 0) > 0:
                r['player2_serve_efficiency'] = round(
                    r.get('player2_first_serve_won', 0) * r.get('player2_first_serve_pct', 0), 3
                )
            
            # Break point conversion
            if r.get('player1_break_points_total', 0) > 0:
                r['player1_bp_conversion'] = round(
                    r.get('player1_break_points_won', 0) / r.get('player1_break_points_total', 1), 3
                )
            if r.get('player2_break_points_total', 0) > 0:
                r['player2_bp_conversion'] = round(
                    r.get('player2_break_points_won', 0) / r.get('player2_break_points_total', 1), 3
                )
            
            # Aggression index (winners + unforced errors)
            r['player1_aggression'] = r.get('player1_winners', 0) + r.get('player1_unforced_errors', 0)
            r['player2_aggression'] = r.get('player2_winners', 0) + r.get('player2_unforced_errors', 0)
    
    async def _engineer_volleyball_features(self, records: List[Dict]):
        """Add volleyball-specific engineered features."""
        for r in records:
            # Attack efficiency
            if r.get('home_attacks', 0) > 0:
                r['home_attack_efficiency'] = round(
                    (r.get('home_total_points', 0) - r.get('home_errors', 0)) / r.get('home_attacks', 1), 3
                )
            if r.get('away_attacks', 0) > 0:
                r['away_attack_efficiency'] = round(
                    (r.get('away_total_points', 0) - r.get('away_errors', 0)) / r.get('away_attacks', 1), 3
                )
            
            # Point scoring rate
            if r.get('home_sets_won', 0) > 0:
                r['home_points_per_set'] = round(r.get('home_total_points', 0) / r.get('home_sets_won', 1), 2)
            if r.get('away_sets_won', 0) > 0:
                r['away_points_per_set'] = round(r.get('away_total_points', 0) / r.get('away_sets_won', 1), 2)
    
    async def _engineer_handball_features(self, records: List[Dict]):
        """Add handball-specific engineered features."""
        for r in records:
            # Shot efficiency
            if r.get('home_shots', 0) > 0:
                r['home_shot_efficiency'] = round(r.get('home_score', 0) / r.get('home_shots', 1), 3)
            if r.get('away_shots', 0) > 0:
                r['away_shot_efficiency'] = round(r.get('away_score', 0) / r.get('away_shots', 1), 3)
            
            # 7m conversion
            if r.get('home_7m_attempts', 0) > 0:
                r['home_7m_conversion'] = round(r.get('home_7m_goals', 0) / r.get('home_7m_attempts', 1), 3)
            if r.get('away_7m_attempts', 0) > 0:
                r['away_7m_conversion'] = round(r.get('away_7m_goals', 0) / r.get('away_7m_attempts', 1), 3)
            
            # Goalkeeper efficiency
            if r.get('home_gk_shots', 0) > 0:
                r['home_gk_efficiency'] = round(r.get('home_gk_saves', 0) / r.get('home_gk_shots', 1), 3)
            if r.get('away_gk_shots', 0) > 0:
                r['away_gk_efficiency'] = round(r.get('away_gk_saves', 0) / r.get('away_gk_shots', 1), 3)
    
    async def _add_rolling_averages(self, records: List[Dict], sport: str):
        """Add rolling average features (simulated based on current match data)."""
        # Sort by date if possible
        try:
            records_sorted = sorted(records, key=lambda x: str(x.get('date', '')))
        except:
            records_sorted = records
        
        # Add simulated form indicators
        for i, r in enumerate(records_sorted):
            if sport in ['basketball', 'football', 'handball']:
                # Simulate last 5 games form
                r['home_form_goals_avg'] = round(r.get('home_score', r.get('home_goals', 0)) * random.uniform(0.9, 1.1), 2)
                r['away_form_goals_avg'] = round(r.get('away_score', r.get('away_goals', 0)) * random.uniform(0.9, 1.1), 2)
            
            elif sport == 'tennis':
                r['player1_form_sets_won_avg'] = round(r.get('player1_sets_won', 0) * random.uniform(0.9, 1.1), 2)
                r['player2_form_sets_won_avg'] = round(r.get('player2_sets_won', 0) * random.uniform(0.9, 1.1), 2)
            
            elif sport == 'volleyball':
                r['home_form_sets_avg'] = round(r.get('home_sets_won', 0) * random.uniform(0.9, 1.1), 2)
                r['away_form_sets_avg'] = round(r.get('away_sets_won', 0) * random.uniform(0.9, 1.1), 2)
    
    async def _add_interaction_features(self, records: List[Dict], sport: str):
        """Add interaction features between variables."""
        for r in records:
            if sport == 'basketball':
                # Home advantage interaction
                r['home_fg_x_rebounds'] = round(r.get('home_fg_pct', 0) * r.get('home_rebounds', 0), 2)
                r['away_fg_x_rebounds'] = round(r.get('away_fg_pct', 0) * r.get('away_rebounds', 0), 2)
                
            elif sport == 'football':
                # Possession x shots interaction
                r['home_poss_x_shots'] = round(r.get('home_possession', 0) * r.get('home_shots', 0) / 100, 2)
                r['away_poss_x_shots'] = round(r.get('away_possession', 0) * r.get('away_shots', 0) / 100, 2)
                
            elif sport == 'tennis':
                # Aces x double faults (risk indicator)
                r['player1_risk_indicator'] = r.get('player1_aces', 0) - r.get('player1_double_faults', 0)
                r['player2_risk_indicator'] = r.get('player2_aces', 0) - r.get('player2_double_faults', 0)
