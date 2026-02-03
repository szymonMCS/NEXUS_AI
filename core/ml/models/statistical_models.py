"""
Statistical models for baseline predictions.
Used as fallback when ML models are unavailable.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.ml.models.base import BasePredictor, PredictionResult


@dataclass
class EloRating:
    """ELO rating for a player/team."""
    rating: float = 1500.0
    matches_played: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def expected_score(self, opponent_rating: float) -> float:
        """Calculate expected score against opponent."""
        return 1 / (1 + 10 ** ((opponent_rating - self.rating) / 400))
    
    def update(self, actual_score: float, expected_score: float, k: int = 32):
        """Update ELO rating after match."""
        self.rating += k * (actual_score - expected_score)
        self.matches_played += 1
        self.last_updated = datetime.now()


class EloPredictor(BasePredictor):
    """
    ELO-based predictor.
    Classic rating system adapted for sports.
    """
    
    def __init__(self, name: str = "ELO", version: str = "1.0", sport: str = "general"):
        super().__init__(name, version, sport)
        self.ratings: Dict[str, EloRating] = {}
        self.k_factor = 32  # Standard K-factor
        self.home_advantage = 100  # Rating points for home advantage
        
    def get_or_create_rating(self, player_name: str) -> EloRating:
        """Get existing rating or create new one."""
        if player_name not in self.ratings:
            self.ratings[player_name] = EloRating()
        return self.ratings[player_name]
    
    def fit(self, matches: List[Dict[str, Any]], **kwargs):
        """
        Train ELO ratings from historical matches.
        
        Args:
            matches: List of match dicts with keys:
                - home_player: str
                - away_player: str
                - home_score: int
                - away_score: int
                - date: datetime (optional)
        """
        # Sort by date if available
        matches = sorted(matches, key=lambda x: x.get('date', datetime.now()))
        
        for match in matches:
            home_name = match.get('home_player') or match.get('home_team')
            away_name = match.get('away_player') or match.get('away_team')
            
            if not home_name or not away_name:
                continue
            
            home_rating = self.get_or_create_rating(home_name)
            away_rating = self.get_or_create_rating(away_name)
            
            # Calculate expected scores (with home advantage)
            home_expected = home_rating.expected_score(away_rating.rating + self.home_advantage)
            away_expected = away_rating.expected_score(home_rating.rating - self.home_advantage)
            
            # Determine actual result
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if home_score > away_score:
                home_actual, away_actual = 1.0, 0.0
            elif home_score < away_score:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5
            
            # Update ratings
            home_rating.update(home_actual, home_expected, self.k_factor)
            away_rating.update(away_actual, away_expected, self.k_factor)
        
        self.is_trained = True
        self.update_metadata(
            training_samples=len(matches),
            hyperparameters={
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage
            }
        )
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Predict using ELO ratings.
        
        Features needed:
        - home_player or home_team
        - away_player or away_team
        """
        home_name = features.get('home_player') or features.get('home_team')
        away_name = features.get('away_player') or features.get('away_team')
        
        if not home_name or not away_name:
            return PredictionResult(0.5, 0.5, confidence=0.0, model_name=self.name)
        
        home_rating = self.get_or_create_rating(home_name)
        away_rating = self.get_or_create_rating(away_name)
        
        # Expected win probability
        home_prob = home_rating.expected_score(away_rating.rating + self.home_advantage)
        away_prob = 1 - home_prob
        
        # Confidence based on number of matches played
        total_matches = home_rating.matches_played + away_rating.matches_played
        confidence = min(0.9, 0.3 + (total_matches / 100))
        
        return PredictionResult(
            home_win_prob=float(home_prob),
            away_win_prob=float(away_prob),
            confidence=float(confidence),
            model_name=self.name,
            version=self.version,
            features_used=['elo_rating', 'matches_played']
        )
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction."""
        return [self.predict(f) for f in features_list]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """ELO only uses rating difference."""
        return {"elo_rating_diff": 1.0}
    
    def get_rating(self, player_name: str) -> Optional[float]:
        """Get rating for a specific player."""
        if player_name in self.ratings:
            return self.ratings[player_name].rating
        return None
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save ELO ratings."""
        import json
        from pathlib import Path
        
        if filepath is None:
            filepath = f"models/{self.sport}/elo_ratings.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "ratings": {
                name: {
                    "rating": r.rating,
                    "matches_played": r.matches_played,
                    "last_updated": r.last_updated.isoformat() if r.last_updated else None
                }
                for name, r in self.ratings.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load(self, filepath: str) -> bool:
        """Load ELO ratings."""
        import json
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.k_factor = data.get('k_factor', 32)
            self.home_advantage = data.get('home_advantage', 100)
            
            for name, r_data in data.get('ratings', {}).items():
                rating = EloRating(
                    rating=r_data['rating'],
                    matches_played=r_data['matches_played']
                )
                if r_data.get('last_updated'):
                    rating.last_updated = datetime.fromisoformat(r_data['last_updated'])
                self.ratings[name] = rating
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading ELO ratings: {e}")
            return False


class FormPredictor(BasePredictor):
    """
    Form-based predictor.
    Uses recent match results to estimate current form.
    """
    
    def __init__(self, name: str = "Form", version: str = "1.0", sport: str = "general"):
        super().__init__(name, version, sport)
        self.form_history: Dict[str, List[Dict]] = {}  # player -> list of recent matches
        self.form_window = 5  # Number of recent matches to consider
        self.decay_factor = 0.9  # Weight decay for older matches
        
    def add_match_result(self, player_name: str, result: str, date: datetime = None):
        """
        Add match result to player's history.
        
        Args:
            player_name: Player/team name
            result: 'W', 'L', or 'D' (win, loss, draw)
            date: Match date
        """
        if player_name not in self.form_history:
            self.form_history[player_name] = []
        
        if date is None:
            date = datetime.now()
        
        self.form_history[player_name].append({
            'result': result,
            'date': date
        })
        
        # Keep only recent matches
        self.form_history[player_name] = sorted(
            self.form_history[player_name],
            key=lambda x: x['date'],
            reverse=True
        )[:self.form_window * 2]
    
    def calculate_form_score(self, player_name: str) -> float:
        """
        Calculate form score for a player.
        Returns value between 0 and 1.
        """
        if player_name not in self.form_history:
            return 0.5  # Unknown form
        
        matches = self.form_history[player_name][:self.form_window]
        
        if not matches:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, match in enumerate(matches):
            weight = self.decay_factor ** i
            
            if match['result'] == 'W':
                score = 1.0
            elif match['result'] == 'D':
                score = 0.5
            else:  # 'L'
                score = 0.0
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return total_score / total_weight
    
    def fit(self, matches: List[Dict[str, Any]], **kwargs):
        """
        Build form history from matches.
        
        Args:
            matches: List of matches with results
        """
        for match in matches:
            home_name = match.get('home_player') or match.get('home_team')
            away_name = match.get('away_player') or match.get('away_team')
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            date = match.get('date', datetime.now())
            
            if not home_name or not away_name:
                continue
            
            # Determine results
            if home_score > away_score:
                self.add_match_result(home_name, 'W', date)
                self.add_match_result(away_name, 'L', date)
            elif home_score < away_score:
                self.add_match_result(home_name, 'L', date)
                self.add_match_result(away_name, 'W', date)
            else:
                self.add_match_result(home_name, 'D', date)
                self.add_match_result(away_name, 'D', date)
        
        self.is_trained = True
        self.update_metadata(training_samples=len(matches))
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Predict based on form.
        
        Features needed:
        - home_player or home_team
        - away_player or away_team
        """
        home_name = features.get('home_player') or features.get('home_team')
        away_name = features.get('away_player') or features.get('away_team')
        
        if not home_name or not away_name:
            return PredictionResult(0.5, 0.5, confidence=0.0, model_name=self.name)
        
        home_form = self.calculate_form_score(home_name)
        away_form = self.calculate_form_score(away_name)
        
        # Calculate win probability from form
        total_form = home_form + away_form
        if total_form == 0:
            home_prob = 0.5
        else:
            home_prob = home_form / total_form
        
        # Apply home advantage
        home_prob = min(0.9, home_prob * 1.1)
        away_prob = 1 - home_prob
        
        # Confidence based on how many matches we have
        home_matches = len(self.form_history.get(home_name, []))
        away_matches = len(self.form_history.get(away_name, []))
        avg_matches = (home_matches + away_matches) / 2
        confidence = min(0.8, 0.3 + (avg_matches / 20))
        
        return PredictionResult(
            home_win_prob=float(home_prob),
            away_win_prob=float(away_prob),
            confidence=float(confidence),
            model_name=self.name,
            version=self.version,
            features_used=['recent_form', 'win_streak']
        )
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction."""
        return [self.predict(f) for f in features_list]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Form features."""
        return {"recent_form": 0.6, "home_advantage": 0.4}


class StatisticalEnsemble(BasePredictor):
    """
    Ensemble of statistical models (ELO + Form + H2H).
    Used when ML models are not available.
    """
    
    def __init__(self, name: str = "StatisticalEnsemble", version: str = "1.0", sport: str = "general"):
        super().__init__(name, version, sport)
        self.elo = EloPredictor(sport=sport)
        self.form = FormPredictor(sport=sport)
        self.h2h_history: Dict[str, Dict[str, List[Dict]]] = {}  # player1 -> player2 -> matches
        
    def add_h2h_match(self, player1: str, player2: str, winner: str, date: datetime = None):
        """Add head-to-head match result."""
        if player1 not in self.h2h_history:
            self.h2h_history[player1] = {}
        if player2 not in self.h2h_history[player1]:
            self.h2h_history[player1][player2] = []
        
        if date is None:
            date = datetime.now()
        
        self.h2h_history[player1][player2].append({
            'winner': winner,
            'date': date
        })
    
    def get_h2h_advantage(self, player1: str, player2: str) -> float:
        """
        Calculate H2H advantage for player1 vs player2.
        Returns value between -1 and 1.
        """
        matches = []
        
        # Get matches from both directions
        if player1 in self.h2h_history and player2 in self.h2h_history[player1]:
            matches.extend(self.h2h_history[player1][player2])
        if player2 in self.h2h_history and player1 in self.h2h_history[player2]:
            for m in self.h2h_history[player2][player1]:
                # Invert winner
                inverted = {'winner': player1 if m['winner'] == player2 else player2, 'date': m['date']}
                matches.append(inverted)
        
        if not matches:
            return 0.0
        
        # Count wins
        p1_wins = sum(1 for m in matches if m['winner'] == player1)
        p2_wins = sum(1 for m in matches if m['winner'] == player2)
        total = p1_wins + p2_wins
        
        if total == 0:
            return 0.0
        
        # Normalize to -1 to 1
        return (p1_wins - p2_wins) / total
    
    def fit(self, matches: List[Dict[str, Any]], **kwargs):
        """Train all statistical models."""
        # Train ELO and Form
        self.elo.fit(matches)
        self.form.fit(matches)
        
        # Build H2H history
        for match in matches:
            home_name = match.get('home_player') or match.get('home_team')
            away_name = match.get('away_player') or match.get('away_team')
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            date = match.get('date', datetime.now())
            
            if not home_name or not away_name:
                continue
            
            if home_score > away_score:
                winner = home_name
            elif home_score < away_score:
                winner = away_name
            else:
                continue  # Skip draws for H2H
            
            self.add_h2h_match(home_name, away_name, winner, date)
        
        self.is_trained = True
        self.update_metadata(training_samples=len(matches))
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Combine ELO, Form, and H2H predictions.
        """
        home_name = features.get('home_player') or features.get('home_team')
        away_name = features.get('away_player') or features.get('away_team')
        
        if not home_name or not away_name:
            return PredictionResult(0.5, 0.5, confidence=0.0, model_name=self.name)
        
        # Get individual predictions
        elo_pred = self.elo.predict(features)
        form_pred = self.form.predict(features)
        
        # Get H2H advantage
        h2h_adv = self.get_h2h_advantage(home_name, away_name)
        
        # Weighted combination
        # ELO: 40%, Form: 35%, H2H: 25%
        elo_weight = 0.4
        form_weight = 0.35
        h2h_weight = 0.25
        
        home_prob = (
            elo_pred.home_win_prob * elo_weight +
            form_pred.home_win_prob * form_weight +
            (0.5 + h2h_adv * 0.2) * h2h_weight
        )
        
        # Normalize
        home_prob = max(0.05, min(0.95, home_prob))
        away_prob = 1 - home_prob
        
        # Confidence based on data availability
        elo_conf = elo_pred.confidence
        form_conf = form_pred.confidence
        h2h_matches = len(self.h2h_history.get(home_name, {}).get(away_name, []))
        h2h_conf = min(1.0, h2h_matches / 5)
        
        confidence = elo_conf * 0.4 + form_conf * 0.35 + h2h_conf * 0.25
        
        return PredictionResult(
            home_win_prob=float(home_prob),
            away_win_prob=float(away_prob),
            confidence=float(confidence),
            model_name=self.name,
            version=self.version,
            features_used=['elo_rating', 'recent_form', 'head_to_head']
        )
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction."""
        return [self.predict(f) for f in features_list]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Feature importance."""
        return {
            "elo_rating": 0.4,
            "recent_form": 0.35,
            "head_to_head": 0.25
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save all models."""
        import json
        from pathlib import Path
        
        if filepath is None:
            filepath = f"models/{self.sport}/statistical_ensemble.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "elo": {
                "k_factor": self.elo.k_factor,
                "home_advantage": self.elo.home_advantage,
                "ratings": {
                    name: {"rating": r.rating, "matches": r.matches_played}
                    for name, r in self.elo.ratings.items()
                }
            },
            "form": {
                "window": self.form.form_window,
                "history": {
                    name: [
                        {"result": m['result'], "date": m['date'].isoformat()}
                        for m in matches[-10:]  # Last 10 only
                    ]
                    for name, matches in self.form.form_history.items()
                }
            },
            "h2h": {
                f"{p1}_vs_{p2}": [
                    {"winner": m['winner'], "date": m['date'].isoformat()}
                    for m in matches[-10:]
                ]
                for p1, opponents in self.h2h_history.items()
                for p2, matches in opponents.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load(self, filepath: str) -> bool:
        """Load all models."""
        import json
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load ELO
            elo_data = data.get('elo', {})
            self.elo.k_factor = elo_data.get('k_factor', 32)
            self.elo.home_advantage = elo_data.get('home_advantage', 100)
            for name, r_data in elo_data.get('ratings', {}).items():
                self.elo.ratings[name] = EloRating(
                    rating=r_data['rating'],
                    matches_played=r_data['matches']
                )
            
            # Load Form
            form_data = data.get('form', {})
            self.form.form_window = form_data.get('window', 5)
            for name, matches in form_data.get('history', {}).items():
                self.form.form_history[name] = [
                    {'result': m['result'], 'date': datetime.fromisoformat(m['date'])}
                    for m in matches
                ]
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading statistical ensemble: {e}")
            return False
