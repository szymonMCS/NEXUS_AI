"""
Meta-Learning for warm start in AutoML.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class MetaLearner:
    """Meta-learning to initialize search from past experiences."""
    
    def __init__(self, meta_data_path: str = "data/meta_learning"):
        self.meta_data_path = Path(meta_data_path)
        self.meta_data_path.mkdir(parents=True, exist_ok=True)
        self.experiences: List[Dict] = []
        self._load_experiences()
    
    def _load_experiences(self):
        """Load past experiences."""
        for file_path in self.meta_data_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    self.experiences.append(json.load(f))
            except:
                continue
    
    def get_warm_start(self, sport: str, league: str) -> Any:
        """Get warm start configuration based on similar past tasks."""
        similar = [
            exp for exp in self.experiences
            if exp.get('sport') == sport or exp.get('league') == league
        ]
        
        if not similar:
            return None
        
        best_configs = [exp.get('best_config') for exp in similar if exp.get('best_config')]
        return best_configs[-1] if best_configs else None
    
    def save_experience(self, sport: str, league: str, result: Any):
        """Save experience for future meta-learning."""
        experience = {
            'sport': sport,
            'league': league,
            'timestamp': datetime.utcnow().isoformat(),
            'best_config': result.best_config.__dict__ if hasattr(result, 'best_config') else result,
        }
        
        file_path = self.meta_data_path / f"{sport}_{league}_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(file_path, 'w') as f:
            json.dump(experience, f, indent=2)
