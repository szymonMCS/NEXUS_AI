"""
Sports Datasets Integration Module.

Integrates open sports datasets for ML training based on sport_datasets_AI_report.md:
- Basketball: NBA Play-by-Play, Shot Logs
- Tennis: ATP/WTA rankings, match history
- Hockey: NHL data
- Baseball: MLB Statcast
- Handball: European leagues data
"""

from core.datasets.base import (
    SportsDataSource,
    DatasetInfo,
    RawMatch,
    DatasetQuality,
    DatasetLicense,
)
from core.datasets.basketball_data import BasketballDataSource
from core.datasets.tennis_data import TennisDataSource
from core.datasets.hockey_data import HockeyDataSource
from core.datasets.baseball_data import BaseballDataSource
from core.datasets.handball_data import HandballDataSource
from core.datasets.manager import DatasetManager, DatasetConfig

__all__ = [
    "SportsDataSource",
    "DatasetInfo",
    "RawMatch",
    "DatasetQuality",
    "DatasetLicense",
    "BasketballDataSource",
    "TennisDataSource",
    "HockeyDataSource",
    "BaseballDataSource",
    "HandballDataSource",
    "DatasetManager",
    "DatasetConfig",
]
