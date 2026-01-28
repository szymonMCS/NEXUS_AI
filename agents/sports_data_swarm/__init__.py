"""
Sports Data Swarm - Agent system for collecting sports data from web sources.

This module provides a swarm of specialized agents for:
- Basketball data collection
- Volleyball data collection
- Handball data collection
- Tennis data collection

Agent Structure:
- ManagerAgent: Coordinates the entire process
- SportAgents: Specialized agents for each sport
- DataAcquisitionAgent: Handles web scraping and API calls
- FormattingAgent: Normalizes and formats data
- StorageAgent: Saves data to datasets
- EvaluatorAgents: Evaluate data quality for each sport
"""

from .manager_agent import ManagerAgent
from .sport_agents import BasketballAgent, VolleyballAgent, HandballAgent, TennisAgent
from .data_acquisition_agent import DataAcquisitionAgent
from .formatting_agent import FormattingAgent
from .storage_agent import StorageAgent
from .evaluator_agents import (
    BasketballEvaluatorAgent,
    VolleyballEvaluatorAgent,
    HandballEvaluatorAgent,
    TennisEvaluatorAgent
)

__all__ = [
    'ManagerAgent',
    'BasketballAgent',
    'VolleyballAgent',
    'HandballAgent',
    'TennisAgent',
    'DataAcquisitionAgent',
    'FormattingAgent',
    'StorageAgent',
    'BasketballEvaluatorAgent',
    'VolleyballEvaluatorAgent',
    'HandballEvaluatorAgent',
    'TennisEvaluatorAgent',
]
