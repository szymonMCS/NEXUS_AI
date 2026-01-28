"""
Base Agent class for Sports Data Swarm.
Provides common functionality for all agents.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status states."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


@dataclass
class TaskResult:
    """Result of agent task execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    records_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all agents in the Sports Data Swarm.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = AgentStatus.IDLE
        self.config = config or {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        self.results: List[TaskResult] = []
        self._setup_handlers()
        
        # Load API keys from environment
        self.brave_api_key = os.getenv('BRAVE_API_KEY', '')
        self.serper_api_key = os.getenv('SERPER_API_KEY', '')
        
    def _setup_handlers(self):
        """Setup message handlers - override in subclasses."""
        pass
    
    async def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any], 
                          correlation_id: Optional[str] = None) -> AgentMessage:
        """Send message to another agent."""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        logger.info(f"[{self.name}] Sending {message_type} to {recipient}")
        return message
    
    async def receive_message(self, message: AgentMessage):
        """Receive and process a message."""
        logger.info(f"[{self.name}] Received {message.message_type} from {message.sender}")
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            logger.warning(f"[{self.name}] No handler for message type: {message.message_type}")
            return None
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """Execute agent's main task."""
        pass
    
    async def search_brave(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """Search using Brave Search API."""
        if not self.brave_api_key:
            logger.error("BRAVE_API_KEY not found in environment")
            return []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.brave_api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": min(count, 20),
            "offset": 0,
            "mkt": "en-US",
            "safesearch": "off",
            "freshness": "all",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('web', {}).get('results', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('url', ''),
                                'description': item.get('description', ''),
                                'source': 'brave'
                            })
                        return results
                    else:
                        logger.error(f"Brave API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []
    
    async def search_serper(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Serper API (Google Search)."""
        if not self.serper_api_key:
            logger.error("SERPER_API_KEY not found in environment")
            return []
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": min(num_results, 20),
            "page": 1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('organic', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('link', ''),
                                'description': item.get('snippet', ''),
                                'source': 'serper'
                            })
                        return results
                    else:
                        logger.error(f"Serper API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []
    
    async def fetch_url(self, url: str, headers: Optional[Dict] = None) -> Optional[str]:
        """Fetch content from URL."""
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        if headers:
            default_headers.update(headers)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=default_headers, timeout=30) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def log_progress(self, message: str, current: int = 0, total: int = 0):
        """Log progress message."""
        if total > 0:
            pct = (current / total) * 100
            logger.info(f"[{self.name}] {message} - {current}/{total} ({pct:.1f}%)")
        else:
            logger.info(f"[{self.name}] {message}")
    
    def save_checkpoint(self, data: Dict[str, Any], filename: str):
        """Save checkpoint data."""
        checkpoint_dir = "datasets/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = f"{checkpoint_dir}/{filename}_{self.name}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[{self.name}] Checkpoint saved: {filepath}")
