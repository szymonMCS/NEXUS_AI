# mcp_servers/__init__.py
"""
MCP (Model Context Protocol) servers for NEXUS AI.

Each server provides specialized tools for LangGraph agents:
- news_server: News aggregation, validation, injury extraction
- odds_server: Odds fetching, value bet calculation, Kelly criterion
- tennis_server: Tennis match data, rankings, H2H
- basketball_server: Basketball match data, standings, team stats
- alerts_server: Alert creation and notification management
- evaluation_server: Data quality evaluation, adjusted value calculation
"""

from mcp_servers.news_server import server as news_server
from mcp_servers.odds_server import server as odds_server
from mcp_servers.tennis_server import server as tennis_server
from mcp_servers.basketball_server import server as basketball_server
from mcp_servers.alerts_server import server as alerts_server
from mcp_servers.evaluation_server import server as evaluation_server

__all__ = [
    "news_server",
    "odds_server",
    "tennis_server",
    "basketball_server",
    "alerts_server",
    "evaluation_server",
]
