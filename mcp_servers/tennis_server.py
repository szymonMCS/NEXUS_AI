# mcp_servers/tennis_server.py
"""
MCP Server for tennis data.
Provides tools for fetching match schedules, player stats, rankings, and H2H records.
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
import json

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

from config.settings import settings
from data.tennis.api_tennis_client import (
    TennisAPIClient,
    get_tennis_match_data,
    get_tennis_rankings,
    get_upcoming_tennis_matches
)
from data.tennis.sofascore_scraper import (
    SofascoreTennisScraper,
    scrape_tennis_match_data,
    scrape_upcoming_tennis_matches
)


# Create MCP server instance
server = Server("nexus-tennis-server")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tennis tools."""
    return [
        Tool(
            name="get_tennis_match",
            description="Get detailed match information for two tennis players",
            inputSchema={
                "type": "object",
                "properties": {
                    "player1": {
                        "type": "string",
                        "description": "First player name"
                    },
                    "player2": {
                        "type": "string",
                        "description": "Second player name"
                    },
                    "date": {
                        "type": "string",
                        "description": "Match date (YYYY-MM-DD), defaults to today"
                    }
                },
                "required": ["player1", "player2"]
            }
        ),
        Tool(
            name="get_tennis_rankings",
            description="Get current ATP or WTA rankings",
            inputSchema={
                "type": "object",
                "properties": {
                    "tour": {
                        "type": "string",
                        "description": "Tour type: atp or wta",
                        "enum": ["atp", "wta"],
                        "default": "atp"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_upcoming_matches",
            description="Get all upcoming tennis matches for a date",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date (YYYY-MM-DD), defaults to today"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_player_stats",
            description="Get detailed statistics for a tennis player",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name to search"
                    }
                },
                "required": ["player_name"]
            }
        ),
        Tool(
            name="get_head_to_head",
            description="Get head-to-head record between two players",
            inputSchema={
                "type": "object",
                "properties": {
                    "player1": {
                        "type": "string",
                        "description": "First player name"
                    },
                    "player2": {
                        "type": "string",
                        "description": "Second player name"
                    }
                },
                "required": ["player1", "player2"]
            }
        ),
        Tool(
            name="get_player_form",
            description="Get recent match results for a player",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name"
                    },
                    "num_matches": {
                        "type": "integer",
                        "description": "Number of recent matches",
                        "default": 10
                    }
                },
                "required": ["player_name"]
            }
        ),
    ]


async def _get_data_source():
    """Get appropriate data source based on settings."""
    if settings.is_pro_mode and settings.API_TENNIS_KEY:
        return "api"
    return "scraper"


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tennis tool."""

    source = await _get_data_source()

    if name == "get_tennis_match":
        player1 = arguments["player1"]
        player2 = arguments["player2"]
        date = arguments.get("date")

        if source == "api":
            match_data = await get_tennis_match_data(player1, player2, date)
        else:
            match_data = await scrape_tennis_match_data(player1, player2, date)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if match_data else "not_found",
                "source": source,
                "match": match_data
            }, indent=2, default=str)
        )]

    elif name == "get_tennis_rankings":
        tour = arguments.get("tour", "atp")

        if source == "api":
            rankings = await get_tennis_rankings(tour)
        else:
            # Sofascore doesn't have direct ranking endpoint, return limited data
            rankings = []

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "source": source,
                "tour": tour,
                "count": len(rankings),
                "rankings": rankings[:50]  # Top 50
            }, indent=2, default=str)
        )]

    elif name == "get_upcoming_matches":
        date = arguments.get("date", datetime.now().strftime("%Y-%m-%d"))

        if source == "api":
            matches = await get_upcoming_tennis_matches(date)
        else:
            matches = await scrape_upcoming_tennis_matches(date)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "source": source,
                "date": date,
                "count": len(matches),
                "matches": matches
            }, indent=2, default=str)
        )]

    elif name == "get_player_stats":
        player_name = arguments["player_name"]

        if source == "api":
            async with TennisAPIClient() as client:
                # Search for player first, then get stats
                # Simplified - in production would need player ID lookup
                stats = None
        else:
            async with SofascoreTennisScraper() as scraper:
                player = await scraper.search_player(player_name)
                if player:
                    stats = await scraper.get_player_stats(player.get("id"))
                else:
                    stats = None

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if stats else "not_found",
                "source": source,
                "player": player_name,
                "stats": stats
            }, indent=2, default=str)
        )]

    elif name == "get_head_to_head":
        player1 = arguments["player1"]
        player2 = arguments["player2"]

        # Get match data which includes H2H
        if source == "api":
            match_data = await get_tennis_match_data(player1, player2)
        else:
            match_data = await scrape_tennis_match_data(player1, player2)

        h2h = match_data.get("h2h") if match_data else None

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if h2h else "not_found",
                "source": source,
                "players": f"{player1} vs {player2}",
                "h2h": h2h
            }, indent=2, default=str)
        )]

    elif name == "get_player_form":
        player_name = arguments["player_name"]
        num_matches = arguments.get("num_matches", 10)

        form = []
        if source == "scraper":
            async with SofascoreTennisScraper() as scraper:
                player = await scraper.search_player(player_name)
                if player:
                    form = await scraper.get_player_form(player.get("id"), num_matches)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "source": source,
                "player": player_name,
                "recent_matches": form
            }, indent=2, default=str)
        )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Unknown tool: {name}"
            })
        )]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="nexus-tennis-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
