# mcp_servers/basketball_server.py
"""
MCP Server for basketball data.
Provides tools for fetching match schedules, team stats, standings, and H2H records.
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
import json

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

from config.settings import settings
from data.basketball.bets_api_client import (
    BetsAPIBasketballClient,
    get_basketball_match_data,
    get_basketball_standings,
    get_upcoming_basketball_matches
)
from data.basketball.euroleague_scraper import (
    SofascoreBasketballScraper,
    scrape_basketball_match_data,
    scrape_upcoming_basketball_matches,
    scrape_basketball_standings
)


# Create MCP server instance
server = Server("nexus-basketball-server")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available basketball tools."""
    return [
        Tool(
            name="get_basketball_match",
            description="Get detailed match information for two basketball teams",
            inputSchema={
                "type": "object",
                "properties": {
                    "home_team": {
                        "type": "string",
                        "description": "Home team name"
                    },
                    "away_team": {
                        "type": "string",
                        "description": "Away team name"
                    },
                    "league": {
                        "type": "string",
                        "description": "League (nba, euroleague, etc.)",
                        "default": "nba"
                    },
                    "date": {
                        "type": "string",
                        "description": "Match date (YYYY-MM-DD), defaults to today"
                    }
                },
                "required": ["home_team", "away_team"]
            }
        ),
        Tool(
            name="get_standings",
            description="Get league standings",
            inputSchema={
                "type": "object",
                "properties": {
                    "league": {
                        "type": "string",
                        "description": "League (nba, euroleague)",
                        "default": "nba"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_upcoming_matches",
            description="Get all upcoming basketball matches for a date",
            inputSchema={
                "type": "object",
                "properties": {
                    "league": {
                        "type": "string",
                        "description": "League filter (nba, euroleague, all)",
                        "default": "nba"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date (YYYY-MM-DD), defaults to today"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_team_stats",
            description="Get detailed statistics for a basketball team",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "Team name to search"
                    },
                    "league": {
                        "type": "string",
                        "description": "League",
                        "default": "nba"
                    }
                },
                "required": ["team_name"]
            }
        ),
        Tool(
            name="get_head_to_head",
            description="Get head-to-head record between two teams",
            inputSchema={
                "type": "object",
                "properties": {
                    "team1": {
                        "type": "string",
                        "description": "First team name"
                    },
                    "team2": {
                        "type": "string",
                        "description": "Second team name"
                    },
                    "league": {
                        "type": "string",
                        "description": "League",
                        "default": "nba"
                    }
                },
                "required": ["team1", "team2"]
            }
        ),
        Tool(
            name="get_team_form",
            description="Get recent match results for a team",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "Team name"
                    },
                    "num_matches": {
                        "type": "integer",
                        "description": "Number of recent matches",
                        "default": 10
                    }
                },
                "required": ["team_name"]
            }
        ),
    ]


async def _get_data_source():
    """Get appropriate data source based on settings."""
    if settings.is_pro_mode and settings.BETS_API_KEY:
        return "api"
    return "scraper"


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a basketball tool."""

    source = await _get_data_source()

    if name == "get_basketball_match":
        home_team = arguments["home_team"]
        away_team = arguments["away_team"]
        league = arguments.get("league", "nba")
        date = arguments.get("date")

        if source == "api":
            match_data = await get_basketball_match_data(home_team, away_team, league, date)
        else:
            match_data = await scrape_basketball_match_data(home_team, away_team, league, date)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if match_data else "not_found",
                "source": source,
                "match": match_data
            }, indent=2, default=str)
        )]

    elif name == "get_standings":
        league = arguments.get("league", "nba")

        if source == "api":
            standings = await get_basketball_standings(league)
        else:
            # Tournament IDs: NBA=132, EuroLeague=7166
            tournament_id = 132 if league == "nba" else 7166
            standings = await scrape_basketball_standings(tournament_id)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "source": source,
                "league": league,
                "standings": standings
            }, indent=2, default=str)
        )]

    elif name == "get_upcoming_matches":
        league = arguments.get("league", "nba")
        date = arguments.get("date", datetime.now().strftime("%Y-%m-%d"))

        if source == "api":
            matches = await get_upcoming_basketball_matches(league, date)
        else:
            matches = await scrape_upcoming_basketball_matches(league, date)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "source": source,
                "league": league,
                "date": date,
                "count": len(matches),
                "matches": matches
            }, indent=2, default=str)
        )]

    elif name == "get_team_stats":
        team_name = arguments["team_name"]
        league = arguments.get("league", "nba")

        stats = None
        if source == "scraper":
            async with SofascoreBasketballScraper() as scraper:
                team = await scraper.search_team(team_name)
                if team:
                    stats = await scraper.get_team_stats(team.get("id"))

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if stats else "not_found",
                "source": source,
                "team": team_name,
                "stats": stats
            }, indent=2, default=str)
        )]

    elif name == "get_head_to_head":
        team1 = arguments["team1"]
        team2 = arguments["team2"]
        league = arguments.get("league", "nba")

        # Get match data which includes H2H
        if source == "api":
            match_data = await get_basketball_match_data(team1, team2, league)
        else:
            match_data = await scrape_basketball_match_data(team1, team2, league)

        h2h = match_data.get("h2h") if match_data else None

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if h2h else "not_found",
                "source": source,
                "teams": f"{team1} vs {team2}",
                "h2h": h2h
            }, indent=2, default=str)
        )]

    elif name == "get_team_form":
        team_name = arguments["team_name"]
        num_matches = arguments.get("num_matches", 10)

        form = []
        if source == "scraper":
            async with SofascoreBasketballScraper() as scraper:
                team = await scraper.search_team(team_name)
                if team:
                    form = await scraper.get_team_form(team.get("id"), num_matches)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "source": source,
                "team": team_name,
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
                server_name="nexus-basketball-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
