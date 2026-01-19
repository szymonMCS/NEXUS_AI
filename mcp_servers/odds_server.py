# mcp_servers/odds_server.py
"""
MCP Server for betting odds data.
Provides tools for fetching, comparing, and analyzing odds from multiple bookmakers.
"""

import asyncio
from typing import List, Dict, Any
import json

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

from data.odds.odds_api_client import OddsAPIClient, get_odds_for_match
from data.odds.pl_scraper import scrape_polish_odds, find_match_odds
from data.odds.odds_merger import OddsMerger, get_merged_odds_analysis


# Create MCP server instance
server = Server("nexus-odds-server")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available odds tools."""
    return [
        Tool(
            name="get_match_odds",
            description="Get odds for a specific match from all available bookmakers",
            inputSchema={
                "type": "object",
                "properties": {
                    "sport": {
                        "type": "string",
                        "description": "Sport type (tennis, basketball, etc.)"
                    },
                    "home_team": {
                        "type": "string",
                        "description": "Home team or player name"
                    },
                    "away_team": {
                        "type": "string",
                        "description": "Away team or player name"
                    }
                },
                "required": ["sport", "home_team", "away_team"]
            }
        ),
        Tool(
            name="find_best_odds",
            description="Find the best odds available for a selection across all bookmakers",
            inputSchema={
                "type": "object",
                "properties": {
                    "odds_list": {
                        "type": "array",
                        "description": "List of odds objects from different bookmakers",
                        "items": {"type": "object"}
                    },
                    "selection": {
                        "type": "string",
                        "description": "Selection to find best odds for (home or away)",
                        "enum": ["home", "away"]
                    }
                },
                "required": ["odds_list", "selection"]
            }
        ),
        Tool(
            name="calculate_value_bet",
            description="Calculate if a bet offers value based on predicted probability",
            inputSchema={
                "type": "object",
                "properties": {
                    "predicted_probability": {
                        "type": "number",
                        "description": "AI predicted win probability (0.0-1.0)"
                    },
                    "best_odds": {
                        "type": "number",
                        "description": "Best available decimal odds"
                    },
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge required (default 0.03 = 3%)",
                        "default": 0.03
                    }
                },
                "required": ["predicted_probability", "best_odds"]
            }
        ),
        Tool(
            name="calculate_kelly_stake",
            description="Calculate optimal stake using Kelly Criterion",
            inputSchema={
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number",
                        "description": "Win probability (0.0-1.0)"
                    },
                    "odds": {
                        "type": "number",
                        "description": "Decimal odds"
                    },
                    "bankroll": {
                        "type": "number",
                        "description": "Current bankroll amount"
                    },
                    "kelly_fraction": {
                        "type": "number",
                        "description": "Fractional Kelly (default 0.25 = quarter Kelly)",
                        "default": 0.25
                    }
                },
                "required": ["probability", "odds", "bankroll"]
            }
        ),
        Tool(
            name="detect_arbitrage",
            description="Detect arbitrage opportunities across bookmakers",
            inputSchema={
                "type": "object",
                "properties": {
                    "odds_list": {
                        "type": "array",
                        "description": "List of odds from multiple bookmakers",
                        "items": {"type": "object"}
                    }
                },
                "required": ["odds_list"]
            }
        ),
        Tool(
            name="get_complete_odds_analysis",
            description="Get comprehensive odds analysis including value bets and Kelly stakes",
            inputSchema={
                "type": "object",
                "properties": {
                    "sport": {
                        "type": "string",
                        "description": "Sport type"
                    },
                    "home_team": {
                        "type": "string",
                        "description": "Home team/player"
                    },
                    "away_team": {
                        "type": "string",
                        "description": "Away team/player"
                    },
                    "predicted_home_prob": {
                        "type": "number",
                        "description": "Predicted home win probability"
                    },
                    "predicted_away_prob": {
                        "type": "number",
                        "description": "Predicted away win probability"
                    },
                    "bankroll": {
                        "type": "number",
                        "description": "Current bankroll",
                        "default": 1000.0
                    }
                },
                "required": ["sport", "home_team", "away_team", "predicted_home_prob", "predicted_away_prob"]
            }
        ),
        Tool(
            name="scrape_polish_bookmakers",
            description="Scrape odds from Polish bookmakers (Fortuna, STS, Betclic)",
            inputSchema={
                "type": "object",
                "properties": {
                    "sport": {
                        "type": "string",
                        "description": "Sport type"
                    }
                },
                "required": ["sport"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute an odds tool."""

    merger = OddsMerger()

    if name == "get_match_odds":
        odds = await merger.get_all_odds(
            sport=arguments["sport"],
            home_team=arguments["home_team"],
            away_team=arguments["away_team"]
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "count": len(odds),
                "odds": odds
            }, indent=2, default=str)
        )]

    elif name == "find_best_odds":
        best = merger.find_best_odds(
            odds_list=arguments["odds_list"],
            selection=arguments["selection"]
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "best_odds": best
            }, indent=2, default=str)
        )]

    elif name == "calculate_value_bet":
        value = merger.calculate_value_bet(
            predicted_probability=arguments["predicted_probability"],
            best_odds=arguments["best_odds"],
            min_edge=arguments.get("min_edge", 0.03)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "value_bet": value
            }, indent=2, default=str)
        )]

    elif name == "calculate_kelly_stake":
        stake = merger.kelly_criterion(
            probability=arguments["probability"],
            odds=arguments["odds"],
            bankroll=arguments["bankroll"],
            kelly_fraction=arguments.get("kelly_fraction", 0.25)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "recommended_stake": stake,
                "bankroll": arguments["bankroll"],
                "stake_percentage": (stake / arguments["bankroll"]) * 100 if arguments["bankroll"] > 0 else 0
            }, indent=2, default=str)
        )]

    elif name == "detect_arbitrage":
        arbitrage = merger.detect_arbitrage(arguments["odds_list"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "arbitrage": arbitrage
            }, indent=2, default=str)
        )]

    elif name == "get_complete_odds_analysis":
        analysis = await get_merged_odds_analysis(
            sport=arguments["sport"],
            home_team=arguments["home_team"],
            away_team=arguments["away_team"],
            predicted_home_prob=arguments["predicted_home_prob"],
            predicted_away_prob=arguments["predicted_away_prob"],
            bankroll=arguments.get("bankroll", 1000.0)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "analysis": analysis
            }, indent=2, default=str)
        )]

    elif name == "scrape_polish_bookmakers":
        odds = await scrape_polish_odds(arguments["sport"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "count": len(odds),
                "odds": odds
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
                server_name="nexus-odds-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
