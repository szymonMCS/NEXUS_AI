# mcp_servers/news_server.py
"""
MCP Server for news data.
Provides tools for fetching, validating, and analyzing news articles.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

from data.news.aggregator import NewsAggregator, get_match_news
from data.news.validator import NewsSourceValidator, validate_news_quality
from data.news.injury_extractor import InjuryExtractor, extract_match_injuries


# Create MCP server instance
server = Server("nexus-news-server")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available news tools."""
    return [
        Tool(
            name="search_match_news",
            description="Search for news articles about a specific match between two players/teams",
            inputSchema={
                "type": "object",
                "properties": {
                    "player1": {
                        "type": "string",
                        "description": "First player or team name"
                    },
                    "player2": {
                        "type": "string",
                        "description": "Second player or team name"
                    },
                    "sport": {
                        "type": "string",
                        "description": "Sport type (tennis, basketball, etc.)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 20
                    }
                },
                "required": ["player1", "player2", "sport"]
            }
        ),
        Tool(
            name="validate_news_sources",
            description="Validate news articles and assess their reliability",
            inputSchema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "description": "List of article objects to validate",
                        "items": {"type": "object"}
                    },
                    "player1": {
                        "type": "string",
                        "description": "First player name"
                    },
                    "player2": {
                        "type": "string",
                        "description": "Second player name"
                    }
                },
                "required": ["articles", "player1", "player2"]
            }
        ),
        Tool(
            name="extract_injuries",
            description="Extract injury information from news articles using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "description": "List of article objects",
                        "items": {"type": "object"}
                    },
                    "player1": {
                        "type": "string",
                        "description": "First player name"
                    },
                    "player2": {
                        "type": "string",
                        "description": "Second player name"
                    }
                },
                "required": ["articles", "player1", "player2"]
            }
        ),
        Tool(
            name="get_news_summary",
            description="Get a comprehensive news summary for a match including validation and injuries",
            inputSchema={
                "type": "object",
                "properties": {
                    "player1": {
                        "type": "string",
                        "description": "First player or team name"
                    },
                    "player2": {
                        "type": "string",
                        "description": "Second player or team name"
                    },
                    "sport": {
                        "type": "string",
                        "description": "Sport type"
                    }
                },
                "required": ["player1", "player2", "sport"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a news tool."""

    if name == "search_match_news":
        articles = await get_match_news(
            home_player=arguments["player1"],
            away_player=arguments["player2"],
            sport=arguments["sport"],
            max_results=arguments.get("max_results", 20)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "count": len(articles),
                "articles": articles
            }, indent=2, default=str)
        )]

    elif name == "validate_news_sources":
        validation = validate_news_quality(
            articles=arguments["articles"],
            player1=arguments["player1"],
            player2=arguments["player2"]
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "validation": validation
            }, indent=2, default=str)
        )]

    elif name == "extract_injuries":
        injuries = extract_match_injuries(
            articles=arguments["articles"],
            player1=arguments["player1"],
            player2=arguments["player2"]
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "injuries": injuries
            }, indent=2, default=str)
        )]

    elif name == "get_news_summary":
        # Comprehensive news analysis
        player1 = arguments["player1"]
        player2 = arguments["player2"]
        sport = arguments["sport"]

        # Fetch news
        articles = await get_match_news(player1, player2, sport)

        # Validate
        validation = validate_news_quality(articles, player1, player2)

        # Extract injuries
        injuries = extract_match_injuries(articles, player1, player2)

        summary = {
            "status": "success",
            "match": f"{player1} vs {player2}",
            "sport": sport,
            "news_count": len(articles),
            "quality_score": validation.get("quality_score", 0),
            "is_valid": validation.get("valid", False),
            "tier1_sources": validation.get("tier1_sources", 0),
            "fresh_articles": validation.get("fresh_articles", 0),
            "injuries": injuries,
            "cross_validation": validation.get("cross_validation", {}),
            "top_articles": articles[:5] if articles else []
        }

        return [TextContent(
            type="text",
            text=json.dumps(summary, indent=2, default=str)
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
                server_name="nexus-news-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
