# mcp_servers/alerts_server.py
"""
MCP Server for alerts and notifications.
Provides tools for creating, managing, and sending alerts about matches and bets.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

from config.settings import settings


class AlertType(str, Enum):
    """Alert types"""
    VALUE_BET_FOUND = "value_bet_found"
    ODDS_CHANGE = "odds_change"
    INJURY_UPDATE = "injury_update"
    MATCH_STARTING = "match_starting"
    BET_RESULT = "bet_result"
    DATA_QUALITY_WARNING = "data_quality_warning"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"


class AlertPriority(str, Enum):
    """Alert priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# In-memory alert storage (would be Redis in production)
alerts_store: List[Dict] = []


# Create MCP server instance
server = Server("nexus-alerts-server")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available alert tools."""
    return [
        Tool(
            name="create_alert",
            description="Create a new alert",
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_type": {
                        "type": "string",
                        "description": "Type of alert",
                        "enum": [t.value for t in AlertType]
                    },
                    "priority": {
                        "type": "string",
                        "description": "Alert priority",
                        "enum": [p.value for p in AlertPriority],
                        "default": "medium"
                    },
                    "title": {
                        "type": "string",
                        "description": "Alert title"
                    },
                    "message": {
                        "type": "string",
                        "description": "Alert message"
                    },
                    "match_id": {
                        "type": "string",
                        "description": "Related match ID (optional)"
                    },
                    "data": {
                        "type": "object",
                        "description": "Additional data (optional)"
                    }
                },
                "required": ["alert_type", "title", "message"]
            }
        ),
        Tool(
            name="get_alerts",
            description="Get alerts with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_type": {
                        "type": "string",
                        "description": "Filter by alert type",
                        "enum": [t.value for t in AlertType]
                    },
                    "priority": {
                        "type": "string",
                        "description": "Filter by priority",
                        "enum": [p.value for p in AlertPriority]
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only return unread alerts",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of alerts to return",
                        "default": 50
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="mark_alert_read",
            description="Mark an alert as read",
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_id": {
                        "type": "string",
                        "description": "Alert ID to mark as read"
                    }
                },
                "required": ["alert_id"]
            }
        ),
        Tool(
            name="delete_alert",
            description="Delete an alert",
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_id": {
                        "type": "string",
                        "description": "Alert ID to delete"
                    }
                },
                "required": ["alert_id"]
            }
        ),
        Tool(
            name="create_value_bet_alert",
            description="Create a value bet found alert with bet details",
            inputSchema={
                "type": "object",
                "properties": {
                    "match": {
                        "type": "string",
                        "description": "Match description (e.g., 'Player1 vs Player2')"
                    },
                    "selection": {
                        "type": "string",
                        "description": "Recommended selection (home/away)"
                    },
                    "odds": {
                        "type": "number",
                        "description": "Best available odds"
                    },
                    "edge": {
                        "type": "number",
                        "description": "Expected edge percentage"
                    },
                    "kelly_stake": {
                        "type": "number",
                        "description": "Recommended Kelly stake"
                    },
                    "bookmaker": {
                        "type": "string",
                        "description": "Bookmaker with best odds"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Prediction confidence"
                    }
                },
                "required": ["match", "selection", "odds", "edge"]
            }
        ),
        Tool(
            name="create_injury_alert",
            description="Create an injury update alert",
            inputSchema={
                "type": "object",
                "properties": {
                    "player": {
                        "type": "string",
                        "description": "Player name"
                    },
                    "injury_type": {
                        "type": "string",
                        "description": "Type of injury"
                    },
                    "status": {
                        "type": "string",
                        "description": "Injury status (out, doubtful, etc.)"
                    },
                    "match": {
                        "type": "string",
                        "description": "Affected match"
                    },
                    "impact_score": {
                        "type": "number",
                        "description": "Impact score on prediction"
                    }
                },
                "required": ["player", "status", "match"]
            }
        ),
        Tool(
            name="get_alert_summary",
            description="Get a summary of current alerts by type and priority",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="send_telegram_notification",
            description="Send an alert via Telegram (if configured)",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send"
                    },
                    "alert_id": {
                        "type": "string",
                        "description": "Alert ID to send (uses alert's message)"
                    }
                },
                "required": []
            }
        ),
    ]


def _generate_alert_id() -> str:
    """Generate unique alert ID."""
    import uuid
    return str(uuid.uuid4())[:8]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute an alert tool."""

    global alerts_store

    if name == "create_alert":
        alert = {
            "id": _generate_alert_id(),
            "type": arguments["alert_type"],
            "priority": arguments.get("priority", "medium"),
            "title": arguments["title"],
            "message": arguments["message"],
            "match_id": arguments.get("match_id"),
            "data": arguments.get("data", {}),
            "created_at": datetime.now().isoformat(),
            "read": False
        }

        alerts_store.append(alert)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "alert": alert
            }, indent=2)
        )]

    elif name == "get_alerts":
        filtered = alerts_store.copy()

        if arguments.get("alert_type"):
            filtered = [a for a in filtered if a["type"] == arguments["alert_type"]]

        if arguments.get("priority"):
            filtered = [a for a in filtered if a["priority"] == arguments["priority"]]

        if arguments.get("unread_only"):
            filtered = [a for a in filtered if not a["read"]]

        # Sort by created_at descending
        filtered.sort(key=lambda x: x["created_at"], reverse=True)

        limit = arguments.get("limit", 50)
        filtered = filtered[:limit]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "count": len(filtered),
                "alerts": filtered
            }, indent=2)
        )]

    elif name == "mark_alert_read":
        alert_id = arguments["alert_id"]

        for alert in alerts_store:
            if alert["id"] == alert_id:
                alert["read"] = True
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "alert": alert
                    }, indent=2)
                )]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Alert not found: {alert_id}"
            })
        )]

    elif name == "delete_alert":
        alert_id = arguments["alert_id"]

        for i, alert in enumerate(alerts_store):
            if alert["id"] == alert_id:
                deleted = alerts_store.pop(i)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "deleted_alert": deleted
                    }, indent=2)
                )]

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Alert not found: {alert_id}"
            })
        )]

    elif name == "create_value_bet_alert":
        match = arguments["match"]
        selection = arguments["selection"]
        odds = arguments["odds"]
        edge = arguments["edge"]

        alert = {
            "id": _generate_alert_id(),
            "type": AlertType.VALUE_BET_FOUND.value,
            "priority": "high" if edge > 5 else "medium",
            "title": f"Value Bet: {match}",
            "message": f"Recommended: {selection.upper()} @ {odds:.2f} (Edge: {edge:.1f}%)",
            "data": {
                "match": match,
                "selection": selection,
                "odds": odds,
                "edge": edge,
                "kelly_stake": arguments.get("kelly_stake"),
                "bookmaker": arguments.get("bookmaker"),
                "confidence": arguments.get("confidence")
            },
            "created_at": datetime.now().isoformat(),
            "read": False
        }

        alerts_store.append(alert)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "alert": alert
            }, indent=2)
        )]

    elif name == "create_injury_alert":
        player = arguments["player"]
        status = arguments["status"]
        match = arguments["match"]

        priority = "high" if status in ["out", "doubtful"] else "medium"

        alert = {
            "id": _generate_alert_id(),
            "type": AlertType.INJURY_UPDATE.value,
            "priority": priority,
            "title": f"Injury Alert: {player}",
            "message": f"{player} is {status} for {match}",
            "data": {
                "player": player,
                "injury_type": arguments.get("injury_type"),
                "status": status,
                "match": match,
                "impact_score": arguments.get("impact_score")
            },
            "created_at": datetime.now().isoformat(),
            "read": False
        }

        alerts_store.append(alert)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "alert": alert
            }, indent=2)
        )]

    elif name == "get_alert_summary":
        summary = {
            "total": len(alerts_store),
            "unread": sum(1 for a in alerts_store if not a["read"]),
            "by_type": {},
            "by_priority": {}
        }

        for alert in alerts_store:
            # Count by type
            alert_type = alert["type"]
            if alert_type not in summary["by_type"]:
                summary["by_type"][alert_type] = 0
            summary["by_type"][alert_type] += 1

            # Count by priority
            priority = alert["priority"]
            if priority not in summary["by_priority"]:
                summary["by_priority"][priority] = 0
            summary["by_priority"][priority] += 1

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "summary": summary
            }, indent=2)
        )]

    elif name == "send_telegram_notification":
        # Placeholder for Telegram integration
        message = arguments.get("message")

        if not message and arguments.get("alert_id"):
            for alert in alerts_store:
                if alert["id"] == arguments["alert_id"]:
                    message = f"[{alert['priority'].upper()}] {alert['title']}\n{alert['message']}"
                    break

        if not message:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "No message provided and alert not found"
                })
            )]

        # TODO: Implement actual Telegram sending
        # For now, just simulate success
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "message": "Telegram notification queued",
                "content": message,
                "note": "Telegram integration not yet implemented"
            }, indent=2)
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
                server_name="nexus-alerts-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
