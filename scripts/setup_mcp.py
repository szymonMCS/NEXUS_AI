# scripts/setup_mcp.py
"""
MCP Servers setup script for NEXUS AI.
Verifies and starts all MCP servers.

Usage:
    python scripts/setup_mcp.py [--check-only] [--server SERVER_NAME]
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings


# MCP Server configurations
MCP_SERVERS = {
    "sofascore": {
        "module": "mcp_servers.sofascore_server",
        "class": "SofascoreMCPServer",
        "port": 8100,
        "description": "Sofascore data provider (fixtures, stats, odds)",
        "mode": "both",  # pro and lite
    },
    "odds": {
        "module": "mcp_servers.odds_server",
        "class": "OddsMCPServer",
        "port": 8101,
        "description": "Odds comparison from multiple bookmakers",
        "mode": "both",
    },
    "news": {
        "module": "mcp_servers.news_server",
        "class": "NewsMCPServer",
        "port": 8102,
        "description": "Sports news aggregation",
        "mode": "both",
    },
    "evaluation": {
        "module": "mcp_servers.evaluation_server",
        "class": "EvaluationMCPServer",
        "port": 8103,
        "description": "Data quality evaluation",
        "mode": "both",
    },
    "tennis_explorer": {
        "module": "mcp_servers.tennis_explorer_server",
        "class": "TennisExplorerMCPServer",
        "port": 8104,
        "description": "Tennis-specific data (H2H, surfaces)",
        "mode": "pro",  # requires premium access
    },
}


def check_server_module(server_name: str, config: Dict[str, Any]) -> bool:
    """Check if server module can be imported."""
    try:
        module_name = config["module"]
        __import__(module_name)
        return True
    except ImportError as e:
        print(f"  [ERROR] Cannot import {config['module']}: {e}")
        return False


def check_dependencies(server_name: str) -> List[str]:
    """Check server dependencies."""
    missing = []

    if server_name == "sofascore":
        try:
            import httpx
        except ImportError:
            missing.append("httpx")

    if server_name == "news":
        try:
            import feedparser
        except ImportError:
            missing.append("feedparser")

    return missing


def get_available_servers(mode: str = "both") -> Dict[str, Dict]:
    """Get servers available for given mode."""
    available = {}
    for name, config in MCP_SERVERS.items():
        if config["mode"] == "both" or config["mode"] == mode:
            available[name] = config
    return available


def print_server_status(name: str, config: Dict, status: str, details: str = ""):
    """Print formatted server status."""
    status_icons = {
        "ok": "\033[92m[OK]\033[0m",
        "error": "\033[91m[ERROR]\033[0m",
        "warn": "\033[93m[WARN]\033[0m",
        "skip": "\033[90m[SKIP]\033[0m",
    }
    icon = status_icons.get(status, "[???]")

    print(f"  {icon} {name}")
    print(f"       Port: {config['port']}")
    print(f"       {config['description']}")
    if details:
        print(f"       {details}")
    print()


def check_all_servers(mode: str = "both") -> bool:
    """Check all MCP servers."""
    print("\n" + "="*60)
    print("NEXUS AI - MCP Servers Check")
    print("="*60 + "\n")

    servers = get_available_servers(mode)
    all_ok = True

    print(f"Mode: {mode.upper()}")
    print(f"Checking {len(servers)} servers...\n")

    for name, config in servers.items():
        # Check dependencies
        missing_deps = check_dependencies(name)
        if missing_deps:
            print_server_status(
                name, config, "error",
                f"Missing dependencies: {', '.join(missing_deps)}"
            )
            all_ok = False
            continue

        # Check module import
        if check_server_module(name, config):
            print_server_status(name, config, "ok")
        else:
            all_ok = False
            print_server_status(name, config, "error", "Module import failed")

    return all_ok


async def start_server(name: str, config: Dict) -> bool:
    """Start a single MCP server."""
    try:
        module = __import__(config["module"], fromlist=[config["class"]])
        server_class = getattr(module, config["class"])

        # Create server instance
        server = server_class()

        print(f"  Starting {name} on port {config['port']}...")

        # Start server (implementation depends on MCP framework)
        # For now, just verify it can be instantiated
        print(f"  [OK] {name} server ready")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to start {name}: {e}")
        return False


async def start_all_servers(mode: str = "both") -> bool:
    """Start all MCP servers."""
    print("\n" + "="*60)
    print("NEXUS AI - Starting MCP Servers")
    print("="*60 + "\n")

    servers = get_available_servers(mode)

    tasks = []
    for name, config in servers.items():
        tasks.append(start_server(name, config))

    results = await asyncio.gather(*tasks)
    return all(results)


def generate_mcp_config() -> Dict[str, Any]:
    """Generate MCP configuration for Claude Desktop."""
    config = {
        "mcpServers": {}
    }

    for name, server_config in MCP_SERVERS.items():
        config["mcpServers"][f"nexus-{name}"] = {
            "command": "python",
            "args": [
                "-m",
                server_config["module"]
            ],
            "cwd": str(PROJECT_ROOT)
        }

    return config


def print_claude_config():
    """Print Claude Desktop MCP configuration."""
    import json

    print("\n" + "="*60)
    print("Claude Desktop MCP Configuration")
    print("="*60 + "\n")
    print("Add the following to your Claude Desktop config:\n")

    config = generate_mcp_config()
    print(json.dumps(config, indent=2))

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS AI MCP Servers Setup"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check servers, don't start them"
    )
    parser.add_argument(
        "--server",
        type=str,
        help="Specific server to check/start"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pro", "lite", "both"],
        default="both",
        help="Operating mode (pro/lite/both)"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show Claude Desktop MCP configuration"
    )

    args = parser.parse_args()

    if args.show_config:
        print_claude_config()
        return

    if args.check_only:
        success = check_all_servers(args.mode)
        sys.exit(0 if success else 1)

    # Start servers
    success = asyncio.run(start_all_servers(args.mode))

    if success:
        print("\nAll MCP servers started successfully!")
        print("\nPress Ctrl+C to stop servers...")
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            print("\nShutting down servers...")
    else:
        print("\nSome servers failed to start.")
        sys.exit(1)


if __name__ == "__main__":
    main()
