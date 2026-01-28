"""
Test API Tier System.

Checks all configured APIs across all tiers and reports status.
Run: python scripts/test_api_tiers.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def main():
    """Test all API tiers."""
    from data.apis import APITierManager, check_api_status

    print("=" * 60)
    print("NEXUS API Tier System Test")
    print("=" * 60)
    print()

    # Check environment variables
    print("Environment Variables Status:")
    print("-" * 40)

    env_vars = {
        # Enterprise
        "SPORTRADAR_API_KEY": "Sportradar (Enterprise)",
        "STATS_PERFORM_API_KEY": "Stats Perform (Enterprise)",
        # Professional
        "GENIUS_SPORTS_API_KEY": "Genius Sports (Professional)",
        "LSPORTS_API_KEY": "LSports (Professional)",
        # Developer
        "SPORTSDATAIO_API_KEY": "SportsDataIO (Developer)",
        "API_FOOTBALL_PRO_KEY": "API-Football Pro (Developer)",
        # Free
        "ODDS_API_KEY": "The Odds API (Free)",
        "X-Auth-Token": "Football-Data.org (Free)",
        "x-apisports-key": "API-Sports (Free)",
        "PANDASCORE_API_KEY": "PandaScore (Free)",
        "NEWSAPI_KEY": "NewsAPI (Free)",
    }

    for var, name in env_vars.items():
        value = os.getenv(var)
        if value:
            # Mask the key for security
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  [OK] {name}: {masked}")
        else:
            print(f"  [--] {name}: Not configured")

    print()
    print("=" * 60)
    print("API Tier Status Report")
    print("=" * 60)
    print()

    # Get full status report
    report = await check_api_status()
    print(report)

    # Test specific functionality
    print()
    print("=" * 60)
    print("Functional Tests")
    print("=" * 60)
    print()

    async with APITierManager() as manager:
        # Test football matches
        print("Testing football matches (auto-tier)...")
        result = await manager.get_football_matches("PL")
        if result.success:
            data = result.data
            count = len(data) if isinstance(data, list) else "N/A"
            print(f"  [OK] Source: {result.source}, Found: {count} matches")
        else:
            print(f"  [ERROR] {result.error}")

        # Test odds
        print()
        print("Testing odds (auto-tier)...")
        result = await manager.get_odds("basketball_nba")
        if result.success:
            data = result.data
            count = len(data) if isinstance(data, list) else "N/A"
            print(f"  [OK] Source: {result.source}, Found: {count} events")
        else:
            print(f"  [ERROR] {result.error}")

        # Test F1 (always free)
        print()
        print("Testing F1 sessions (OpenF1 - free)...")
        result = await manager.get_f1_sessions(year=2024, session_type="Race")
        if result.success:
            data = result.data
            count = len(data) if isinstance(data, list) else "N/A"
            print(f"  [OK] Source: {result.source}, Found: {count} sessions")
        else:
            print(f"  [ERROR] {result.error}")

        # Test MLB (free)
        print()
        print("Testing MLB schedule (MLB Stats - free)...")
        result = await manager.get_mlb_schedule()
        if result.success:
            data = result.data
            count = len(data) if isinstance(data, list) else "N/A"
            print(f"  [OK] Source: {result.source}, Found: {count} dates")
        else:
            print(f"  [ERROR] {result.error}")

        # Test eSports
        print()
        print("Testing eSports matches (PandaScore)...")
        result = await manager.get_esports_matches("lol", "upcoming")
        if result.success:
            data = result.data
            count = len(data) if isinstance(data, list) else "N/A"
            print(f"  [OK] Source: {result.source}, Found: {count} matches")
        else:
            print(f"  [ERROR] {result.error}")

    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)
    print()
    print("To add premium APIs, configure these keys in .env:")
    print("  - SPORTRADAR_API_KEY (enterprise)")
    print("  - SPORTSDATAIO_API_KEY (developer-friendly)")
    print("  - LSPORTS_API_KEY (professional)")
    print()
    print("When keys are added, the system will automatically")
    print("use the best available data source.")


if __name__ == "__main__":
    asyncio.run(main())
