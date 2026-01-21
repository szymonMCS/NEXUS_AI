# main.py
#!/usr/bin/env python3
"""
NEXUS AI v2.2.0 - Main Entry Point

A comprehensive sports prediction system that combines:
- AI-powered analysis (Claude)
- Statistical models (Tennis, Basketball, Greyhound, Handball, Table Tennis)
- Value betting algorithms
- Real-time data collection
- Web-based UI

Usage:
    python main.py                    # Start API server
    python main.py --help             # Show all options
    python main.py --dev              # Development mode
    python main.py --analyze tennis   # Run single analysis
    python main.py --analysis-loop    # Continuous analysis
"""

import argparse
import asyncio
import sys
import uvicorn
from pathlib import Path

from config.settings import settings
from nexus import print_banner, print_step


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NEXUS AI v2.2.0 - Sports Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                       # Start API server on port 8000
    python main.py --port 8080           # Start on port 8080
    python main.py --dev                 # Development mode with reload
    python main.py --analyze tennis      # Run tennis analysis
    python main.py --analyze basketball --date 2026-01-21
    python main.py --analysis-loop       # Continuous analysis mode
    python main.py --version             # Show version
        """
    )
    
    # Server options
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for API server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for API server (default: 0.0.0.0)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers (default: 1)")
    parser.add_argument("--dev", action="store_true",
                        help="Development mode with auto-reload")
    parser.add_argument("--ssl", action="store_true",
                        help="Enable SSL/TLS (requires cert files)")
    
    # Analysis options
    parser.add_argument("--analyze", type=str, metavar="SPORT",
                        help="Run single analysis for sport (tennis, basketball, etc.)")
    parser.add_argument("--date", type=str, metavar="DATE",
                        help="Date for analysis (YYYY-MM-DD, default: today)")
    parser.add_argument("--min-quality", type=float, metavar="N",
                        help="Minimum quality threshold (0-100, default: 45)")
    parser.add_argument("--top", type=int, metavar="N", default=3,
                        help="Number of top bets to return (default: 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("--format", type=str, choices=["text", "json", "md"],
                        default="text", help="Output format (default: text)")
    
    # Continuous analysis
    parser.add_argument("--analysis-loop", action="store_true",
                        help="Run continuous analysis loop")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Analysis interval in seconds (default: 3600)")
    
    # Other options
    parser.add_argument("--version", action="store_true",
                        help="Show version information")
    parser.add_argument("--check", action="store_true",
                        help="Check system configuration")
    parser.add_argument("--init-db", action="store_true",
                        help="Initialize database")
    
    return parser.parse_args()


def run_server(args):
    """Run the FastAPI server."""
    from api.main import app
    
    print_banner()
    print_step("Starting NEXUS AI API Server...")
    print(f"  Version: 2.2.0")
    print(f"  Port: {args.port}")
    print(f"  Host: {args.host}")
    print(f"  Mode: {'Development' if args.dev else 'Production'}")
    print()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        reload=args.dev,
        workers=1 if args.dev else args.workers,
        log_level="info" if args.dev else "warning",
    )
    
    server = uvicorn.Server(config)
    server.run()


async def run_single_analysis(args):
    """Run a single analysis."""
    from nexus import run_analysis as nexus_run_analysis
    
    print_banner()
    print_step(f"Running analysis for {args.analyze}...")
    
    result = await nexus_run_analysis(
        sport=args.analyze,
        date=args.date,
        min_quality=args.min_quality or 45,
        top_n=args.top,
        quiet=args.quiet,
        output_format=args.format,
    )
    
    return result


async def run_analysis_loop(args):
    """Run continuous analysis loop."""
    from nexus import run_analysis as nexus_run_analysis
    import aioschedule as schedule
    
    print_banner()
    print_step("Starting Analysis Loop...")
    print(f"  Interval: {args.interval} seconds")
    print()
    
    # Schedule daily analysis for each sport
    sports = ["tennis", "basketball", "greyhound", "handball", "table_tennis"]
    
    for sport in sports:
        schedule.every().day.at("08:00").do(
            lambda s=sport: nexus_run_analysis(sport=s, quiet=args.quiet)
        )
    
    print("Scheduled daily analysis for:")
    for sport in sports:
        print(f"  - {sport}")
    print()
    
    while True:
        await schedule.run_pending()
        await asyncio.sleep(10)


def check_system():
    """Check system configuration."""
    print_banner()
    print_step("Checking System Configuration...")
    print()
    
    checks = []
    
    # Check API keys
    api_keys = {
        "ANTHROPIC_API_KEY": settings.ANTHROPIC_API_KEY,
        "BRAVE_API_KEY": settings.BRAVE_API_KEY,
        "SERPER_API_KEY": settings.SERPER_API_KEY,
        "ODDS_API_KEY": settings.ODDS_API_KEY,
    }
    
    print("API Keys:")
    for key, value in api_keys.items():
        configured = bool(value)
        status = "✓ Configured" if configured else "✗ Not set"
        print(f"  {key}: {status}")
        checks.append(configured)
    print()
    
    # Check database
    print("Database:")
    try:
        from database.db import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("  PostgreSQL: ✓ Connected")
    except Exception as e:
        print(f"  PostgreSQL: ✗ Failed ({e})")
        checks.append(False)
    print()
    
    # Check Redis
    print("Cache:")
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        print("  Redis: ✓ Connected")
    except Exception as e:
        print(f"  Redis: ✗ Failed ({e})")
        checks.append(False)
    print()
    
    # Summary
    all_passed = all(checks)
    print("=" * 40)
    if all_passed:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed - review configuration")
    print("=" * 40)
    
    return all_passed


def main():
    """Main entry point."""
    args = parse_args()
    
    # Show version
    if args.version:
        print("NEXUS AI v2.2.0")
        print("Sports Prediction System")
        print()
        print("Supported sports:")
        print("  - Tennis (ATP, WTA)")
        print("  - Basketball (NBA, EuroLeague)")
        print("  - Greyhound Racing")
        print("  - Handball")
        print("  - Table Tennis")
        return 0
    
    # Check system
    if args.check:
        success = check_system()
        return 0 if success else 1
    
    # Initialize database
    if args.init_db:
        print_banner()
        print_step("Initializing Database...")
        from scripts.init_db import init_database
        init_database()
        return 0
    
    # Run server
    if len(sys.argv) == 1 or not args.analyze:
        run_server(args)
        return 0
    
    # Run single analysis
    if args.analyze:
        try:
            asyncio.run(run_single_analysis(args))
            return 0
        except KeyboardInterrupt:
            print("\nAnalysis cancelled")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            return 1
    
    # Run analysis loop
    if args.analysis_loop:
        try:
            asyncio.run(run_analysis_loop(args))
            return 0
        except KeyboardInterrupt:
            print("\nAnalysis loop stopped")
            return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
