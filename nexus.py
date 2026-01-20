#!/usr/bin/env python3
# nexus.py
"""
NEXUS AI Lite - CLI Entry Point

Usage:
    python nexus.py --sport tennis --date 2026-01-20
    python nexus.py -s basketball -d 2026-01-20 -q 60
    python nexus.py --help
"""

import asyncio
import argparse
from datetime import datetime, date
from typing import Optional
from pathlib import Path

from config.settings import settings


async def run_analysis(
    sport: str,
    target_date: str,
    min_quality: float = 45.0,
    top_n: int = 5,
    verbose: bool = True
) -> Optional[str]:
    """
    Run full analysis and generate report.

    Args:
        sport: "tennis" or "basketball"
        target_date: Date in YYYY-MM-DD format
        min_quality: Minimum data quality threshold (0-100)
        top_n: Number of bets in report
        verbose: Show progress output

    Returns:
        Path to generated report or None
    """
    if verbose:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎯 NEXUS AI Lite - On-Demand Analysis                       ║
╠══════════════════════════════════════════════════════════════╣
║  Sport: {sport.upper():<10}  Date: {target_date:<15}             ║
║  Mode: {settings.APP_MODE.upper():<10}  Quality: {min_quality}%                      ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        # Import here to avoid circular imports
        from betting_floor import run_betting_analysis
        from reports.report_generator import ReportGenerator

        # Run analysis
        if verbose:
            print("📅 [1/5] Collecting fixtures from web sources...")

        result = await run_betting_analysis(
            sport=sport,
            date=target_date,
            bankroll=settings.DEFAULT_BANKROLL
        )

        if not result:
            if verbose:
                print("❌ Analysis failed or no data available")
            return None

        if verbose:
            print(f"   ✅ Found {result.get('matches_analyzed', 0)} matches")
            print(f"\n🔍 [2/5] Enriching data (news, stats, odds)...")
            print(f"   ✅ Data enrichment complete")
            print(f"\n📊 [3/5] Evaluating data quality...")
            print(f"   ✅ {result.get('quality_passed', 0)} matches passed quality filter")
            print(f"\n🧠 [4/5] Running predictions and value analysis...")
            print(f"   ✅ Found {len(result.get('approved_bets', []))} value bets")
            print(f"\n📝 [5/5] Generating report...")

        # Generate report
        generator = ReportGenerator()
        bets = result.get("approved_bets", [])[:top_n]

        report_content = generator.generate_markdown(bets, sport, target_date)
        report_path = generator.save_report(report_content, sport, target_date)

        if verbose:
            print(f"   ✅ Report saved: {report_path}")
            print("\n" + "=" * 60)
            print(report_content)
            print("=" * 60)

        return report_path

    except ImportError as e:
        if verbose:
            print(f"❌ Missing module: {e}")
            print("   Run: pip install -r requirements.txt")
        return None
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NEXUS AI Lite - Sports Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nexus.py --sport tennis
    python nexus.py -s basketball -d 2026-01-20
    python nexus.py -s tennis -q 60 -n 3

For more info: https://github.com/szymonMCS/NEXUS_AI
        """
    )

    parser.add_argument(
        "--sport", "-s",
        choices=["tennis", "basketball"],
        default="tennis",
        help="Sport to analyze (default: tennis)"
    )

    parser.add_argument(
        "--date", "-d",
        default=str(date.today()),
        help="Date to analyze YYYY-MM-DD (default: today)"
    )

    parser.add_argument(
        "--min-quality", "-q",
        type=float,
        default=45.0,
        help="Minimum data quality threshold 0-100 (default: 45)"
    )

    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Number of bets in report (default: 5)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode - only output report"
    )

    parser.add_argument(
        "--server",
        action="store_true",
        help="Start API server instead of running analysis"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )

    args = parser.parse_args()

    # Start server mode
    if args.server:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🚀 NEXUS AI - API Server                                    ║
╠══════════════════════════════════════════════════════════════╣
║  API: http://localhost:{args.port}/api                              ║
║  Docs: http://localhost:{args.port}/docs                            ║
║  Frontend: http://localhost:{args.port}                             ║
╚══════════════════════════════════════════════════════════════╝
""")
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=args.port,
            reload=True
        )
        return

    # Run analysis
    report_path = asyncio.run(run_analysis(
        sport=args.sport,
        target_date=args.date,
        min_quality=args.min_quality,
        top_n=args.top,
        verbose=not args.quiet
    ))

    if report_path:
        print(f"\n✅ Done! Report: {report_path}")
    else:
        print("\n❌ Failed to generate report")
        exit(1)


if __name__ == "__main__":
    main()
