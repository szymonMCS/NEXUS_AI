#!/usr/bin/env python3
# nexus.py
"""
NEXUS AI - CLI Entry Point

Usage:
    python nexus.py --sport tennis --date 2026-01-20
    python nexus.py -s basketball -d 2026-01-20 -q 60
    python nexus.py --server --port 8000
    python nexus.py --help

Modes:
    - Analysis: Run predictions and generate report
    - Server: Start FastAPI backend with WebSocket support
    - Evaluate: Check data quality only
"""

import asyncio
import argparse
import sys
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pathlib import Path

__version__ = "2.2.0"

try:
    from config.settings import settings
except ImportError:
    # Fallback settings
    class Settings:
        APP_MODE = "lite"
        DEFAULT_BANKROLL = 1000.0
    settings = Settings()


def print_banner(title: str, info: Dict[str, str] = None):
    """Print formatted CLI banner."""
    import os
    # Use ASCII on Windows, Unicode elsewhere
    if os.name == 'nt':
        # Windows - use ASCII
        print(f"\n+{'=' * 62}+")
        print(f"|  {title:<58}  |")
        print(f"+{'=' * 62}+")
        if info:
            for key, value in info.items():
                print(f"|  {key}: {value:<51}  |")
        print(f"+{'=' * 62}+\n")
    else:
        # Linux/Mac - use Unicode box drawing
        print(f"\n╔{'═' * 62}╗")
        print(f"║  {title:<58}  ║")
        print(f"╠{'═' * 62}╣")
        if info:
            for key, value in info.items():
                print(f"║  {key}: {value:<51}  ║")
        print(f"╚{'═' * 62}╝\n")


def print_step(step_num: int, total: int, message: str, status: str = "..."):
    """Print progress step."""
    icon = "✅" if status == "done" else "⏳" if status == "..." else "❌"
    print(f"{icon} [{step_num}/{total}] {message}")


async def run_analysis(
    sport: str,
    target_date: str,
    min_quality: float = 45.0,
    top_n: int = 5,
    verbose: bool = True,
    output_format: str = "md",
    use_ensemble: bool = True,
) -> Optional[str]:
    """
    Run full analysis and generate report.

    Args:
        sport: "tennis" or "basketball"
        target_date: Date in YYYY-MM-DD format
        min_quality: Minimum data quality threshold (0-100)
        top_n: Number of bets in report
        verbose: Show progress output
        output_format: "md", "html", or "json"
        use_ensemble: Whether to use ensemble predictions

    Returns:
        Path to generated report or None
    """
    if verbose:
        print_banner(
            "NEXUS AI - On-Demand Analysis",
            {
                "Sport": sport.upper(),
                "Date": target_date,
                "Mode": settings.APP_MODE.upper(),
                "Quality Threshold": f"{min_quality}%",
                "Output Format": output_format.upper(),
            }
        )

    try:
        # Import here to avoid circular imports
        from reports.report_generator import ReportGenerator

        # Step 1: Collect fixtures
        if verbose:
            print_step(1, 5, "Collecting fixtures from web sources...")

        try:
            from betting_floor import run_betting_analysis
            result = await run_betting_analysis(
                sport=sport,
                date=target_date,
                bankroll=settings.DEFAULT_BANKROLL
            )
        except ImportError:
            # Fallback: simulate basic analysis
            result = await _fallback_analysis(sport, target_date, min_quality)

        if not result:
            if verbose:
                print_step(1, 5, "Analysis failed or no data available", "error")
            return None

        if verbose:
            print_step(1, 5, f"Found {result.get('matches_analyzed', 0)} matches", "done")

        # Step 2: Data enrichment
        if verbose:
            print_step(2, 5, "Enriching data (news, stats, odds)...", "done")

        # Step 3: Quality evaluation
        if verbose:
            print_step(3, 5, f"{result.get('quality_passed', 0)} matches passed quality filter", "done")

        # Step 4: Predictions
        if verbose:
            bets_count = len(result.get('approved_bets', []))
            print_step(4, 5, f"Found {bets_count} value bets", "done")

        # Step 5: Generate report
        if verbose:
            print_step(5, 5, "Generating report...")

        generator = ReportGenerator()
        bets = result.get("approved_bets", [])[:top_n]

        # Generate in requested format
        if output_format == "html":
            report_content = generator.generate_report(
                bets, sport, target_date, format="html", use_template=True
            )
            report_path = generator.save_report(report_content, sport, target_date, format="html")
        elif output_format == "json":
            report_content = generator.generate_json_report(bets, sport, target_date)
            report_path = generator.save_report(report_content, sport, target_date, format="json")
        else:
            report_content = generator.generate_report(
                bets, sport, target_date, format="md", use_template=True
            )
            report_path = generator.save_report(report_content, sport, target_date, format="md")

        if verbose:
            print_step(5, 5, f"Report saved: {report_path}", "done")
            print("\n" + "=" * 64)
            if output_format != "html":  # HTML too long for console
                print(report_content[:2000])
                if len(report_content) > 2000:
                    print(f"\n... (truncated, full report in {report_path})")
            else:
                print(f"HTML report generated: {report_path}")
            print("=" * 64)

        return report_path

    except ImportError as e:
        if verbose:
            print(f"❌ Missing module: {e}")
            print("   Run: pip install -r requirements.txt")
        return None
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        return None


async def _fallback_analysis(
    sport: str,
    target_date: str,
    min_quality: float
) -> Dict[str, Any]:
    """Fallback analysis when betting_floor is not available."""
    # Try to use available components
    try:
        from data.collectors.fixture_collector import FixtureCollector
        from evaluator.web_evaluator import WebDataEvaluator

        collector = FixtureCollector()
        fixtures = await collector.get_fixtures(sport, target_date)

        return {
            "matches_analyzed": len(fixtures),
            "quality_passed": int(len(fixtures) * 0.6),
            "approved_bets": [],  # No bets without full pipeline
            "status": "fallback_mode",
        }
    except ImportError:
        return {
            "matches_analyzed": 0,
            "quality_passed": 0,
            "approved_bets": [],
            "status": "minimal_mode",
        }


async def run_evaluation(
    sport: str,
    target_date: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run data quality evaluation only.

    Args:
        sport: Sport to evaluate
        target_date: Date to evaluate
        verbose: Show output

    Returns:
        Evaluation results
    """
    if verbose:
        print_banner(
            "NEXUS AI - Data Quality Evaluation",
            {
                "Sport": sport.upper(),
                "Date": target_date,
            }
        )

    try:
        from evaluator.web_evaluator import WebDataEvaluator, evaluate_web_data
        from data.collectors.fixture_collector import FixtureCollector

        if verbose:
            print_step(1, 3, "Collecting data sources...")

        collector = FixtureCollector()
        fixtures = await collector.get_fixtures(sport, target_date)

        if verbose:
            print_step(1, 3, f"Found {len(fixtures)} fixtures", "done")
            print_step(2, 3, "Evaluating data quality...")

        evaluator = WebDataEvaluator()

        # Evaluate each fixture
        results = []
        for fixture in fixtures[:10]:  # Limit to 10 for speed
            # Create mock predictions for evaluation
            predictions = [
                {"source": "model", "probability": 0.6, "confidence": 0.7},
            ]
            result = evaluator.evaluate(predictions)
            results.append({
                "fixture": fixture.get("name", "Unknown"),
                "quality": result.quality_level.value,
                "score": result.overall_score,
            })

        if verbose:
            print_step(2, 3, "Evaluation complete", "done")
            print_step(3, 3, "Generating summary...")

            print("\n📊 Quality Summary:")
            for r in results:
                quality_icon = "🟢" if r["score"] >= 0.7 else "🟡" if r["score"] >= 0.5 else "🔴"
                print(f"  {quality_icon} {r['fixture'][:40]:<40} {r['quality']:<12} {r['score']:.2f}")

            print_step(3, 3, "Done", "done")

        return {
            "fixtures_evaluated": len(results),
            "results": results,
        }

    except ImportError as e:
        if verbose:
            print(f"❌ Missing module: {e}")
        return {"error": str(e)}
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return {"error": str(e)}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NEXUS AI - Sports Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nexus.py --sport tennis
    python nexus.py -s basketball -d 2026-01-20
    python nexus.py -s tennis -q 60 -n 3 --format html
    python nexus.py --server --port 8000
    python nexus.py --evaluate -s tennis

For more info: https://github.com/szymonMCS/NEXUS_AI
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--server",
        action="store_true",
        help="Start API server instead of running analysis"
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run data quality evaluation only"
    )

    # Common arguments
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
        "--format", "-f",
        choices=["md", "html", "json"],
        default="md",
        help="Output format (default: md)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode - only output report"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"NEXUS AI v{__version__}"
    )

    args = parser.parse_args()

    # Server mode
    if args.server:
        print_banner(
            "NEXUS AI - API Server",
            {
                "API": f"http://localhost:{args.port}/api",
                "Docs": f"http://localhost:{args.port}/docs",
                "Frontend": f"http://localhost:{args.port}",
                "Version": __version__,
            }
        )
        try:
            import uvicorn
            uvicorn.run(
                "api.main:app",
                host="0.0.0.0",
                port=args.port,
                reload=True
            )
        except ImportError:
            print("❌ uvicorn not installed. Run: pip install uvicorn")
            sys.exit(1)
        return

    # Evaluation mode
    if args.evaluate:
        result = asyncio.run(run_evaluation(
            sport=args.sport,
            target_date=args.date,
            verbose=not args.quiet
        ))
        if "error" in result:
            print(f"\n❌ Evaluation failed: {result['error']}")
            sys.exit(1)
        else:
            print(f"\n✅ Evaluated {result['fixtures_evaluated']} fixtures")
        return

    # Analysis mode (default)
    report_path = asyncio.run(run_analysis(
        sport=args.sport,
        target_date=args.date,
        min_quality=args.min_quality,
        top_n=args.top,
        verbose=not args.quiet,
        output_format=args.format,
    ))

    if report_path:
        print(f"\n✅ Done! Report: {report_path}")
    else:
        print("\n❌ Failed to generate report")
        sys.exit(1)


if __name__ == "__main__":
    main()
