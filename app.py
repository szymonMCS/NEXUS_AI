# app.py
"""
NEXUS AI - Gradio Web Interface
Interactive dashboard for sports betting analysis.
"""

import gradio as gr
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import pandas as pd

from config.settings import settings
from database.db import init_db, get_db_session
from database.crud import (
    get_performance_summary, get_roi_by_sport,
    get_pending_bets, get_active_session
)
from betting_floor import BettingFloor, run_once


# Initialize components
floor = None


def get_floor():
    """Get or create BettingFloor instance."""
    global floor
    if floor is None:
        floor = BettingFloor()
        asyncio.run(floor.initialize())
    return floor


# === UI FUNCTIONS ===

async def run_analysis_async(sport: str, date: str) -> tuple:
    """
    Run betting analysis asynchronously.

    Args:
        sport: Sport to analyze
        date: Date to analyze

    Returns:
        Tuple of (results_text, bets_df, messages_text)
    """
    try:
        results = await run_once(sport, date or datetime.now().strftime("%Y-%m-%d"))

        # Format results
        results_text = f"""
## Analysis Complete

**Sport:** {results.get('sport', 'N/A')}
**Date:** {results.get('date', 'N/A')}
**Matches Analyzed:** {results.get('matches_analyzed', 0)}
**Top Matches:** {results.get('top_matches', 0)}
**Duration:** {results.get('duration_seconds', 0):.1f} seconds
"""

        # Format approved bets as DataFrame
        approved = results.get('approved_bets', [])
        if approved:
            bets_data = []
            for bet in approved:
                value_bet = bet.get('value_bet', {})
                prediction = bet.get('prediction', {})

                bets_data.append({
                    'Match': f"{bet.get('home_player', {}).get('name', '?')} vs {bet.get('away_player', {}).get('name', '?')}",
                    'Bet': value_bet.get('bet_on', 'N/A').upper(),
                    'Odds': f"{value_bet.get('odds', 0):.2f}",
                    'Edge': f"{value_bet.get('edge', 0):.1%}",
                    'Stake %': f"{value_bet.get('kelly_stake', 0):.1%}",
                    'Confidence': f"{prediction.get('confidence', 0):.1%}",
                })

            bets_df = pd.DataFrame(bets_data)
        else:
            bets_df = pd.DataFrame({'Message': ['No approved bets found']})

        # Format messages
        messages = results.get('messages', [])
        messages_text = "\n".join([
            f"[{m.get('agent', '?')}] {m.get('message', '')}"
            for m in messages[-20:]  # Last 20 messages
        ])

        return results_text, bets_df, messages_text

    except Exception as e:
        error_text = f"## Error\n\n{str(e)}"
        return error_text, pd.DataFrame({'Error': [str(e)]}), str(e)


def run_analysis(sport: str, date: str) -> tuple:
    """Wrapper to run async function in sync context."""
    return asyncio.run(run_analysis_async(sport, date))


def get_dashboard_stats() -> tuple:
    """
    Get dashboard statistics.

    Returns:
        Tuple of (stats_text, roi_df, pending_df)
    """
    try:
        init_db()

        with get_db_session() as db:
            performance = get_performance_summary(db, days=30)
            roi_by_sport = get_roi_by_sport(db, days=30)
            pending = get_pending_bets(db)
            session = get_active_session(db)

        # Format stats
        stats_text = f"""
## Performance Summary (Last 30 Days)

**Total Bets:** {performance.get('total_bets', 0)}
**Win Rate:** {performance.get('win_rate', 0):.1f}%
**ROI:** {performance.get('roi', 0):.2f}%
**Total Profit:** ${performance.get('total_profit', 0):.2f}
**Avg Stake:** ${performance.get('avg_stake', 0):.2f}

### Current Session
**Bankroll:** ${session.current_bankroll if session else 1000:.2f}
**Session ROI:** {session.roi_percentage if session else 0:.2f}%
**Mode:** {settings.APP_MODE.upper()}
"""

        # ROI by sport
        if roi_by_sport:
            roi_data = [{'Sport': sport, 'ROI %': f"{roi:.2f}"} for sport, roi in roi_by_sport.items()]
            roi_df = pd.DataFrame(roi_data)
        else:
            roi_df = pd.DataFrame({'Sport': ['No data'], 'ROI %': ['-']})

        # Pending bets
        if pending:
            pending_data = []
            for bet in pending[:10]:
                pending_data.append({
                    'Match ID': bet.match_id,
                    'Selection': bet.selection,
                    'Odds': f"{bet.odds:.2f}",
                    'Stake': f"${bet.stake:.2f}",
                })
            pending_df = pd.DataFrame(pending_data)
        else:
            pending_df = pd.DataFrame({'Status': ['No pending bets']})

        return stats_text, roi_df, pending_df

    except Exception as e:
        error = f"## Error loading stats\n\n{str(e)}"
        return error, pd.DataFrame(), pd.DataFrame()


def get_system_status() -> str:
    """Get current system status."""
    try:
        floor = get_floor()
        status = asyncio.run(floor.get_status())

        return f"""
## System Status

**Running:** {'Yes' if status.get('is_running') else 'No'}
**Last Run:** {status.get('last_run', 'Never')}
**Mode:** {status.get('mode', 'unknown').upper()}

### Current Session
{json.dumps(status.get('current_session', {}), indent=2)}

### Performance
{json.dumps(status.get('performance', {}), indent=2)}
"""
    except Exception as e:
        return f"## Error\n\n{str(e)}"


# === BUILD UI ===

def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="NEXUS AI - Sports Betting Analysis",
        theme=gr.themes.Soft()
    ) as app:

        gr.Markdown("""
        # NEXUS AI
        ### Intelligent Sports Betting Analysis System

        Multi-agent AI system for finding value bets in tennis and basketball.
        """)

        with gr.Tabs():
            # === ANALYSIS TAB ===
            with gr.TabItem("Run Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sport_dropdown = gr.Dropdown(
                            choices=["tennis", "basketball"],
                            value="tennis",
                            label="Sport"
                        )
                        date_input = gr.Textbox(
                            label="Date (YYYY-MM-DD)",
                            placeholder="Leave empty for today",
                            value=""
                        )
                        run_btn = gr.Button("Run Analysis", variant="primary")

                    with gr.Column(scale=2):
                        results_output = gr.Markdown(label="Results")

                with gr.Row():
                    bets_table = gr.DataFrame(label="Approved Bets")

                with gr.Row():
                    messages_output = gr.Textbox(
                        label="Agent Messages",
                        lines=10,
                        max_lines=20
                    )

                run_btn.click(
                    fn=run_analysis,
                    inputs=[sport_dropdown, date_input],
                    outputs=[results_output, bets_table, messages_output]
                )

            # === DASHBOARD TAB ===
            with gr.TabItem("Dashboard"):
                refresh_btn = gr.Button("Refresh Stats")

                with gr.Row():
                    stats_output = gr.Markdown()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ROI by Sport")
                        roi_table = gr.DataFrame()

                    with gr.Column():
                        gr.Markdown("### Pending Bets")
                        pending_table = gr.DataFrame()

                refresh_btn.click(
                    fn=get_dashboard_stats,
                    outputs=[stats_output, roi_table, pending_table]
                )

                # Load on page open
                app.load(
                    fn=get_dashboard_stats,
                    outputs=[stats_output, roi_table, pending_table]
                )

            # === STATUS TAB ===
            with gr.TabItem("System Status"):
                status_refresh_btn = gr.Button("Refresh Status")
                status_output = gr.Markdown()

                status_refresh_btn.click(
                    fn=get_system_status,
                    outputs=[status_output]
                )

            # === SETTINGS TAB ===
            with gr.TabItem("Settings"):
                gr.Markdown(f"""
                ## Current Configuration

                | Setting | Value |
                |---------|-------|
                | App Mode | {settings.APP_MODE.upper()} |
                | Database | {settings.DATABASE_URL[:30]}... |
                | Run Interval | {settings.RUN_EVERY_N_MINUTES} minutes |
                | Max Concurrent | {settings.MAX_CONCURRENT_REQUESTS} |
                | Debug Mode | {settings.DEBUG} |

                ### API Status

                | API | Status |
                |-----|--------|
                | Brave Search | {'Configured' if settings.BRAVE_API_KEY else 'Not configured'} |
                | Serper | {'Configured' if settings.SERPER_API_KEY else 'Not configured'} |
                | The Odds API | {'Configured' if settings.ODDS_API_KEY else 'Not configured'} |
                | API-Tennis | {'Configured' if settings.API_TENNIS_KEY else 'Not configured'} |
                | Anthropic | {'Configured' if settings.ANTHROPIC_API_KEY else 'Not configured'} |

                ---

                To change settings, edit the `.env` file or set environment variables.
                """)

            # === HELP TAB ===
            with gr.TabItem("Help"):
                gr.Markdown("""
                ## How to Use NEXUS AI

                ### Quick Start

                1. **Run Analysis**: Select a sport and date, then click "Run Analysis"
                2. **Review Bets**: Check the approved bets table for recommendations
                3. **Monitor Performance**: Use the Dashboard to track your results

                ### Understanding the Output

                - **Edge**: Expected profit percentage over implied probability
                - **Stake %**: Recommended stake as percentage of bankroll (Kelly Criterion)
                - **Confidence**: AI's confidence in the prediction

                ### Quality Levels

                - **Excellent** (>85%): High confidence, multiple data sources
                - **Good** (70-85%): Solid data quality
                - **Moderate** (50-70%): Acceptable for lower stakes
                - **High Risk** (40-50%): Limited data, proceed with caution
                - **Insufficient** (<40%): Rejected, not enough data

                ### Tips

                - Focus on matches with edge > 3% for popular leagues
                - Higher edge requirements for less popular leagues (5%+)
                - Use fractional Kelly (25%) to reduce variance
                - Monitor injury reports before placing bets

                ### Support

                For issues and feature requests, visit the GitHub repository.
                """)

        gr.Markdown("""
        ---
        *NEXUS AI v2.0 | Powered by Claude & LangGraph*
        """)

    return app


# === MAIN ===

def main():
    """Launch the Gradio application."""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
