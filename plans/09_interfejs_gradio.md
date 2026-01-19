## 9. INTERFEJS GRADIO

### 9.1 `ui/top3_tab.py` - Zakładka Top 3

```python
# ui/top3_tab.py
import gradio as gr
import pandas as pd
from datetime import datetime, date
from typing import List, Tuple
import asyncio

from agents.ranker import MatchRanker, RankedMatch, format_top_3_report

class Top3Tab:
    """
    Zakładka Gradio z Top 3 Value Bets.
    """

    def __init__(self):
        self.rankers = {
            "tennis": MatchRanker("tennis"),
            "basketball": MatchRanker("basketball")
        }
        self._last_results = {}

    def render(self) -> gr.Column:
        """Renderuje zakładkę Top 3"""
        with gr.Column() as tab:
            gr.HTML("""
            <div style='text-align: center; padding: 20px;'>
                <h2 style='color: #2e7d32;'>TOP 3 VALUE BETS TODAY</h2>
                <p style='color: #666;'>System automatically selects 3 best matches based on Value Edge × Data Quality × Confidence</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    sport_dropdown = gr.Dropdown(
                        choices=["tennis", "basketball", "both"],
                        value="tennis",
                        label="Sport"
                    )

                    date_picker = gr.Textbox(
                        value=str(date.today()),
                        label="Date (YYYY-MM-DD)"
                    )

                    min_quality = gr.Slider(
                        minimum=30,
                        maximum=80,
                        value=50,
                        step=5,
                        label="Minimum data quality threshold (%)"
                    )

                    generate_btn = gr.Button(
                        "Generate Top 3 Ranking",
                        variant="primary"
                    )

                with gr.Column(scale=2):
                    status_output = gr.Markdown("*Click 'Generate Ranking' to start analysis...*")

            gr.Markdown("---")

            # Top 3 Cards
            with gr.Row():
                top3_html = gr.HTML()

            # Detailed table
            with gr.Accordion("Detailed table of all matches", open=False):
                all_matches_table = gr.Dataframe(
                    headers=[
                        "Match", "League", "Type", "Edge", "Quality",
                        "Odds", "Bookmaker", "Stake", "Risk"
                    ],
                    interactive=False
                )

            # Quality reports
            with gr.Accordion("Data quality reports", open=False):
                quality_reports_md = gr.Markdown()

            # Event handler
            generate_btn.click(
                fn=self._generate_top3_async,
                inputs=[sport_dropdown, date_picker, min_quality],
                outputs=[top3_html, all_matches_table, quality_reports_md, status_output]
            )

        return tab

    def _generate_top3_async(
        self,
        sport: str,
        date_str: str,
        min_quality: float
    ) -> Tuple[str, pd.DataFrame, str, str]:
        """
        Wrapper async dla generowania Top 3.
        """
        return asyncio.run(self._generate_top3(sport, date_str, min_quality))

    async def _generate_top3(
        self,
        sport: str,
        date_str: str,
        min_quality: float
    ) -> Tuple[str, pd.DataFrame, str, str]:
        """
        Generuje Top 3 dla wybranego sportu.
        """
        status = f"Analyzing {sport} matches on {date_str}..."

        try:
            # Get fixtures (placeholder - would fetch from API)
            fixtures = await self._get_fixtures(sport, date_str)

            if not fixtures:
                return (
                    self._render_no_matches_html(),
                    pd.DataFrame(),
                    "No matches to analyze",
                    "No matches found for this date"
                )

            status = f"Analyzing {len(fixtures)} matches..."

            # Run ranker
            if sport == "both":
                tennis_top3 = await self.rankers["tennis"].rank_top_3_matches(date_str, fixtures["tennis"])
                basketball_top3 = await self.rankers["basketball"].rank_top_3_matches(date_str, fixtures["basketball"])
                all_matches = tennis_top3 + basketball_top3
                all_matches.sort(key=lambda x: x.composite_score, reverse=True)
                top3 = all_matches[:3]
            else:
                top3 = await self.rankers[sport].rank_top_3_matches(date_str, fixtures)

            # Render outputs
            html = self._render_top3_html(top3)
            df = self._create_dataframe(top3)
            quality_md = self._render_quality_reports(top3)

            status = f"Analysis complete! Found {len(top3)} value bets."

            return html, df, quality_md, status

        except Exception as e:
            return (
                f"<div style='color: red;'>Error: {str(e)}</div>",
                pd.DataFrame(),
                f"Error: {str(e)}",
                f"Error: {str(e)}"
            )

    def _render_top3_html(self, matches: List[RankedMatch]) -> str:
        """Renderuje Top 3 jako HTML cards"""
        if not matches:
            return self._render_no_matches_html()

        rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]  # Gold, Silver, Bronze
        risk_colors = {"LOW": "#4CAF50", "MEDIUM": "#FF9800", "HIGH": "#f44336"}

        html = "<div style='display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;'>"

        for match in matches:
            color = rank_colors[match.rank - 1] if match.rank <= 3 else "#666"
            risk_color = risk_colors.get(match.risk_level, "#666")

            html += f"""
            <div style='
                background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color};
                border-radius: 15px;
                padding: 20px;
                width: 300px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <div style='text-align: center; margin-bottom: 15px;'>
                    <span style='font-size: 2em;'>{"GoldMedalSilverMedalBronzeMedal"[match.rank-1] if match.rank <= 3 else "Medal"}</span>
                    <h3 style='margin: 5px 0; color: #333;'>#{match.rank}</h3>
                </div>

                <h4 style='margin: 10px 0; color: #222; text-align: center;'>
                    {match.match_name}
                </h4>

                <p style='color: #666; text-align: center; font-size: 0.9em;'>
                    {match.league}
                </p>

                <hr style='border: 1px solid #eee; margin: 15px 0;'>

                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #2e7d32;'>
                            +{match.adjusted_edge:.1%}
                        </div>
                        <div style='font-size: 0.8em; color: #666;'>Edge</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #1976d2;'>
                            {match.quality_score:.0f}
                        </div>
                        <div style='font-size: 0.8em; color: #666;'>Quality</div>
                    </div>
                </div>

                <div style='margin-top: 15px; padding: 10px; background: #f5f5f5; border-radius: 8px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>Odds:</span>
                        <strong>{match.best_odds:.2f}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>Bookmaker:</span>
                        <strong>{match.best_bookmaker}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Stake:</span>
                        <strong>{match.stake_recommendation}</strong>
                    </div>
                </div>

                <div style='
                    margin-top: 15px;
                    padding: 5px 10px;
                    background: {risk_color}22;
                    border: 1px solid {risk_color};
                    border-radius: 20px;
                    text-align: center;
                    color: {risk_color};
                    font-weight: bold;
                '>
                    Risk: {match.risk_level}
                </div>

                <div style='margin-top: 15px; font-size: 0.85em;'>
                    <strong>Reasoning:</strong>
                    <ul style='margin: 5px 0; padding-left: 20px;'>
                        {"".join(f"<li>{r}</li>" for r in match.reasoning[:3])}
                    </ul>
                </div>
            </div>
            """

        html += "</div>"
        return html

    def _render_no_matches_html(self) -> str:
        """Renderuje komunikat o braku meczów"""
        return """
        <div style='
            text-align: center;
            padding: 40px;
            background: #fff3e0;
            border-radius: 15px;
            border: 2px solid #ff9800;
        '>
            <span style='font-size: 3em;'>Magnifying Glass</span>
            <h3 style='color: #e65100;'>No value bets today</h3>
            <p style='color: #666;'>
                All matches either lack sufficient data quality,
                or do not offer a positive edge.
            </p>
        </div>
        """

    def _create_dataframe(self, matches: List[RankedMatch]) -> pd.DataFrame:
        """Tworzy DataFrame z meczami"""
        if not matches:
            return pd.DataFrame()

        data = []
        for m in matches:
            data.append({
                "Match": m.match_name,
                "League": m.league,
                "Type": m.selection,
                "Edge": f"+{m.adjusted_edge:.1%}",
                "Quality": f"{m.quality_score:.0f}%",
                "Odds": f"{m.best_odds:.2f}",
                "Bookmaker": m.best_bookmaker,
                "Stake": m.stake_recommendation,
                "Risk": m.risk_level
            })

        return pd.DataFrame(data)

    def _render_quality_reports(self, matches: List[RankedMatch]) -> str:
        """Renderuje raporty jakości jako Markdown"""
        if not matches:
            return "No data"

        md = ""
        for m in matches:
            qr = m.quality_report
            md += f"""
### {m.match_name}

| Metric | Value |
|--------|-------|
| Overall Score | {qr.overall_score:.1f}/100 |
| News Quality | {qr.news_quality_score:.1f}/100 |
| Stats Complete | {qr.stats_completeness_score:.1f}/100 |
| Odds Quality | {qr.odds_quality_score:.1f}/100 |
| Bookmakers | {qr.odds_sources_count} |
| Recommendation | **{qr.recommendation}** |

"""
            if qr.injuries_found:
                md += "**Injuries:**\n"
                for inj in qr.injuries_found:
                    md += f"- {inj['player']}: {inj['status']} ({inj['injury_type']})\n"
                md += "\n"

            if qr.warnings:
                md += "**Warnings:**\n"
                for w in qr.warnings[:3]:
                    md += f"- {w}\n"
                md += "\n"

            md += "---\n\n"

        return md

    async def _get_fixtures(self, sport: str, date_str: str) -> List[Dict]:
        """
        Pobiera fixtures z API.
        Placeholder - w rzeczywistej implementacji użyłby MCP servers.
        """
        # This would call the MCP servers to get real data
        # For now, return mock data
        return []
```

---
