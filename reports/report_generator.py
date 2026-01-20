# reports/report_generator.py
"""
Report generator for NEXUS AI.
Generates Markdown and HTML reports from analysis results.
Supports templates and multiple report types.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import re
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankedBet:
    """Structured bet for reports."""
    rank: int
    match_name: str
    league: str
    selection: str
    odds: float
    bookmaker: str
    edge: float
    quality_score: float
    stake_recommendation: str
    confidence: float
    reasoning: List[str]
    warnings: List[str] = None
    match_time: str = ""
    factors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.factors is None:
            self.factors = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "match_name": self.match_name,
            "league": self.league,
            "selection": self.selection,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "edge": f"{self.edge:.1%}",
            "quality_score": self.quality_score,
            "stake_recommendation": self.stake_recommendation,
            "confidence": f"{self.confidence:.0%}",
            "reasoning": self.reasoning,
            "warnings": self.warnings,
            "match_time": self.match_time,
            "factors": self.factors,
        }


@dataclass
class ReportContext:
    """Context data for report generation."""
    sport: str
    date: str
    bets: List[Union[Dict[str, Any], RankedBet]]
    generated_at: str = ""
    bet_count: int = 0
    avg_edge: str = "0%"
    avg_quality: str = "0"
    total_stake: str = "0%"
    quality_details: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_risk: str = "Low"
    max_single_bet: str = "5%"
    correlation_warning: str = "None"

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
        if not self.bet_count:
            self.bet_count = len(self.bets)


class TemplateEngine:
    """Simple template engine for report generation."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with given context.

        Args:
            template_name: Name of template file
            context: Dictionary of variables

        Returns:
            Rendered template string
        """
        template_path = self.templates_dir / template_name

        if not template_path.exists():
            logger.warning(f"Template {template_name} not found, using inline generation")
            return None

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        return self._process_template(template, context)

    def _process_template(self, template: str, context: Dict[str, Any]) -> str:
        """Process template with simple variable substitution and loops."""
        result = template

        # Process for loops: {% for item in items %}...{% endfor %}
        for_pattern = r'\{% for (\w+) in (\w+) %\}(.*?)\{% endfor %\}'

        def replace_for(match):
            item_name = match.group(1)
            list_name = match.group(2)
            block = match.group(3)

            items = context.get(list_name, [])
            if not items:
                return ""

            rendered_blocks = []
            for item in items:
                block_context = {**context, item_name: item}
                rendered = self._substitute_variables(block, block_context)
                rendered_blocks.append(rendered)

            return "".join(rendered_blocks)

        result = re.sub(for_pattern, replace_for, result, flags=re.DOTALL)

        # Process conditionals: {% if condition %}...{% endif %}
        if_pattern = r'\{% if (\w+(?:\.\w+)*) %\}(.*?)\{% endif %\}'

        def replace_if(match):
            condition_path = match.group(1)
            block = match.group(2)

            value = self._get_nested_value(context, condition_path)
            if value:
                return self._substitute_variables(block, context)
            return ""

        result = re.sub(if_pattern, replace_if, result, flags=re.DOTALL)

        # Process simple variables: {{ variable }}
        result = self._substitute_variables(result, context)

        return result

    def _substitute_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Substitute {{ variable }} patterns."""
        var_pattern = r'\{\{\s*(\w+(?:\.\w+)*)\s*\}\}'

        def replace_var(match):
            path = match.group(1)
            value = self._get_nested_value(context, path)
            if value is None:
                return ""
            return str(value)

        return re.sub(var_pattern, replace_var, template)

    def _get_nested_value(self, context: Dict[str, Any], path: str) -> Any:
        """Get nested value from context using dot notation."""
        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value


class ReportGenerator:
    """
    Generates reports in Markdown and HTML formats.
    Supports template-based generation and multiple report types.
    """

    def __init__(self, output_dir: str = "outputs", templates_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.template_engine = TemplateEngine(templates_dir)

    def generate_report(
        self,
        bets: List[Union[Dict[str, Any], RankedBet]],
        sport: str,
        date: str,
        format: str = "md",
        use_template: bool = True,
        quality_details: Optional[List[Dict[str, Any]]] = None,
        data_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate report using templates or inline generation.

        Args:
            bets: List of bet dictionaries or RankedBet objects
            sport: Sport name
            date: Analysis date
            format: "md" or "html"
            use_template: Whether to use template files
            quality_details: Optional quality breakdown data
            data_sources: Optional data sources info

        Returns:
            Generated report content
        """
        # Build context
        context = self._build_context(
            bets, sport, date, quality_details, data_sources
        )

        if use_template:
            template_name = f"report_template.{format}"
            rendered = self.template_engine.render(template_name, context)
            if rendered:
                return rendered

        # Fallback to inline generation
        if format == "html":
            return self.generate_html(bets, sport, date)
        else:
            return self.generate_markdown(bets, sport, date)

    def _build_context(
        self,
        bets: List[Union[Dict[str, Any], RankedBet]],
        sport: str,
        date: str,
        quality_details: Optional[List[Dict[str, Any]]] = None,
        data_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build template context from bets data."""
        # Convert bets to dicts
        bet_dicts = []
        for bet in bets:
            if isinstance(bet, RankedBet):
                bet_dicts.append(bet.to_dict())
            else:
                bet_dicts.append(self._normalize_bet_dict(bet))

        # Calculate aggregates
        if bets:
            avg_edge = sum(
                (b.get("edge", 0) if isinstance(b, dict) else b.edge)
                for b in bets
            ) / len(bets)
            avg_quality = sum(
                (b.get("quality_score", 0) if isinstance(b, dict) else b.quality_score)
                for b in bets
            ) / len(bets)
        else:
            avg_edge = 0
            avg_quality = 0

        return {
            "sport": sport.upper(),
            "date": date,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "bets": bet_dicts,
            "bet_count": len(bets),
            "avg_edge": f"{avg_edge:.1%}",
            "avg_quality": f"{avg_quality:.0f}",
            "total_stake": self._calculate_total_stake(bets),
            "quality_details": quality_details or [],
            "data_sources": data_sources or [],
            "portfolio_risk": self._assess_portfolio_risk(bets),
            "max_single_bet": "5%",
            "correlation_warning": self._check_correlations(bets),
        }

    def _normalize_bet_dict(self, bet: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a bet dictionary to standard format."""
        edge = bet.get("edge", 0)
        confidence = bet.get("confidence", 0)

        return {
            "rank": bet.get("rank", 0),
            "match_name": bet.get("match_name", bet.get("match", "Unknown")),
            "league": bet.get("league", "Unknown"),
            "selection": bet.get("selection", "").upper(),
            "odds": bet.get("odds", 0),
            "bookmaker": bet.get("bookmaker", ""),
            "edge": f"+{edge:.1%}" if isinstance(edge, float) else edge,
            "quality_score": bet.get("quality_score", 0),
            "stake_recommendation": bet.get("stake_recommendation", bet.get("stake", "1%")),
            "confidence": f"{confidence:.0%}" if isinstance(confidence, float) else confidence,
            "reasoning": bet.get("reasoning", []),
            "warnings": bet.get("warnings", []),
            "match_time": bet.get("match_time", bet.get("time", "")),
            "factors": bet.get("factors", []),
        }

    def _calculate_total_stake(self, bets: List[Union[Dict[str, Any], RankedBet]]) -> str:
        """Calculate total recommended stake."""
        total = 0.0
        for bet in bets:
            if isinstance(bet, RankedBet):
                stake_str = bet.stake_recommendation
            else:
                stake_str = bet.get("stake_recommendation", bet.get("stake", "0%"))

            # Parse stake percentage
            try:
                stake_val = float(stake_str.rstrip("%"))
                total += stake_val
            except (ValueError, AttributeError):
                pass

        return f"{total:.1f}%"

    def _assess_portfolio_risk(self, bets: List[Union[Dict[str, Any], RankedBet]]) -> str:
        """Assess overall portfolio risk level."""
        if not bets:
            return "None"

        total_stake = float(self._calculate_total_stake(bets).rstrip("%"))

        if total_stake > 15:
            return "High"
        elif total_stake > 10:
            return "Medium"
        else:
            return "Low"

    def _check_correlations(self, bets: List[Union[Dict[str, Any], RankedBet]]) -> str:
        """Check for correlated bets (same league/tournament)."""
        leagues = []
        for bet in bets:
            if isinstance(bet, RankedBet):
                leagues.append(bet.league)
            else:
                leagues.append(bet.get("league", ""))

        # Check for duplicates
        unique_leagues = set(leagues)
        if len(unique_leagues) < len(leagues):
            duplicates = [l for l in unique_leagues if leagues.count(l) > 1]
            return f"Multiple bets in: {', '.join(duplicates)}"

        return "None"

    def generate_json_report(
        self,
        bets: List[Union[Dict[str, Any], RankedBet]],
        sport: str,
        date: str,
    ) -> str:
        """Generate JSON report for API/programmatic use."""
        context = self._build_context(bets, sport, date)
        return json.dumps(context, indent=2, default=str)

    def generate_quality_report(
        self,
        match_id: str,
        match_name: str,
        quality_components: Dict[str, float],
        issues: List[str],
    ) -> str:
        """Generate a detailed quality report for a single match."""
        overall = quality_components.get("overall_score", 0)

        # Determine quality level
        if overall >= 0.85:
            level = "EXCELLENT"
            recommendation = "Full confidence - proceed with standard stake"
        elif overall >= 0.70:
            level = "GOOD"
            recommendation = "Good quality - standard betting recommended"
        elif overall >= 0.50:
            level = "MODERATE"
            recommendation = "Moderate quality - consider reduced stake"
        elif overall >= 0.30:
            level = "HIGH RISK"
            recommendation = "High risk - minimal bet or skip"
        else:
            level = "INSUFFICIENT"
            recommendation = "Insufficient data - do not bet"

        lines = [
            f"# Quality Report: {match_name}",
            f"",
            f"**Match ID:** {match_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"---",
            f"",
            f"## Quality Scores",
            f"",
            f"| Component | Score |",
            f"|-----------|-------|",
            f"| News Quality | {quality_components.get('news_score', 0):.1%} |",
            f"| Odds Quality | {quality_components.get('odds_score', 0):.1%} |",
            f"| Stats Quality | {quality_components.get('stats_score', 0):.1%} |",
            f"| **Overall** | **{overall:.1%}** |",
            f"",
            f"## Assessment",
            f"",
            f"**Quality Level:** {level}",
            f"",
            f"**Recommendation:** {recommendation}",
            f"",
        ]

        if issues:
            lines.extend([
                f"## Issues Identified",
                f"",
            ])
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

        lines.extend([
            f"---",
            f"",
            f"*Report generated by NEXUS AI v2.2.0*",
        ])

        return "\n".join(lines)

    def generate_markdown(
        self,
        bets: List[Dict[str, Any]],
        sport: str,
        date: str
    ) -> str:
        """
        Generate Markdown report.

        Args:
            bets: List of bet dictionaries
            sport: Sport name
            date: Analysis date

        Returns:
            Markdown content string
        """
        if not bets:
            return self._generate_no_bets_report(sport, date)

        lines = [
            f"# NEXUS AI - Prediction Report",
            f"",
            f"**Sport:** {sport.upper()}  ",
            f"**Date:** {date}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"---",
            f"",
            f"## TOP {len(bets)} VALUE BETS",
            f""
        ]

        rank_emoji = ["1.", "2.", "3.", "4.", "5."]

        for i, bet in enumerate(bets):
            emoji = rank_emoji[i] if i < 5 else f"{i+1}."

            # Handle both dict and RankedBet objects
            if isinstance(bet, dict):
                match_name = bet.get("match_name", bet.get("match", "Unknown"))
                league = bet.get("league", "Unknown")
                selection = bet.get("selection", "")
                odds = bet.get("odds", 0)
                bookmaker = bet.get("bookmaker", "")
                edge = bet.get("edge", 0)
                quality_score = bet.get("quality_score", 0)
                stake = bet.get("stake_recommendation", bet.get("stake", "1%"))
                confidence = bet.get("confidence", 0)
                reasoning = bet.get("reasoning", [])
                warnings = bet.get("warnings", [])
            else:
                match_name = bet.match_name
                league = bet.league
                selection = bet.selection
                odds = bet.odds
                bookmaker = bet.bookmaker
                edge = bet.edge
                quality_score = bet.quality_score
                stake = bet.stake_recommendation
                confidence = bet.confidence
                reasoning = bet.reasoning
                warnings = bet.warnings

            lines.extend([
                f"### {emoji} {match_name}",
                f"",
                f"**League:** {league}  ",
                f"**Selection:** {selection.upper()}  ",
                f"**Odds:** {odds:.2f} @ {bookmaker}  ",
                f"**Edge:** +{edge:.1%}  ",
                f"**Data Quality:** {quality_score:.0f}/100  ",
                f"**Confidence:** {confidence:.0%}  ",
                f"**Stake:** {stake}",
                f""
            ])

            if reasoning:
                lines.append("**Reasoning:**")
                for r in reasoning[:3]:
                    lines.append(f"> {r}")
                lines.append("")

            if warnings:
                lines.append("**Warnings:**")
                for w in warnings[:2]:
                    lines.append(f"- {w}")
                lines.append("")

            lines.extend(["---", ""])

        # Summary
        avg_edge = sum(
            (b.get("edge", 0) if isinstance(b, dict) else b.edge)
            for b in bets
        ) / len(bets)
        avg_quality = sum(
            (b.get("quality_score", 0) if isinstance(b, dict) else b.quality_score)
            for b in bets
        ) / len(bets)

        lines.extend([
            f"## Summary",
            f"",
            f"- **Value bets found:** {len(bets)}",
            f"- **Average edge:** {avg_edge:.1%}",
            f"- **Average data quality:** {avg_quality:.0f}/100",
            f"",
            f"---",
            f"",
            f"*Report generated by NEXUS AI Lite v2.0*"
        ])

        return "\n".join(lines)

    def _generate_no_bets_report(self, sport: str, date: str) -> str:
        """Generate report when no bets are found."""
        return f"""# NEXUS AI - Prediction Report

**Sport:** {sport.upper()}
**Date:** {date}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## No Value Bets Found

The system did not find any bets meeting quality and value criteria.

**Possible reasons:**
- Insufficient data quality from web sources
- No matches with positive edge
- Bookmaker odds are too efficient

**Recommendation:** Try again later or check a different sport.

---

*Report generated by NEXUS AI Lite v2.0*
"""

    def generate_html(
        self,
        bets: List[Dict[str, Any]],
        sport: str,
        date: str
    ) -> str:
        """
        Generate HTML report with styling.

        Args:
            bets: List of bet dictionaries
            sport: Sport name
            date: Analysis date

        Returns:
            HTML content string
        """
        if not bets:
            return self._generate_no_bets_html(sport, date)

        bet_cards = ""
        rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#666", "#666"]

        for i, bet in enumerate(bets):
            color = rank_colors[i] if i < 5 else "#666"

            if isinstance(bet, dict):
                match_name = bet.get("match_name", bet.get("match", "Unknown"))
                league = bet.get("league", "Unknown")
                selection = bet.get("selection", "")
                odds = bet.get("odds", 0)
                bookmaker = bet.get("bookmaker", "")
                edge = bet.get("edge", 0)
                quality_score = bet.get("quality_score", 0)
                stake = bet.get("stake_recommendation", bet.get("stake", "1%"))
            else:
                match_name = bet.match_name
                league = bet.league
                selection = bet.selection
                odds = bet.odds
                bookmaker = bet.bookmaker
                edge = bet.edge
                quality_score = bet.quality_score
                stake = bet.stake_recommendation

            bet_cards += f"""
            <div class="bet-card" style="border-left: 4px solid {color};">
                <div class="rank">#{i+1}</div>
                <h3>{match_name}</h3>
                <p class="league">{league}</p>
                <div class="stats">
                    <div class="stat">
                        <span class="label">Selection</span>
                        <span class="value">{selection.upper()}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Odds</span>
                        <span class="value">{odds:.2f}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Edge</span>
                        <span class="value edge">+{edge:.1%}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Quality</span>
                        <span class="value">{quality_score:.0f}/100</span>
                    </div>
                </div>
                <div class="recommendation">
                    <strong>Stake:</strong> {stake} @ {bookmaker}
                </div>
            </div>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEXUS AI Report - {sport.upper()} {date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta {{ color: #888; font-size: 0.9rem; }}
        .bets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
        }}
        .bet-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }}
        .rank {{
            font-size: 2rem;
            font-weight: bold;
            color: #00d9ff;
            margin-bottom: 0.5rem;
        }}
        .bet-card h3 {{
            font-size: 1.2rem;
            margin-bottom: 0.25rem;
        }}
        .league {{
            color: #888;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .stat {{
            text-align: center;
        }}
        .stat .label {{
            display: block;
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
        }}
        .stat .value {{
            font-size: 1.25rem;
            font-weight: bold;
        }}
        .stat .value.edge {{
            color: #00ff88;
        }}
        .recommendation {{
            background: rgba(0,217,255,0.1);
            padding: 0.75rem;
            border-radius: 8px;
            text-align: center;
        }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            color: #666;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NEXUS AI</h1>
            <p class="meta">{sport.upper()} | {date} | Generated {datetime.now().strftime('%H:%M')}</p>
        </header>

        <div class="bets-grid">
            {bet_cards}
        </div>

        <footer>
            <p>Report generated by NEXUS AI Lite v2.0</p>
        </footer>
    </div>
</body>
</html>
"""

    def _generate_no_bets_html(self, sport: str, date: str) -> str:
        """Generate HTML report when no bets are found."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NEXUS AI Report - No Bets</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
        }}
        h1 {{ color: #ff6b6b; }}
    </style>
</head>
<body>
    <div>
        <h1>No Value Bets Found</h1>
        <p>{sport.upper()} | {date}</p>
        <p>Try again later or check a different sport.</p>
    </div>
</body>
</html>
"""

    def save_report(
        self,
        content: str,
        sport: str,
        date: str,
        format: str = "md"
    ) -> str:
        """
        Save report to file.

        Args:
            content: Report content
            sport: Sport name
            date: Analysis date
            format: "md" or "html"

        Returns:
            Path to saved file
        """
        filename = f"report_{date}_{sport}.{format}"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return str(filepath)
