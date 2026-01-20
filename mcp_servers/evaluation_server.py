# mcp_servers/evaluation_server.py
"""
MCP Server for data quality evaluation.
Provides tools for evaluating match data quality, calculating adjusted values,
and generating quality recommendations.
"""

import asyncio
from typing import List, Dict, Any
import json
from datetime import datetime

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, Resource, ResourceTemplate

from core.quality_scorer import QualityScorer, QualityScoreComponents
from core.value_calculator import ValueCalculator, LeagueType
from config.thresholds import thresholds, LEAGUE_REQUIREMENTS


# Create MCP server instance
server = Server("nexus-evaluation-server")


# Quality thresholds for recommendations
QUALITY_RECOMMENDATIONS = {
    "excellent": {
        "action": "STRONG_BET",
        "description": "Excellent data quality - full confidence in prediction",
        "stake_multiplier": 1.0,
        "min_edge": 0.02
    },
    "good": {
        "action": "NORMAL_BET",
        "description": "Good data quality - standard betting recommended",
        "stake_multiplier": 0.8,
        "min_edge": 0.03
    },
    "moderate": {
        "action": "REDUCED_BET",
        "description": "Moderate data quality - reduce stake size",
        "stake_multiplier": 0.5,
        "min_edge": 0.05
    },
    "high_risk": {
        "action": "SKIP_OR_MINIMAL",
        "description": "High risk - minimal bet or skip",
        "stake_multiplier": 0.25,
        "min_edge": 0.08
    },
    "insufficient": {
        "action": "SKIP",
        "description": "Insufficient data - do not bet",
        "stake_multiplier": 0.0,
        "min_edge": 1.0  # Effectively impossible
    }
}


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available evaluation tools."""
    return [
        Tool(
            name="evaluate_data_quality",
            description="Evaluate overall data quality for a match including news, odds, and stats",
            inputSchema={
                "type": "object",
                "properties": {
                    "match_id": {
                        "type": "string",
                        "description": "Unique match identifier"
                    },
                    "news_count": {
                        "type": "integer",
                        "description": "Number of news articles collected"
                    },
                    "news_avg_relevance": {
                        "type": "number",
                        "description": "Average relevance score of news (0-1)"
                    },
                    "news_fresh_count": {
                        "type": "integer",
                        "description": "Number of fresh news articles (< 24h)"
                    },
                    "odds_count": {
                        "type": "integer",
                        "description": "Number of bookmakers with odds"
                    },
                    "odds_variance": {
                        "type": "number",
                        "description": "Variance in odds across bookmakers"
                    },
                    "has_rankings": {
                        "type": "boolean",
                        "description": "Whether player/team rankings are available"
                    },
                    "has_form": {
                        "type": "boolean",
                        "description": "Whether recent form data is available"
                    },
                    "has_h2h": {
                        "type": "boolean",
                        "description": "Whether head-to-head data is available"
                    },
                    "league_type": {
                        "type": "string",
                        "description": "League classification",
                        "enum": ["popular", "medium", "unpopular"],
                        "default": "popular"
                    }
                },
                "required": ["match_id", "news_count", "odds_count"]
            }
        ),
        Tool(
            name="batch_evaluate_matches",
            description="Evaluate data quality for multiple matches at once",
            inputSchema={
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "description": "List of match data objects to evaluate",
                        "items": {
                            "type": "object",
                            "properties": {
                                "match_id": {"type": "string"},
                                "news_count": {"type": "integer"},
                                "odds_count": {"type": "integer"},
                                "has_rankings": {"type": "boolean"},
                                "has_form": {"type": "boolean"}
                            }
                        }
                    },
                    "min_quality": {
                        "type": "number",
                        "description": "Minimum quality score to include (0-1)",
                        "default": 0.45
                    }
                },
                "required": ["matches"]
            }
        ),
        Tool(
            name="get_quality_recommendation",
            description="Get betting recommendation based on quality score",
            inputSchema={
                "type": "object",
                "properties": {
                    "quality_score": {
                        "type": "number",
                        "description": "Overall quality score (0-1)"
                    },
                    "edge": {
                        "type": "number",
                        "description": "Calculated edge (e.g., 0.05 for 5%)"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Model confidence (0-1)"
                    }
                },
                "required": ["quality_score", "edge"]
            }
        ),
        Tool(
            name="calculate_adjusted_value",
            description="Calculate quality-adjusted value and stake for a bet",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimated_probability": {
                        "type": "number",
                        "description": "Estimated win probability (0-1)"
                    },
                    "bookmaker_odds": {
                        "type": "number",
                        "description": "Bookmaker decimal odds"
                    },
                    "quality_score": {
                        "type": "number",
                        "description": "Data quality score (0-1)"
                    },
                    "bankroll": {
                        "type": "number",
                        "description": "Current bankroll",
                        "default": 1000.0
                    },
                    "league_type": {
                        "type": "string",
                        "description": "League classification",
                        "enum": ["popular", "medium", "unpopular"],
                        "default": "popular"
                    }
                },
                "required": ["estimated_probability", "bookmaker_odds", "quality_score"]
            }
        ),
        Tool(
            name="check_source_agreement",
            description="Check agreement between multiple data sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "description": "List of predictions from different sources",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "probability": {"type": "number"},
                                "confidence": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["predictions"]
            }
        ),
        Tool(
            name="get_quality_thresholds",
            description="Get current quality thresholds and requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "league_type": {
                        "type": "string",
                        "description": "League classification",
                        "enum": ["popular", "medium", "unpopular"],
                        "default": "popular"
                    }
                }
            }
        ),
        Tool(
            name="generate_quality_report",
            description="Generate detailed quality report for a match",
            inputSchema={
                "type": "object",
                "properties": {
                    "match_id": {
                        "type": "string",
                        "description": "Match identifier"
                    },
                    "match_name": {
                        "type": "string",
                        "description": "Human-readable match name"
                    },
                    "quality_components": {
                        "type": "object",
                        "description": "Quality score components",
                        "properties": {
                            "news_score": {"type": "number"},
                            "odds_score": {"type": "number"},
                            "stats_score": {"type": "number"},
                            "overall_score": {"type": "number"}
                        }
                    },
                    "issues": {
                        "type": "array",
                        "description": "List of identified issues",
                        "items": {"type": "string"}
                    }
                },
                "required": ["match_id", "match_name", "quality_components"]
            }
        ),
    ]


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="evaluation://thresholds",
            name="Quality Thresholds",
            description="Current quality thresholds configuration",
            mimeType="application/json"
        ),
        Resource(
            uri="evaluation://recommendations",
            name="Betting Recommendations",
            description="Quality-based betting recommendations",
            mimeType="application/json"
        )
    ]


@server.list_resource_templates()
async def list_resource_templates() -> List[ResourceTemplate]:
    """List available resource templates."""
    return [
        ResourceTemplate(
            uriTemplate="evaluation://{match_id}/report",
            name="Match Quality Report",
            description="Detailed quality report for a specific match",
            mimeType="application/json"
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute an evaluation tool."""

    if name == "evaluate_data_quality":
        result = await _evaluate_data_quality(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "batch_evaluate_matches":
        result = await _batch_evaluate_matches(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_quality_recommendation":
        result = _get_quality_recommendation(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "calculate_adjusted_value":
        result = _calculate_adjusted_value(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "check_source_agreement":
        result = _check_source_agreement(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_quality_thresholds":
        result = _get_quality_thresholds(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "generate_quality_report":
        result = _generate_quality_report(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
        )]


async def _evaluate_data_quality(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate overall data quality for a match."""
    league_type = args.get("league_type", "popular")
    requirements = LEAGUE_REQUIREMENTS.get(league_type, LEAGUE_REQUIREMENTS["popular"])

    # Calculate component scores
    news_score = _calculate_news_score(
        news_count=args.get("news_count", 0),
        avg_relevance=args.get("news_avg_relevance", 0.5),
        fresh_count=args.get("news_fresh_count", args.get("news_count", 0)),
        min_articles=requirements.min_news_articles
    )

    odds_score = _calculate_odds_score(
        odds_count=args.get("odds_count", 0),
        variance=args.get("odds_variance", 0.05),
        min_bookmakers=requirements.min_bookmakers
    )

    stats_score = _calculate_stats_score(
        has_rankings=args.get("has_rankings", False),
        has_form=args.get("has_form", False),
        has_h2h=args.get("has_h2h", False)
    )

    # Weighted overall score
    weights = {"news": 0.3, "odds": 0.4, "stats": 0.3}
    overall_score = (
        news_score * weights["news"] +
        odds_score * weights["odds"] +
        stats_score * weights["stats"]
    )

    # Determine quality level
    quality_level = _get_quality_level(overall_score)

    # Identify issues
    issues = []
    if args.get("news_count", 0) < requirements.min_news_articles:
        issues.append(f"Insufficient news ({args.get('news_count', 0)}/{requirements.min_news_articles})")
    if args.get("odds_count", 0) < requirements.min_bookmakers:
        issues.append(f"Insufficient odds sources ({args.get('odds_count', 0)}/{requirements.min_bookmakers})")
    if not args.get("has_rankings"):
        issues.append("Missing rankings data")
    if not args.get("has_form"):
        issues.append("Missing form data")

    return {
        "status": "success",
        "match_id": args.get("match_id"),
        "quality": {
            "news_score": round(news_score, 3),
            "odds_score": round(odds_score, 3),
            "stats_score": round(stats_score, 3),
            "overall_score": round(overall_score, 3),
            "quality_level": quality_level,
            "issues": issues
        },
        "recommendation": QUALITY_RECOMMENDATIONS.get(quality_level, QUALITY_RECOMMENDATIONS["insufficient"]),
        "timestamp": datetime.now().isoformat()
    }


async def _batch_evaluate_matches(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate multiple matches at once."""
    matches = args.get("matches", [])
    min_quality = args.get("min_quality", 0.45)

    results = []
    passed = []
    failed = []

    for match in matches:
        eval_result = await _evaluate_data_quality(match)
        score = eval_result["quality"]["overall_score"]

        results.append({
            "match_id": match.get("match_id"),
            "quality_score": score,
            "quality_level": eval_result["quality"]["quality_level"],
            "passed": score >= min_quality
        })

        if score >= min_quality:
            passed.append(match.get("match_id"))
        else:
            failed.append(match.get("match_id"))

    return {
        "status": "success",
        "total_matches": len(matches),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "min_quality_threshold": min_quality,
        "passed_matches": passed,
        "failed_matches": failed,
        "detailed_results": results
    }


def _get_quality_recommendation(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get betting recommendation based on quality."""
    quality_score = args.get("quality_score", 0)
    edge = args.get("edge", 0)
    confidence = args.get("confidence", 0.5)

    quality_level = _get_quality_level(quality_score)
    rec = QUALITY_RECOMMENDATIONS.get(quality_level, QUALITY_RECOMMENDATIONS["insufficient"])

    # Check if edge meets minimum requirement
    min_edge = rec["min_edge"]
    edge_sufficient = edge >= min_edge

    # Final action
    if not edge_sufficient:
        action = "SKIP"
        reason = f"Edge ({edge:.1%}) below minimum ({min_edge:.1%}) for {quality_level} quality"
    else:
        action = rec["action"]
        reason = rec["description"]

    return {
        "status": "success",
        "quality_score": quality_score,
        "quality_level": quality_level,
        "edge": edge,
        "min_edge_required": min_edge,
        "edge_sufficient": edge_sufficient,
        "confidence": confidence,
        "action": action,
        "reason": reason,
        "stake_multiplier": rec["stake_multiplier"] if edge_sufficient else 0
    }


def _calculate_adjusted_value(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate quality-adjusted value and stake."""
    prob = args.get("estimated_probability", 0.5)
    odds = args.get("bookmaker_odds", 2.0)
    quality = args.get("quality_score", 0.5)
    bankroll = args.get("bankroll", 1000.0)
    league_type_str = args.get("league_type", "popular")

    # Map to LeagueType
    league_map = {
        "popular": LeagueType.POPULAR,
        "medium": LeagueType.MEDIUM,
        "unpopular": LeagueType.UNPOPULAR
    }
    league_type = league_map.get(league_type_str, LeagueType.POPULAR)

    # Use ValueCalculator
    calc = ValueCalculator(bankroll=bankroll)

    # Raw edge
    raw_edge = calc.calculate_edge(prob, odds)

    # Quality-adjusted probability (move towards 0.5 for low quality)
    adjusted_prob = calc.adjust_probability_for_quality(prob, quality)

    # Quality-adjusted edge
    quality_multiplier = calc.get_quality_multiplier(quality)
    adjusted_edge = raw_edge * quality_multiplier

    # Check minimum edge
    min_edge = calc.MIN_EDGE.get(league_type, 0.03)
    has_value = adjusted_edge >= min_edge

    # Kelly stake
    kelly_fraction = calc.calculate_kelly_stake(adjusted_prob, odds) if has_value else 0
    adjusted_kelly = kelly_fraction * quality_multiplier

    # Recommended stake
    stake_pct = min(adjusted_kelly, calc.MAX_STAKE_PCT)
    recommended_stake = bankroll * stake_pct

    return {
        "status": "success",
        "input": {
            "probability": prob,
            "odds": odds,
            "quality_score": quality,
            "bankroll": bankroll,
            "league_type": league_type_str
        },
        "calculations": {
            "raw_edge": round(raw_edge, 4),
            "adjusted_probability": round(adjusted_prob, 4),
            "quality_multiplier": quality_multiplier,
            "adjusted_edge": round(adjusted_edge, 4),
            "min_edge_required": min_edge,
            "has_value": has_value
        },
        "stake": {
            "kelly_fraction": round(kelly_fraction, 4),
            "adjusted_kelly": round(adjusted_kelly, 4),
            "stake_percentage": round(stake_pct * 100, 2),
            "recommended_stake": round(recommended_stake, 2)
        },
        "fair_odds": round(calc.calculate_fair_odds(prob), 2),
        "implied_probability": round(calc.calculate_implied_probability(odds), 4)
    }


def _check_source_agreement(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check agreement between prediction sources."""
    predictions = args.get("predictions", [])

    if not predictions:
        return {"status": "error", "message": "No predictions provided"}

    # Extract probabilities
    probs = [p.get("probability", 0.5) for p in predictions]
    confidences = [p.get("confidence", 0.5) for p in predictions]
    sources = [p.get("source", "unknown") for p in predictions]

    # Calculate statistics
    avg_prob = sum(probs) / len(probs)
    avg_confidence = sum(confidences) / len(confidences)

    # Variance (disagreement)
    variance = sum((p - avg_prob) ** 2 for p in probs) / len(probs)
    std_dev = variance ** 0.5

    # Agreement score (lower variance = higher agreement)
    max_variance = 0.25  # Maximum theoretical variance
    agreement_score = max(0, 1 - (variance / max_variance))

    # Determine consensus
    if std_dev < 0.05:
        consensus = "strong"
        consensus_description = "Sources strongly agree"
    elif std_dev < 0.10:
        consensus = "moderate"
        consensus_description = "Sources moderately agree"
    elif std_dev < 0.15:
        consensus = "weak"
        consensus_description = "Sources weakly agree"
    else:
        consensus = "none"
        consensus_description = "Sources disagree significantly"

    return {
        "status": "success",
        "source_count": len(predictions),
        "sources": sources,
        "probabilities": probs,
        "statistics": {
            "average_probability": round(avg_prob, 4),
            "average_confidence": round(avg_confidence, 4),
            "variance": round(variance, 4),
            "std_deviation": round(std_dev, 4)
        },
        "agreement": {
            "score": round(agreement_score, 3),
            "level": consensus,
            "description": consensus_description
        }
    }


def _get_quality_thresholds(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get quality thresholds configuration."""
    league_type = args.get("league_type", "popular")
    requirements = LEAGUE_REQUIREMENTS.get(league_type, LEAGUE_REQUIREMENTS["popular"])

    return {
        "status": "success",
        "league_type": league_type,
        "requirements": {
            "min_news_articles": requirements.min_news_articles,
            "min_bookmakers": requirements.min_bookmakers,
            "min_quality_score": requirements.min_quality_score
        },
        "global_thresholds": {
            "quality_excellent": thresholds.quality_excellent,
            "quality_good": thresholds.quality_good,
            "quality_moderate": thresholds.quality_moderate,
            "quality_high_risk": thresholds.quality_high_risk,
            "min_confidence": thresholds.min_confidence_threshold,
            "min_edge": thresholds.min_edge_threshold,
            "max_odds_variance": thresholds.max_odds_variance,
            "news_freshness_hours": thresholds.news_freshness_hours
        },
        "recommendations": QUALITY_RECOMMENDATIONS
    }


def _generate_quality_report(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed quality report."""
    match_id = args.get("match_id")
    match_name = args.get("match_name", match_id)
    components = args.get("quality_components", {})
    issues = args.get("issues", [])

    overall_score = components.get("overall_score", 0)
    quality_level = _get_quality_level(overall_score)
    rec = QUALITY_RECOMMENDATIONS.get(quality_level, QUALITY_RECOMMENDATIONS["insufficient"])

    # Generate report text
    report_lines = [
        f"# Quality Report: {match_name}",
        f"Match ID: {match_id}",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Quality Scores",
        f"- News Score: {components.get('news_score', 0):.1%}",
        f"- Odds Score: {components.get('odds_score', 0):.1%}",
        f"- Stats Score: {components.get('stats_score', 0):.1%}",
        f"- **Overall Score: {overall_score:.1%}**",
        f"- Quality Level: {quality_level.upper()}",
        "",
        "## Recommendation",
        f"- Action: {rec['action']}",
        f"- {rec['description']}",
        f"- Stake Multiplier: {rec['stake_multiplier']}",
        f"- Minimum Edge Required: {rec['min_edge']:.1%}",
    ]

    if issues:
        report_lines.extend([
            "",
            "## Issues Identified",
        ])
        for issue in issues:
            report_lines.append(f"- {issue}")

    return {
        "status": "success",
        "match_id": match_id,
        "match_name": match_name,
        "quality_level": quality_level,
        "overall_score": overall_score,
        "recommendation": rec,
        "issues": issues,
        "report_text": "\n".join(report_lines)
    }


# Helper functions
def _calculate_news_score(news_count: int, avg_relevance: float,
                          fresh_count: int, min_articles: int) -> float:
    """Calculate news quality score."""
    if news_count == 0:
        return 0.0

    # Quantity (0-0.4)
    quantity = min(news_count / (min_articles * 2), 1.0) * 0.4

    # Freshness (0-0.3)
    freshness = min(fresh_count / max(min_articles, 1), 1.0) * 0.3

    # Relevance (0-0.3)
    relevance = avg_relevance * 0.3

    return min(quantity + freshness + relevance, 1.0)


def _calculate_odds_score(odds_count: int, variance: float,
                          min_bookmakers: int) -> float:
    """Calculate odds quality score."""
    if odds_count == 0:
        return 0.0

    # Quantity (0-0.5)
    quantity = min(odds_count / (min_bookmakers * 2), 1.0) * 0.5

    # Consistency (0-0.3) - lower variance is better
    max_variance = thresholds.max_odds_variance
    consistency = max(1.0 - (variance / max_variance), 0.0) * 0.3

    # Recency bonus (0-0.2) - assume recent if count > 0
    recency = 0.2 if odds_count >= min_bookmakers else 0.1

    return min(quantity + consistency + recency, 1.0)


def _calculate_stats_score(has_rankings: bool, has_form: bool,
                           has_h2h: bool) -> float:
    """Calculate stats quality score."""
    score = 0.0

    if has_rankings:
        score += 0.4
    if has_form:
        score += 0.35
    if has_h2h:
        score += 0.25

    return min(score, 1.0)


def _get_quality_level(score: float) -> str:
    """Map score to quality level."""
    if score >= thresholds.quality_excellent:
        return "excellent"
    elif score >= thresholds.quality_good:
        return "good"
    elif score >= thresholds.quality_moderate:
        return "moderate"
    elif score >= thresholds.quality_high_risk:
        return "high_risk"
    else:
        return "insufficient"


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="nexus-evaluation-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
