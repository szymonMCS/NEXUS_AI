"""
MCP Server for ML Predictions.
Exposes prediction models as tools for LangGraph agents.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP not installed. Prediction server unavailable.")

from core.ml.registry import get_model_registry, ModelRegistry
from core.ml.models.base import PredictionResult
from core.ml.training.online_trainer import get_online_trainer


class MCPPredictionServer:
    """
    MCP Server exposing prediction capabilities.
    Agents can call tools to get predictions from ML models.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
        self.server = None
        
        if MCP_AVAILABLE:
            self.server = Server("nexus-prediction-server")
            self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools."""
        if not self.server:
            return
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available prediction tools."""
            return [
                Tool(
                    name="predict_match",
                    description="Predict outcome of a sports match using ML models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sport": {
                                "type": "string",
                                "description": "Sport type (tennis, basketball, etc.)"
                            },
                            "home_player": {
                                "type": "string",
                                "description": "Home player or team name"
                            },
                            "away_player": {
                                "type": "string",
                                "description": "Away player or team name"
                            },
                            "features": {
                                "type": "object",
                                "description": "Optional match features (ranking, form, etc.)"
                            }
                        },
                        "required": ["sport", "home_player", "away_player"]
                    }
                ),
                Tool(
                    name="get_model_info",
                    description="Get information about available prediction models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sport": {
                                "type": "string",
                                "description": "Sport type (optional, returns all if not specified)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_feature_importance",
                    description="Get feature importance for model explainability",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sport": {
                                "type": "string",
                                "description": "Sport type"
                            },
                            "model_name": {
                                "type": "string",
                                "description": "Model name (optional, uses best if not specified)"
                            }
                        },
                        "required": ["sport"]
                    }
                ),
                Tool(
                    name="train_models",
                    description="Trigger model training for a sport",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sport": {
                                "type": "string",
                                "description": "Sport type to train"
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force training even if not needed"
                            }
                        },
                        "required": ["sport"]
                    }
                ),
                Tool(
                    name="batch_predict",
                    description="Predict outcomes for multiple matches",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sport": {
                                "type": "string",
                                "description": "Sport type"
                            },
                            "matches": {
                                "type": "array",
                                "description": "List of matches with features",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "home_player": {"type": "string"},
                                        "away_player": {"type": "string"},
                                        "features": {"type": "object"}
                                    },
                                    "required": ["home_player", "away_player"]
                                }
                            }
                        },
                        "required": ["sport", "matches"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "predict_match":
                    return await self._handle_predict(arguments)
                elif name == "get_model_info":
                    return await self._handle_model_info(arguments)
                elif name == "get_feature_importance":
                    return await self._handle_feature_importance(arguments)
                elif name == "train_models":
                    return await self._handle_train(arguments)
                elif name == "batch_predict":
                    return await self._handle_batch_predict(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_predict(self, arguments: Dict) -> List[TextContent]:
        """Handle predict_match tool."""
        sport = arguments.get("sport")
        home_player = arguments.get("home_player")
        away_player = arguments.get("away_player")
        features = arguments.get("features", {})
        
        # Add player names to features
        features['home_player'] = home_player
        features['away_player'] = away_player
        features['home_team'] = home_player  # Alias
        features['away_team'] = away_player  # Alias
        
        # Load best model
        model = self.registry.get_best_model(sport)
        
        if not model:
            return [TextContent(type="text", text=f"No model available for {sport}")]
        
        # Predict
        result = model.predict(features)
        
        # Format response
        response = {
            "prediction": {
                "home_win_probability": round(result.home_win_prob, 3),
                "away_win_probability": round(result.away_win_prob, 3),
                "confidence": round(result.confidence, 3),
                "model_used": result.model_name,
                "model_version": result.version
            },
            "analysis": {
                "favorite": "home" if result.home_win_prob > result.away_win_prob else "away",
                "probability_spread": round(abs(result.home_win_prob - result.away_win_prob), 3),
                "reliability": "high" if result.confidence > 0.7 else "medium" if result.confidence > 0.5 else "low"
            }
        }
        
        if result.draw_prob:
            response["prediction"]["draw_probability"] = round(result.draw_prob, 3)
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_model_info(self, arguments: Dict) -> List[TextContent]:
        """Handle get_model_info tool."""
        sport = arguments.get("sport")
        
        models = self.registry.list_models(sport)
        
        response = {}
        for s, model_list in models.items():
            response[s] = [
                {
                    "name": m.name,
                    "version": m.version,
                    "type": m.model_type,
                    "accuracy": m.accuracy,
                    "created": m.created_at.isoformat() if m.created_at else None,
                    "active": m.is_active
                }
                for m in model_list
            ]
        
        if sport and sport in response:
            return [TextContent(type="text", text=json.dumps({sport: response[sport]}, indent=2))]
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_feature_importance(self, arguments: Dict) -> List[TextContent]:
        """Handle get_feature_importance tool."""
        sport = arguments.get("sport")
        model_name = arguments.get("model_name")
        
        if model_name:
            model = self.registry.load(sport, model_name)
        else:
            model = self.registry.get_best_model(sport)
        
        if not model:
            return [TextContent(type="text", text=f"Model not found for {sport}")]
        
        importance = model.get_feature_importance()
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        response = {
            "model": model.name,
            "version": model.version,
            "features": [
                {"name": name, "importance": round(imp, 4)}
                for name, imp in sorted_features
            ]
        }
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_train(self, arguments: Dict) -> List[TextContent]:
        """Handle train_models tool."""
        sport = arguments.get("sport")
        force = arguments.get("force", False)
        
        trainer = get_online_trainer()
        
        if not force:
            should_train = await trainer.should_retrain(sport)
            if not should_train:
                return [TextContent(type="text", text=f"Training not needed for {sport}. Use force=true to override.")]
        
        # Run training
        await trainer.run_training_cycle(sport)
        
        response = {
            "status": "training_complete",
            "sport": sport,
            "timestamp": datetime.now().isoformat(),
            "metrics": trainer.get_training_report()
        }
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_batch_predict(self, arguments: Dict) -> List[TextContent]:
        """Handle batch_predict tool."""
        sport = arguments.get("sport")
        matches = arguments.get("matches", [])
        
        model = self.registry.get_best_model(sport)
        
        if not model:
            return [TextContent(type="text", text=f"No model available for {sport}")]
        
        # Prepare features
        features_list = []
        for match in matches:
            f = match.get("features", {})
            f["home_player"] = match["home_player"]
            f["away_player"] = match["away_player"]
            f["home_team"] = match["home_player"]
            f["away_team"] = match["away_player"]
            features_list.append(f)
        
        # Batch predict
        results = model.predict_batch(features_list)
        
        response = {
            "predictions": [
                {
                    "match_index": i,
                    "home_player": m["home_player"],
                    "away_player": m["away_player"],
                    "home_win_probability": round(r.home_win_prob, 3),
                    "away_win_probability": round(r.away_win_prob, 3),
                    "confidence": round(r.confidence, 3),
                    "model": r.model_name
                }
                for i, (m, r) in enumerate(zip(matches, results))
            ]
        }
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def run(self, transport="stdio"):
        """Run the MCP server."""
        if not MCP_AVAILABLE or not self.server:
            print("MCP not available, server cannot start")
            return
        
        print("Starting MCP Prediction Server...")
        
        if transport == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        else:
            # SSE transport would go here
            pass


# Simple wrapper for direct use (without MCP)
class PredictionService:
    """
    Direct prediction service for use by agents.
    Wraps model registry with additional logic.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or get_model_registry()
    
    def predict(self, 
                sport: str,
                home_player: str,
                away_player: str,
                features: Optional[Dict] = None) -> PredictionResult:
        """
        Get prediction for a match.
        
        Args:
            sport: Sport type
            home_player: Home player name
            away_player: Away player name
            features: Optional additional features
            
        Returns:
            PredictionResult
        """
        if features is None:
            features = {}
        
        features['home_player'] = home_player
        features['away_player'] = away_player
        features['home_team'] = home_player
        features['away_team'] = away_player
        
        # Get best model
        model = self.registry.get_best_model(sport)
        
        if not model:
            # Fallback to 50/50
            return PredictionResult(
                home_win_prob=0.5,
                away_win_prob=0.5,
                confidence=0.0,
                model_name="fallback"
            )
        
        return model.predict(features)
    
    def predict_with_explanation(self,
                                 sport: str,
                                 home_player: str,
                                 away_player: str,
                                 features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get prediction with detailed explanation.
        """
        result = self.predict(sport, home_player, away_player, features)
        
        # Get model for feature importance
        model = self.registry.get_best_model(sport)
        importance = model.get_feature_importance() if model else {}
        
        # Determine recommendation
        if result.home_win_prob > result.away_win_prob:
            selection = "home"
            selection_prob = result.home_win_prob
        else:
            selection = "away"
            selection_prob = result.away_win_prob
        
        return {
            "prediction": {
                "home_win_probability": round(result.home_win_prob, 3),
                "away_win_probability": round(result.away_win_prob, 3),
                "confidence": round(result.confidence, 3),
                "selection": selection,
                "selection_probability": round(selection_prob, 3)
            },
            "model_info": {
                "name": result.model_name,
                "version": result.version,
                "type": type(model).__name__ if model else "unknown"
            },
            "factors": {
                "top_features": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5] if importance else [],
                "confidence_level": "high" if result.confidence > 0.7 else "medium" if result.confidence > 0.5 else "low"
            },
            "reasoning": self._generate_reasoning(result, features or {}, importance)
        }
    
    def _generate_reasoning(self, 
                           result: PredictionResult, 
                           features: Dict,
                           importance: Dict) -> List[str]:
        """Generate human-readable reasoning."""
        reasoning = []
        
        # Based on probability
        prob_diff = abs(result.home_win_prob - result.away_win_prob)
        if prob_diff > 0.3:
            reasoning.append(f"Strong favorite identified ({prob_diff:.1%} probability advantage)")
        elif prob_diff > 0.15:
            reasoning.append(f"Moderate favorite identified ({prob_diff:.1%} advantage)")
        else:
            reasoning.append("Close match - minimal advantage detected")
        
        # Based on confidence
        if result.confidence > 0.7:
            reasoning.append("High confidence prediction based on strong data signals")
        elif result.confidence > 0.5:
            reasoning.append("Moderate confidence - adequate data available")
        else:
            reasoning.append("Low confidence due to limited data")
        
        # Based on features
        if features.get('home_rank') and features.get('away_rank'):
            home_rank = features['home_rank']
            away_rank = features['away_rank']
            if home_rank < away_rank:
                reasoning.append(f"Home player has better ranking (#{home_rank} vs #{away_rank})")
            elif away_rank < home_rank:
                reasoning.append(f"Away player has better ranking (#{away_rank} vs #{home_rank})")
        
        # Based on feature importance
        if importance:
            top_feature = max(importance.items(), key=lambda x: x[1])
            reasoning.append(f"Key factor: {top_feature[0]} ({top_feature[1]:.1%} influence)")
        
        return reasoning
    
    def batch_predict(self,
                     sport: str,
                     matches: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Predict multiple matches."""
        model = self.registry.get_best_model(sport)
        
        if not model:
            return [PredictionResult(0.5, 0.5, confidence=0.0) for _ in matches]
        
        features_list = []
        for match in matches:
            f = match.get("features", {})
            f["home_player"] = match.get("home_player") or match.get("home_team")
            f["away_player"] = match.get("away_player") or match.get("away_team")
            f["home_team"] = match.get("home_player") or match.get("home_team")
            f["away_team"] = match.get("away_player") or match.get("away_team")
            features_list.append(f)
        
        return model.predict_batch(features_list)


# Singleton
_service = None

def get_prediction_service() -> PredictionService:
    """Get singleton prediction service."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
