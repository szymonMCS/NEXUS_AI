"""
Graph Neural Networks for Sports Team Analysis.

Models teams and players as graphs:
- Nodes: Players
- Edges: Passes, interactions, chemistry

Implements:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Message passing between players
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PlayerNode:
    """Node representing a player in the graph."""
    player_id: str
    name: str
    position: str
    features: np.ndarray  # Player stats
    team: str


@dataclass
class TeamGraph:
    """Graph representing a team."""
    team_id: str
    team_name: str
    players: List[PlayerNode]
    adjacency_matrix: np.ndarray  # Player interactions
    edge_weights: np.ndarray  # Chemistry/pass weights


class GraphConvolutionalLayer:
    """
    Graph Convolutional Layer.
    
    Propagates information through the graph using:
    H^(l+1) = activation(D^(-1/2) * A * D^(-1/2) * H^(l) * W)
    
    where:
    - A: Adjacency matrix
    - D: Degree matrix
    - H: Node features
    - W: Learnable weights
    """
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros(out_features)
    
    def normalized_adjacency(self, A: np.ndarray) -> np.ndarray:
        """
        Compute normalized adjacency matrix.
        
        A_norm = D^(-1/2) * A * D^(-1/2)
        """
        # Degree matrix
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(D_inv_sqrt)
        
        # Normalized adjacency
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        return A_norm
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Forward pass through GCN layer.
        
        Args:
            X: Node features (n_nodes, in_features)
            A: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Updated features (n_nodes, out_features)
        """
        # Normalize adjacency
        A_norm = self.normalized_adjacency(A)
        
        # Message passing
        messages = A_norm @ X  # Aggregate neighbor features
        
        # Linear transformation
        output = messages @ self.W + self.b
        
        # Activation
        output = self.relu(output)
        
        return output


class GraphAttentionLayer:
    """
    Graph Attention Layer (GAT).
    
    Learns attention weights between connected nodes:
    e_ij = LeakyReLU(a^T * [W*h_i || W*h_j])
    alpha_ij = softmax_j(e_ij)
    h_i' = sum(alpha_ij * W * h_j)
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.d_k = out_features // n_heads
        
        # Initialize weights for each head
        self.W = np.random.randn(n_heads, in_features, self.d_k) * 0.1
        self.a = np.random.randn(n_heads, 2 * self.d_k) * 0.1
    
    def leaky_relu(self, x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)
    
    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Forward pass through GAT layer.
        
        Args:
            X: Node features (n_nodes, in_features)
            A: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Updated features (n_nodes, out_features)
        """
        n_nodes = X.shape[0]
        
        # Multi-head attention
        head_outputs = []
        
        for h in range(self.n_heads):
            # Linear transformation
            Wh = X @ self.W[h]  # (n_nodes, d_k)
            
            # Compute attention scores
            e = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if A[i, j] > 0:  # Only for neighbors
                        concat = np.concatenate([Wh[i], Wh[j]])
                        e[i, j] = self.leaky_relu(np.dot(self.a[h], concat))
            
            # Apply softmax
            alpha = self.softmax(e)
            
            # Aggregate
            head_out = alpha @ Wh
            head_outputs.append(head_out)
        
        # Concatenate heads
        output = np.concatenate(head_outputs, axis=1)
        
        return output


class TeamStrengthPredictor:
    """
    Predict team strength using GNN.
    
    Analyzes player interactions and chemistry to predict
    team performance.
    """
    
    def __init__(
        self,
        player_features: int = 10,
        hidden_dim: int = 32,
        n_layers: int = 2,
        use_attention: bool = True,
    ):
        self.player_features = player_features
        self.hidden_dim = hidden_dim
        
        # GNN layers
        if use_attention:
            self.layers = [
                GraphAttentionLayer(
                    player_features if i == 0 else hidden_dim,
                    hidden_dim
                )
                for i in range(n_layers)
            ]
        else:
            self.layers = [
                GraphConvolutionalLayer(
                    player_features if i == 0 else hidden_dim,
                    hidden_dim
                )
                for i in range(n_layers)
            ]
        
        # Output head
        self.output_w = np.random.randn(hidden_dim, 1) * 0.1
        self.output_b = 0.0
    
    def forward(self, team_graph: TeamGraph) -> float:
        """
        Predict team strength score.
        
        Args:
            team_graph: Team graph with players and interactions
            
        Returns:
            Strength score (0-1)
        """
        # Get node features
        X = np.array([p.features for p in team_graph.players])
        A = team_graph.adjacency_matrix
        
        # Pass through GNN layers
        for layer in self.layers:
            X = layer.forward(X, A)
        
        # Global pooling (mean of all players)
        team_embedding = np.mean(X, axis=0)
        
        # Predict strength
        strength = float(sigmoid(np.dot(team_embedding, self.output_w) + self.output_b))
        
        return strength
    
    def predict_match(
        self,
        home_team: TeamGraph,
        away_team: TeamGraph,
    ) -> Dict[str, Any]:
        """
        Predict match outcome using team graphs.
        
        Args:
            home_team: Home team graph
            away_team: Away team graph
            
        Returns:
            Prediction with probabilities
        """
        # Get team strengths
        home_strength = self.forward(home_team)
        away_strength = self.forward(away_team)
        
        # Home advantage
        home_advantage = 0.1
        
        # Calculate probabilities using ELO-like formula
        total_strength = home_strength + away_strength + home_advantage
        
        home_prob = (home_strength + home_advantage) / total_strength
        away_prob = away_strength / total_strength
        draw_prob = 0.25  # Base draw probability
        
        # Normalize
        remaining = 1 - draw_prob
        home_prob = home_prob * remaining
        away_prob = away_prob * remaining
        
        # Determine winner
        probs = {'home': home_prob, 'draw': draw_prob, 'away': away_prob}
        predicted = max(probs, key=probs.get)
        
        return {
            'home_win_prob': home_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_prob,
            'predicted_outcome': predicted,
            'confidence': probs[predicted],
            'home_strength': home_strength,
            'away_strength': away_strength,
            'method': 'gnn_team_analysis',
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def create_team_graph(
    team_name: str,
    player_stats: List[Dict[str, Any]],
    chemistry_matrix: Optional[np.ndarray] = None,
) -> TeamGraph:
    """
    Create team graph from player statistics.
    
    Args:
        team_name: Name of team
        player_stats: List of player statistics
        chemistry_matrix: Optional chemistry matrix between players
        
    Returns:
        TeamGraph object
    """
    players = []
    for i, stats in enumerate(player_stats):
        features = np.array([
            stats.get('rating', 7.0) / 10.0,
            stats.get('goals_per_game', 0.3) / 2.0,
            stats.get('assists_per_game', 0.2) / 1.5,
            stats.get('pass_accuracy', 80) / 100.0,
            stats.get('tackles_per_game', 2.0) / 5.0,
            stats.get('minutes_played', 1800) / 3000.0,
        ])
        
        # Pad features
        while len(features) < 10:
            features = np.append(features, 0.0)
        
        player = PlayerNode(
            player_id=stats.get('id', f'p{i}'),
            name=stats.get('name', f'Player {i}'),
            position=stats.get('position', 'Unknown'),
            features=features,
            team=team_name,
        )
        players.append(player)
    
    n_players = len(players)
    
    # Create adjacency matrix
    if chemistry_matrix is not None:
        adjacency = chemistry_matrix
    else:
        # Default: connect players based on position
        adjacency = np.eye(n_players)  # Self-connections
        for i in range(n_players):
            for j in range(i + 1, n_players):
                # Higher weight for same position group
                pos_i = players[i].position
                pos_j = players[j].position
                
                if pos_i == pos_j:
                    weight = 0.3
                elif ('DEF' in pos_i and 'DEF' in pos_j) or ('MID' in pos_i and 'MID' in pos_j):
                    weight = 0.5
                elif ('FWD' in pos_i and 'MID' in pos_j) or ('MID' in pos_i and 'FWD' in pos_j):
                    weight = 0.7
                else:
                    weight = 0.4
                
                adjacency[i, j] = weight
                adjacency[j, i] = weight
    
    return TeamGraph(
        team_id=team_name.lower().replace(' ', '_'),
        team_name=team_name,
        players=players,
        adjacency_matrix=adjacency,
        edge_weights=adjacency,
    )


# Convenience function
def create_gnn_predictor() -> TeamStrengthPredictor:
    """Create GNN-based team strength predictor."""
    return TeamStrengthPredictor(
        player_features=10,
        hidden_dim=32,
        n_layers=2,
        use_attention=True,
    )
