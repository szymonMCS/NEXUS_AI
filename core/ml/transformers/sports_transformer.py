"""
Transformers for Sports Sequence Modeling.

Uses Transformer architecture (like BERT, GPT) for:
- Match sequence modeling
- Team form analysis
- Player trajectory prediction

Implements:
- Multi-head self-attention
- Positional encoding
- Transformer encoder blocks
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PositionalEncoding:
    """
    Positional encoding for sequence order.
    
    Adds position information to embeddings.
    """
    
    def __init__(self, d_model: int = 64, max_len: int = 100):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        self.pe = np.zeros((max_len, d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
    
    def encode(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for sequence length."""
        return self.pe[:seq_len]


class MultiHeadAttention:
    """
    Multi-head self-attention mechanism.
    
    Allows model to attend to different parts of sequence
    with different representation subspaces.
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through attention.
        
        Args:
            x: Input sequence (seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Attended output (seq_len, d_model)
        """
        seq_len = x.shape[0]
        
        # Linear projections
        Q = np.dot(x, self.W_q)  # (seq_len, d_model)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)  # (n_heads, seq_len, d_k)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = self.softmax(scores)
        
        # Apply attention to values
        attended = np.matmul(attn_weights, V)  # (n_heads, seq_len, d_k)
        
        # Concatenate heads
        attended = attended.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        
        # Final linear
        output = np.dot(attended, self.W_o)
        
        return output


class FeedForward:
    """Feed-forward network with ReLU activation."""
    
    def __init__(self, d_model: int = 64, d_ff: int = 256):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        hidden = self.relu(np.dot(x, self.W1) + self.b1)
        output = np.dot(hidden, self.W2) + self.b2
        return output


class TransformerEncoderBlock:
    """
    Single Transformer encoder block.
    
    Contains:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer norm parameters
        self.norm1_scale = np.ones(d_model)
        self.norm1_shift = np.zeros(d_model)
        self.norm2_scale = np.ones(d_model)
        self.norm2_shift = np.zeros(d_model)
    
    def layer_norm(self, x: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-6)
        return scale * normalized + shift
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through encoder block."""
        # Self-attention with residual
        attended = self.attention.forward(x)
        x = self.layer_norm(x + attended, self.norm1_scale, self.norm1_shift)
        
        # Feed-forward with residual
        ff_out = self.feed_forward.forward(x)
        x = self.layer_norm(x + ff_out, self.norm2_scale, self.norm2_shift)
        
        return x


class SportsTransformer:
    """
    Transformer model for sports match sequences.
    
    Processes sequences of matches to predict outcomes.
    For example: last 5 matches of team A vs last 5 of team B
    """
    
    def __init__(
        self,
        n_features: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 10,
    ):
        """
        Initialize Sports Transformer.
        
        Args:
            n_features: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
        """
        self.n_features = n_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = np.random.randn(n_features, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        # Output head
        self.output_head = np.random.randn(d_model, 3) * 0.1  # 3 classes: H, D, A
        self.output_bias = np.zeros(3)
    
    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """
        Forward pass through transformer.
        
        Args:
            sequence: (seq_len, n_features) - sequence of match features
            
        Returns:
            Probabilities: (3,) - home, draw, away
        """
        seq_len = sequence.shape[0]
        
        # Project input
        x = np.dot(sequence, self.input_projection)  # (seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding.encode(seq_len)
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block.forward(x)
        
        # Global average pooling
        pooled = np.mean(x, axis=0)  # (d_model,)
        
        # Output projection
        logits = np.dot(pooled, self.output_head) + self.output_bias
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def predict(self, match_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict outcome based on match history.
        
        Args:
            match_history: List of past match features
            
        Returns:
            Prediction dictionary
        """
        # Convert history to sequence
        sequence = np.array([match['features'] for match in match_history])
        
        # Ensure correct shape
        if len(sequence) > 10:
            sequence = sequence[-10:]  # Keep last 10
        elif len(sequence) < 2:
            # Pad with zeros
            padding = np.zeros((2 - len(sequence), self.n_features))
            sequence = np.vstack([padding, sequence])
        
        # Predict
        probs = self.forward(sequence)
        
        return {
            'home_win_prob': float(probs[2]),
            'draw_prob': float(probs[1]),
            'away_win_prob': float(probs[0]),
            'confidence': float(np.max(probs)),
            'predicted_outcome': ['away', 'draw', 'home'][np.argmax(probs)],
            'method': 'transformer',
            'sequence_length': len(match_history),
        }
    
    def train_step(self, sequence: np.ndarray, target: int, lr: float = 0.001):
        """
        Single training step (simplified gradient descent).
        
        In production, use PyTorch/TensorFlow for proper autodiff.
        """
        # Forward
        probs = self.forward(sequence)
        
        # Cross-entropy loss gradient (simplified)
        grad = probs.copy()
        grad[target] -= 1
        
        # Update output head (simplified)
        pooled = np.mean(np.dot(sequence, self.input_projection), axis=0)
        self.output_head -= lr * np.outer(pooled, grad)
        self.output_bias -= lr * grad


class TeamFormAnalyzer:
    """
    Analyze team form using Transformer.
    
    Processes sequence of recent matches to assess
    current team strength and momentum.
    """
    
    def __init__(self, n_features: int = 10):
        self.transformer = SportsTransformer(
            n_features=n_features,
            d_model=32,
            n_heads=2,
            n_layers=1,
        )
    
    def analyze_form(self, recent_matches: List[Dict]) -> Dict[str, Any]:
        """
        Analyze team form from recent matches.
        
        Args:
            recent_matches: List of recent match data
            
        Returns:
            Form analysis with momentum score
        """
        if len(recent_matches) < 3:
            return {
                'form_score': 0.5,
                'momentum': 'neutral',
                'confidence': 0.3,
            }
        
        # Convert to features
        features = []
        for match in recent_matches:
            feat = [
                match.get('goals_scored', 0) / 5.0,  # Normalize
                match.get('goals_conceded', 0) / 5.0,
                match.get('possession', 50) / 100.0,
                match.get('shots_on_target', 0) / 10.0,
                1.0 if match.get('result') == 'W' else 0.0 if match.get('result') == 'D' else -1.0,
            ]
            features.append(feat)
        
        # Pad to 10 features
        while len(features[0]) < 10:
            for f in features:
                f.append(0.0)
        
        # Predict trend
        sequence = np.array(features)
        trend_probs = self.transformer.forward(sequence)
        
        # Form score based on trend
        form_score = trend_probs[2] * 0.5 + trend_probs[1] * 0.25  # Weight wins higher
        
        # Momentum
        if form_score > 0.6:
            momentum = 'high'
        elif form_score > 0.4:
            momentum = 'neutral'
        else:
            momentum = 'low'
        
        return {
            'form_score': float(form_score),
            'momentum': momentum,
            'confidence': float(trend_probs[np.argmax(trend_probs)]),
            'recent_results': [m.get('result') for m in recent_matches[-5:]],
        }


# Convenience function
def create_transformer_predictor(n_features: int = 20) -> SportsTransformer:
    """Create transformer-based predictor."""
    return SportsTransformer(
        n_features=n_features,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
