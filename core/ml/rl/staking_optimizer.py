"""
Reinforcement Learning for Staking Optimization.

Optimizes betting stake sizes using:
- Q-Learning
- Policy Gradient
- Multi-Armed Bandit approaches

Maximizes long-term bankroll growth with Kelly Criterion
and risk management.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class BetAction:
    """Action representing betting decision."""
    stake_fraction: float  # 0 to 1 (fraction of bankroll)
    selection: str  # 'home', 'draw', 'away'
    confidence: float  # Model confidence


@dataclass
class BetState:
    """State representation for RL."""
    bankroll: float
    recent_performance: float  # Win rate last 10 bets
    current_streak: int  # Positive for wins, negative for losses
    model_confidence: float
    odds: float
    edge: float
    day_of_week: int
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.bankroll / 10000.0,  # Normalize
            self.recent_performance,
            self.current_streak / 10.0,  # Normalize
            self.model_confidence,
            self.odds / 10.0,  # Normalize
            self.edge,
            self.day_of_week / 7.0,
        ])


class KellyCriterionOptimizer:
    """
    Kelly Criterion with RL-based adjustments.
    
    f* = (bp - q) / b
    
    where:
    - b = odds - 1
    - p = probability of winning
    - q = 1 - p
    """
    
    def __init__(self, fraction: float = 0.25, max_bet: float = 0.05):
        """
        Initialize Kelly optimizer.
        
        Args:
            fraction: Kelly fraction (usually 0.25 for safety)
            max_bet: Maximum bet as fraction of bankroll
        """
        self.fraction = fraction
        self.max_bet = max_bet
        self.bet_history: List[Dict] = []
    
    def calculate_stake(
        self,
        probability: float,
        odds: float,
        bankroll: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.
        
        Args:
            probability: Estimated probability of winning
            odds: Decimal odds
            bankroll: Current bankroll
            confidence: Confidence in probability estimate (0-1)
            
        Returns:
            Recommended stake amount
        """
        if odds <= 1 or probability <= 0:
            return 0.0
        
        b = odds - 1  # Net odds
        p = probability
        q = 1 - p
        
        # Kelly fraction
        kelly = (b * p - q) / b
        
        # Apply safety fraction
        kelly = kelly * self.fraction
        
        # Adjust by confidence
        kelly = kelly * confidence
        
        # Cap at maximum
        kelly = min(kelly, self.max_bet)
        
        # Ensure non-negative
        kelly = max(0, kelly)
        
        stake = kelly * bankroll
        
        return stake
    
    def update_from_result(self, stake: float, profit: float, odds: float):
        """Update history with bet result."""
        self.bet_history.append({
            'stake': stake,
            'profit': profit,
            'odds': odds,
            'won': profit > 0,
        })
        
        # Keep last 100
        self.bet_history = self.bet_history[-100:]


class QLearningStakingAgent:
    """
    Q-Learning agent for stake optimization.
    
    Learns optimal stake sizes through experience:
    Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state -> action values
        self.q_table: Dict[Tuple, np.ndarray] = {}
        
        # Discretization bins
        self.stake_bins = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
        self.n_actions = len(self.stake_bins)
    
    def _discretize_state(self, state: BetState) -> Tuple:
        """Discretize continuous state."""
        return (
            round(state.recent_performance * 10),  # 0-10
            int(np.sign(state.current_streak) * min(abs(state.current_streak), 5)),
            round(state.model_confidence * 5),  # 0-5
            round(state.edge * 10),  # Discretize edge
        )
    
    def get_action(self, state: BetState) -> float:
        """
        Select action using epsilon-greedy policy.
        
        Returns:
            Stake fraction
        """
        s = self._discretize_state(state)
        
        # Initialize Q-values if new state
        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.n_actions)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # Explore
            action_idx = np.random.randint(self.n_actions)
        else:
            # Exploit
            action_idx = np.argmax(self.q_table[s])
        
        return self.stake_bins[action_idx]
    
    def update(
        self,
        state: BetState,
        action: float,
        reward: float,
        next_state: BetState,
    ):
        """
        Update Q-values using Bellman equation.
        
        Args:
            state: Current state
            action: Action taken (stake fraction)
            reward: Reward received (profit normalized)
            next_state: Next state
        """
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        
        action_idx = self.stake_bins.index(action)
        
        # Initialize if needed
        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.n_actions)
        if s_next not in self.q_table:
            self.q_table[s_next] = np.zeros(self.n_actions)
        
        # Q-learning update
        current_q = self.q_table[s][action_idx]
        max_next_q = np.max(self.q_table[s_next])
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[s][action_idx] = new_q
    
    def save(self, path: str):
        """Save Q-table."""
        # Convert numpy arrays to lists for JSON
        q_data = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(path, 'w') as f:
            json.dump(q_data, f)
    
    def load(self, path: str):
        """Load Q-table."""
        with open(path, 'r') as f:
            q_data = json.load(f)
        self.q_table = {eval(k): np.array(v) for k, v in q_data.items()}


class PolicyGradientAgent:
    """
    Policy Gradient (REINFORCE) for stake optimization.
    
    Directly learns policy π(a|s) using gradient ascent.
    """
    
    def __init__(
        self,
        state_dim: int = 7,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Neural network weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = 0.0
        
        self.lr = learning_rate
        
        # Memory for episode
        self.states: List[np.ndarray] = []
        self.actions: List[float] = []
        self.rewards: List[float] = []
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Forward pass to get mean and std of stake.
        
        Returns:
            mean, std for Gaussian policy
        """
        hidden = self.relu(np.dot(state, self.W1) + self.b1)
        mean = np.tanh(np.dot(hidden, self.W2) + self.b2) * 0.05  # Scale to 0-5%
        std = 0.01  # Fixed std for exploration
        
        return float(mean), std
    
    def select_action(self, state: BetState) -> float:
        """Sample action from policy."""
        mean, std = self.forward(state.to_array())
        
        # Sample from Gaussian
        action = np.random.normal(mean, std)
        action = np.clip(action, 0.0, 0.1)  # Clip to valid range
        
        return action
    
    def store_transition(self, state: BetState, action: float, reward: float):
        """Store transition for episode."""
        self.states.append(state.to_array())
        self.actions.append(action)
        self.rewards.append(reward)
    
    def update(self):
        """Update policy using REINFORCE."""
        if len(self.states) == 0:
            return
        
        # Calculate returns (cumulative rewards)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Policy gradient update (simplified)
        for s, a, G in zip(self.states, self.actions, returns):
            # Forward
            hidden = self.relu(np.dot(s, self.W1) + self.b1)
            mean = np.tanh(np.dot(hidden, self.W2) + self.b2) * 0.05
            
            # Gradient (simplified - in practice use autodiff)
            grad = G * (a - mean)
            
            # Update (very simplified)
            self.W2 += self.lr * grad * hidden.reshape(-1, 1)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []


class StakingOptimizer:
    """
    Main staking optimizer combining multiple approaches.
    
    Uses:
    1. Kelly Criterion as baseline
    2. Q-Learning for adjustments
    3. Risk management rules
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        use_rl: bool = True,
        kelly_fraction: float = 0.25,
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.use_rl = use_rl
        
        # Components
        self.kelly = KellyCriterionOptimizer(fraction=kelly_fraction)
        self.q_agent = QLearningStakingAgent() if use_rl else None
        
        # Tracking
        self.bet_count = 0
        self.win_count = 0
        self.total_profit = 0.0
        self.peak_bankroll = initial_bankroll
        self.max_drawdown = 0.0
    
    def optimize_stake(
        self,
        prediction_prob: float,
        odds: float,
        model_confidence: float,
        recent_win_rate: float = 0.5,
        current_streak: int = 0,
    ) -> Dict[str, Any]:
        """
        Optimize stake size for a bet.
        
        Args:
            prediction_prob: Predicted probability
            odds: Decimal odds
            model_confidence: Confidence in prediction
            recent_win_rate: Win rate in recent bets
            current_streak: Current win/loss streak
            
        Returns:
            Staking recommendation
        """
        # Calculate edge
        implied_prob = 1 / odds
        edge = prediction_prob - implied_prob
        
        if edge <= 0:
            return {
                'recommendation': 'NO_BET',
                'stake': 0.0,
                'stake_fraction': 0.0,
                'reason': 'No positive edge',
            }
        
        # Base Kelly stake
        kelly_stake = self.kelly.calculate_stake(
            prediction_prob,
            odds,
            self.current_bankroll,
            model_confidence,
        )
        
        # RL adjustment
        if self.use_rl and self.q_agent:
            state = BetState(
                bankroll=self.current_bankroll,
                recent_performance=recent_win_rate,
                current_streak=current_streak,
                model_confidence=model_confidence,
                odds=odds,
                edge=edge,
                day_of_week=0,  # Simplified
            )
            
            rl_fraction = self.q_agent.get_action(state)
            rl_stake = rl_fraction * self.current_bankroll
            
            # Combine Kelly and RL
            final_stake = 0.7 * kelly_stake + 0.3 * rl_stake
        else:
            final_stake = kelly_stake
        
        # Risk management: reduce stake during drawdown
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        if drawdown > 0.2:  # 20% drawdown
            final_stake *= 0.5  # Reduce by half
        
        # Final bounds
        final_stake = max(0, min(final_stake, self.current_bankroll * 0.05))
        
        stake_fraction = final_stake / self.current_bankroll if self.current_bankroll > 0 else 0
        
        return {
            'recommendation': 'BET' if final_stake > 0 else 'NO_BET',
            'stake': round(final_stake, 2),
            'stake_fraction': round(stake_fraction, 4),
            'kelly_stake': round(kelly_stake, 2),
            'edge': round(edge, 4),
            'expected_value': round(edge * final_stake, 2),
            'method': 'kelly_rl_hybrid' if self.use_rl else 'kelly',
        }
    
    def update_after_bet(self, stake: float, profit: float, odds: float):
        """Update state after bet result."""
        self.bet_count += 1
        if profit > 0:
            self.win_count += 1
        
        self.current_bankroll += profit
        self.total_profit += profit
        
        # Update peak and drawdown
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
        
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Update Kelly history
        self.kelly.update_from_result(stake, profit, odds)
        
        logger.info(f"Bet result: P&L=${profit:.2f}, Bankroll=${self.current_bankroll:.2f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        win_rate = self.win_count / self.bet_count if self.bet_count > 0 else 0
        roi = self.total_profit / self.initial_bankroll if self.initial_bankroll > 0 else 0
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': round(self.current_bankroll, 2),
            'total_profit': round(self.total_profit, 2),
            'roi': round(roi, 4),
            'win_rate': round(win_rate, 4),
            'total_bets': self.bet_count,
            'max_drawdown': round(self.max_drawdown, 4),
        }


def create_staking_optimizer(
    bankroll: float = 1000.0,
    use_rl: bool = True,
) -> StakingOptimizer:
    """Create staking optimizer."""
    return StakingOptimizer(
        initial_bankroll=bankroll,
        use_rl=use_rl,
        kelly_fraction=0.25,
    )
