// lib/api.ts
/**
 * NEXUS AI API Client
 * Connects React frontend to FastAPI backend
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types
export interface AnalysisRequest {
  sport: 'tennis' | 'basketball';
  date?: string;
  min_quality?: number;
  top_n?: number;
}

export interface ValueBet {
  rank: number;
  match: string;
  league: string;
  selection: string;
  odds: number;
  bookmaker: string;
  edge: number;
  quality_score: number;
  stake_recommendation: string;
  confidence: number;
  reasoning: string[];
}

export interface AnalysisResponse {
  status: string;
  sport: string;
  date: string;
  value_bets: ValueBet[];
  matches_analyzed: number;
  quality_filtered: number;
  timestamp: string;
}

export interface SystemStatus {
  status: string;
  mode: string;
  api_keys_configured: Record<string, boolean>;
  last_analysis: string | null;
  uptime_seconds: number;
}

export interface SystemStats {
  total_analyses: number;
  successful_bets: number;
  win_rate: number;
  total_profit: number;
  avg_edge: number;
  avg_quality: number;
  sports_analyzed: string[];
  last_7_days: Array<{ date: string; profit: number }>;
}

// Handicap types
export interface HalfStats {
  avg_scored: number;
  avg_conceded: number;
  avg_margin: number;
}

export interface HandicapRequest {
  sport: 'tennis' | 'basketball' | string;
  market_type: 'match_handicap' | 'first_half' | 'second_half' | 'total_over' | 'total_under' | 'first_half_total';
  home_stats: Record<string, unknown>;
  away_stats: Record<string, unknown>;
  line: number;
  bookmaker_odds?: Record<string, [number, number]>;
}

export interface HandicapValueBet {
  line: number;
  side: 'home' | 'away';
  odds: number;
  fair_odds: number;
  probability: number;
  edge: number;
  confidence: number;
  reasoning: string[];
}

export interface HandicapResponse {
  market_type: string;
  line: number;
  cover_probability: number;
  fair_odds: number;
  expected_margin: number;
  confidence: number;
  reasoning: string[];
  half_patterns: {
    first_half: HalfStats;
    second_half: HalfStats;
  } | null;
  value_bets: HandicapValueBet[] | null;
}

export interface HandicapMarket {
  type: string;
  name: string;
  example_lines: number[];
}

export interface HandicapMarketsResponse {
  sport: string;
  markets: HandicapMarket[];
}

// WebSocket connection
let ws: WebSocket | null = null;
type ProgressCallback = (data: {
  type: string;
  step: string;
  progress: number;
  message: string;
  data?: ValueBet[];
}) => void;

// API Functions
export const api = {
  // System
  async getStatus(): Promise<SystemStatus> {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error('Failed to get status');
    return res.json();
  },

  async getStats(): Promise<SystemStats> {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) throw new Error('Failed to get stats');
    return res.json();
  },

  // Analysis
  async runAnalysis(request: AnalysisRequest): Promise<{ status: string; message: string }> {
    const res = await fetch(`${API_BASE}/api/analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Analysis failed');
    }
    return res.json();
  },

  // Predictions
  async getPredictions(sport?: string, date?: string): Promise<AnalysisResponse> {
    const params = new URLSearchParams();
    if (sport) params.append('sport', sport);
    if (date) params.append('date', date);

    const res = await fetch(`${API_BASE}/api/predictions?${params}`);
    if (!res.ok) throw new Error('Failed to get predictions');
    return res.json();
  },

  // Value Bets
  async getValueBets(): Promise<ValueBet[]> {
    const res = await fetch(`${API_BASE}/api/value-bets`);
    if (!res.ok) throw new Error('Failed to get value bets');
    return res.json();
  },

  // Matches
  async getMatches(sport: string, date?: string): Promise<{ matches: unknown[] }> {
    const params = new URLSearchParams({ sport });
    if (date) params.append('date', date);

    const res = await fetch(`${API_BASE}/api/matches?${params}`);
    if (!res.ok) throw new Error('Failed to get matches');
    return res.json();
  },

  // Handicap predictions
  async predictHandicap(request: HandicapRequest): Promise<HandicapResponse> {
    const res = await fetch(`${API_BASE}/api/handicap`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Handicap prediction failed');
    }
    return res.json();
  },

  async getHandicapMarkets(sport: string): Promise<HandicapMarketsResponse> {
    const res = await fetch(`${API_BASE}/api/handicap/markets?sport=${sport}`);
    if (!res.ok) throw new Error('Failed to get handicap markets');
    return res.json();
  },

  // WebSocket
  connectWebSocket(onProgress: ProgressCallback): WebSocket {
    if (ws && ws.readyState === WebSocket.OPEN) {
      return ws;
    }

    const wsUrl = API_BASE.replace('http', 'ws');
    ws = new WebSocket(`${wsUrl}/api/ws`);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onProgress(data);
      } catch (e) {
        console.error('WebSocket message error:', e);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      ws = null;
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return ws;
  },

  disconnectWebSocket() {
    if (ws) {
      ws.close();
      ws = null;
    }
  },
};

// Hooks helper
export function formatEdge(edge: number): string {
  return `+${(edge * 100).toFixed(1)}%`;
}

export function formatOdds(odds: number): string {
  return odds.toFixed(2);
}

export function getQualityColor(score: number): string {
  if (score >= 80) return 'text-green-500';
  if (score >= 60) return 'text-yellow-500';
  if (score >= 40) return 'text-orange-500';
  return 'text-red-500';
}

export function getRankColor(rank: number): string {
  switch (rank) {
    case 1:
      return 'from-yellow-400 to-yellow-600'; // Gold
    case 2:
      return 'from-gray-300 to-gray-500'; // Silver
    case 3:
      return 'from-orange-400 to-orange-600'; // Bronze
    default:
      return 'from-blue-400 to-blue-600';
  }
}

export default api;
