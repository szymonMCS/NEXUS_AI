// lib/api.ts
/**
 * NEXUS AI API Client
 * Connects React frontend to FastAPI backend with Clerk authentication
 */

import { useAuth } from '@clerk/clerk-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Sport types
export type SportId = 'tennis' | 'basketball' | 'greyhound' | 'handball' | 'table_tennis';

export interface Sport {
  id: SportId;
  name: string;
  icon: string;
  markets: string[];
  models: string[];
  status: 'active' | 'beta' | 'coming_soon';
}

export interface AvailableSportsResponse {
  sports: Sport[];
  default: SportId;
  total: number;
}

// Types
export interface AnalysisRequest {
  sport: SportId;
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
  roi: number;
  avg_edge: number;
  avg_quality: number;
  roi_by_sport: Record<string, { win_rate: number; roi: number; total_bets: number; profit: number }>;
  sports_analyzed: string[];
  last_7_days: Array<{ date: string; profit: number }>;
}

// API Match (from /api/matches)
export interface ApiMatch {
  id: number;
  external_id: string;
  home_team: string;
  away_team: string;
  league: string;
  start_time: string;
  is_live: boolean;
  is_finished: boolean;
  home_score: number | null;
  away_score: number | null;
  quality_score: number | null;
}

export interface MatchesResponse {
  status: string;
  sport: string;
  date: string;
  matches: ApiMatch[];
  total: number;
  message?: string;
}

// ML types
export interface MlModelInfo {
  name: string;
  version: string;
  type: string;
  accuracy: number | null;
  created: string | null;
}

export interface MlStatusResponse {
  status: string;
  ml_status: Record<string, {
    models_available: number;
    models: MlModelInfo[];
  }>;
}

export interface EnsembleStatusResponse {
  status: string;
  ensemble_status: Record<string, {
    available_models: number;
    ensemble_enabled: boolean;
    primary_model: string;
    model_rankings: Record<string, number>;
  }>;
}

export interface EnsembleRankingsResponse {
  sport: string;
  rankings: Record<string, number>;
  primary_model: string;
  total_models: number;
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

// Bet history types
export interface BetHistoryItem {
  id: number;
  match: string;
  league: string;
  sport: string;
  selection: string;
  odds: number;
  stake: number;
  status: 'pending' | 'won' | 'lost' | 'void';
  profit_loss: number;
  edge: number;
  confidence: number;
  created_at: string;
  settled_at: string | null;
}

export interface BetHistoryResponse {
  status: string;
  bets: BetHistoryItem[];
  total: number;
  offset: number;
  limit: number;
}

// Get auth token from Clerk
async function getAuthToken(): Promise<string | null> {
  try {
    // Check if we're in development mode without Clerk
    const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
    if (!publishableKey || publishableKey.includes('your_publishable_key_here')) {
      return null; // No auth in development mode
    }
    
    // Try to get token from Clerk
    const { getToken } = window.Clerk?.session || {};
    if (getToken) {
      return await getToken();
    }
    return null;
  } catch {
    return null;
  }
}

// Helper to make authenticated requests
async function fetchWithAuth(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = await getAuthToken();
  
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((options.headers as Record<string, string>) || {}),
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  return fetch(url, {
    ...options,
    headers,
  });
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
    const res = await fetchWithAuth(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error('Failed to get status');
    return res.json();
  },

  async getStats(): Promise<SystemStats> {
    const res = await fetchWithAuth(`${API_BASE}/api/stats`);
    if (!res.ok) throw new Error('Failed to get stats');
    return res.json();
  },

  // Analysis
  async runAnalysis(request: AnalysisRequest): Promise<{ status: string; message: string }> {
    const res = await fetchWithAuth(`${API_BASE}/api/analysis`, {
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

    const res = await fetchWithAuth(`${API_BASE}/api/predictions?${params}`);
    if (!res.ok) throw new Error('Failed to get predictions');
    return res.json();
  },

  // Value Bets
  async getValueBets(): Promise<ValueBet[]> {
    const res = await fetchWithAuth(`${API_BASE}/api/value-bets`);
    if (!res.ok) throw new Error('Failed to get value bets');
    return res.json();
  },

  // Matches
  async getMatches(sport: string, date?: string): Promise<MatchesResponse> {
    const params = new URLSearchParams({ sport });
    if (date) params.append('match_date', date);

    const res = await fetchWithAuth(`${API_BASE}/api/matches?${params}`);
    if (!res.ok) throw new Error('Failed to get matches');
    return res.json();
  },

  // Available Sports
  async getAvailableSports(): Promise<AvailableSportsResponse> {
    const res = await fetchWithAuth(`${API_BASE}/api/sports/available`);
    if (!res.ok) throw new Error('Failed to get available sports');
    return res.json();
  },

  // Live Predictions
  async getLivePredictions(): Promise<{
    is_analyzing: boolean;
    current_step: string | null;
    progress: number;
    last_update: string | null;
    live_bets: ValueBet[];
    sport?: string;
    date?: string;
    matches_analyzed?: number;
  }> {
    const res = await fetchWithAuth(`${API_BASE}/api/predictions/live`);
    if (!res.ok) throw new Error('Failed to get live predictions');
    return res.json();
  },

  // Handicap predictions
  async predictHandicap(request: HandicapRequest): Promise<HandicapResponse> {
    const res = await fetchWithAuth(`${API_BASE}/api/handicap`, {
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
    const res = await fetchWithAuth(`${API_BASE}/api/handicap/markets?sport=${sport}`);
    if (!res.ok) throw new Error('Failed to get handicap markets');
    return res.json();
  },

  // Bet History
  async getBetHistory(params?: {
    sport?: string;
    status?: string;
    days?: number;
    limit?: number;
    offset?: number;
  }): Promise<BetHistoryResponse> {
    const searchParams = new URLSearchParams();
    if (params?.sport) searchParams.append('sport', params.sport);
    if (params?.status) searchParams.append('status', params.status);
    if (params?.days) searchParams.append('days', String(params.days));
    if (params?.limit) searchParams.append('limit', String(params.limit));
    if (params?.offset) searchParams.append('offset', String(params.offset));

    const res = await fetchWithAuth(`${API_BASE}/api/bets/history?${searchParams}`);
    if (!res.ok) throw new Error('Failed to get bet history');
    return res.json();
  },

  // ML Status
  async getMlStatus(): Promise<MlStatusResponse> {
    const res = await fetchWithAuth(`${API_BASE}/api/ml/status`);
    if (!res.ok) throw new Error('Failed to get ML status');
    return res.json();
  },

  async getEnsembleStatus(): Promise<EnsembleStatusResponse> {
    const res = await fetchWithAuth(`${API_BASE}/api/ml/ensemble/status`);
    if (!res.ok) throw new Error('Failed to get ensemble status');
    return res.json();
  },

  async getEnsembleRankings(sport: string): Promise<EnsembleRankingsResponse> {
    const res = await fetchWithAuth(`${API_BASE}/api/ml/ensemble/rankings/${sport}`);
    if (!res.ok) throw new Error('Failed to get ensemble rankings');
    return res.json();
  },

  // Match Details
  async getMatchDetails(matchId: string): Promise<MatchDetailsResponse> {
    const res = await fetchWithAuth(`${API_BASE}/api/match/${encodeURIComponent(matchId)}`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to get match details' }));
      throw new Error(error.detail || 'Failed to get match details');
    }
    return res.json();
  },

  // WebSocket with automatic reconnection
  connectWebSocket(onProgress: ProgressCallback): WebSocket {
    if (ws && ws.readyState === WebSocket.OPEN) {
      return ws;
    }

    let reconnectAttempts = 0;
    const maxReconnects = 5;

    function connect(): WebSocket {
      const wsUrl = API_BASE.replace('http', 'ws');
      ws = new WebSocket(`${wsUrl}/api/ws`);

      ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data !== 'pong') {
            onProgress(data);
          }
        } catch (e) {
          console.error('WebSocket message error:', e);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected', event.code);
        ws = null;
        // Auto-reconnect unless intentionally closed
        if (event.code !== 1000 && event.code !== 1001 && reconnectAttempts < maxReconnects) {
          reconnectAttempts++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
          console.log(`WebSocket reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
          setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        // onclose will fire after this
      };

      return ws;
    }

    return connect();
  },

  disconnectWebSocket() {
    if (ws) {
      ws.close(1000, 'Client disconnect');
      ws = null;
    }
  },
};

// Match Details types
export interface MatchDetailsResponse {
  match: string;
  match_id: string;
  league: string;
  sport: SportId;
  match_time: string;
  home_team: {
    name: string;
    stats: Record<string, unknown>;
  };
  away_team: {
    name: string;
    stats: Record<string, unknown>;
  };
  value_bet: {
    selection: string;
    selection_name: string;
    probability: number;
    odds: number;
    fair_odds: number;
    edge: number;
    confidence: number;
    quality_score: number;
    bookmaker: string;
    stake_recommendation: string;
    kelly_stake: number;
  };
  ai_analysis: {
    summary: string;
    reasoning: string[];
    key_factors: Array<{
      factor: string;
      impact: 'high' | 'medium' | 'low';
      description: string;
    }>;
    warnings: string[];
    confidence_breakdown: {
      data_quality: number;
      model_agreement: number;
      market_efficiency: number;
    };
  };
  data_quality: {
    overall: number;
    components: Array<{
      name: string;
      score: number;
      description: string;
    }>;
  };
  timestamp: string;
}

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

// Extend Window interface for Clerk
declare global {
  interface Window {
    Clerk?: {
      session?: {
        getToken: () => Promise<string>;
      };
    };
  }
}
