// hooks/use-sports.ts
/**
 * Hook for fetching and managing available sports
 */

import { useState, useEffect, useCallback } from 'react';
import api from '@/lib/api';
import type { Sport, SportId } from '@/lib/api';

interface UseSportsReturn {
  sports: Sport[];
  loading: boolean;
  error: string | null;
  defaultSport: SportId;
  activeSports: Sport[];
  betaSports: Sport[];
  getSportById: (id: SportId) => Sport | undefined;
  refetch: () => Promise<void>;
}

export function useSports(): UseSportsReturn {
  const [sports, setSports] = useState<Sport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [defaultSport, setDefaultSport] = useState<SportId>('tennis');

  const fetchSports = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getAvailableSports();
      setSports(data.sports);
      setDefaultSport(data.default);
    } catch (err) {
      console.error('Failed to fetch sports:', err);
      setError('Nie udało się pobrać listy sportów');
      // Fallback data
      setSports([
        {
          id: 'tennis',
          name: 'Tennis',
          icon: '🎾',
          markets: ['match_winner', 'set_handicap', 'total_games'],
          models: ['TennisHandicapModel'],
          status: 'active',
        },
        {
          id: 'basketball',
          name: 'Basketball',
          icon: '🏀',
          markets: ['match_winner', 'point_spread', 'total_points'],
          models: ['BasketballHandicapModel'],
          status: 'active',
        },
        {
          id: 'greyhound',
          name: 'Greyhound Racing',
          icon: '🐕',
          markets: ['winner', 'place', 'forecast'],
          models: ['GreyhoundPredictor'],
          status: 'beta',
        },
        {
          id: 'handball',
          name: 'Handball',
          icon: '🤾',
          markets: ['match_winner', 'handicap', 'total_goals'],
          models: ['HandballPredictor'],
          status: 'beta',
        },
        {
          id: 'table_tennis',
          name: 'Table Tennis',
          icon: '🏓',
          markets: ['match_winner', 'set_handicap'],
          models: ['TableTennisPredictor'],
          status: 'beta',
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSports();
  }, [fetchSports]);

  const activeSports = sports.filter((s) => s.status === 'active');
  const betaSports = sports.filter((s) => s.status === 'beta');

  const getSportById = useCallback(
    (id: SportId) => sports.find((s) => s.id === id),
    [sports]
  );

  return {
    sports,
    loading,
    error,
    defaultSport,
    activeSports,
    betaSports,
    getSportById,
    refetch: fetchSports,
  };
}

export default useSports;
