// hooks/use-stats.ts
/**
 * Hook for fetching system statistics
 */

import { useState, useEffect, useCallback } from 'react';
import api from '@/lib/api';
import type { SystemStats } from '@/lib/api';

interface UseStatsReturn {
  stats: SystemStats | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useStats(): UseStatsReturn {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      setError('Nie udało się pobrać statystyk');
      // Fallback data
      setStats({
        total_analyses: 0,
        successful_bets: 0,
        win_rate: 0,
        total_profit: 0,
        avg_edge: 0,
        avg_quality: 0,
        sports_analyzed: ['tennis', 'basketball', 'greyhound', 'handball', 'table_tennis'],
        last_7_days: [],
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return {
    stats,
    loading,
    error,
    refetch: fetchStats,
  };
}

export default useStats;
