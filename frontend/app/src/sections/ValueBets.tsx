import { useEffect, useState, useRef, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Trophy, TrendingUp, ArrowRight, CheckCircle, RefreshCw, AlertCircle } from 'lucide-react';
import api, { formatEdge, getQualityColor, getRankColor } from '@/lib/api';
import type { ValueBet, SportId } from '@/lib/api';
import { useSports } from '@/hooks/use-sports';

export function ValueBets() {
  const [isVisible, setIsVisible] = useState(false);
  const [valueBets, setValueBets] = useState<ValueBet[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSport, setSelectedSport] = useState<SportId | 'all'>('all');
  const sectionRef = useRef<HTMLElement>(null);
  const { sports } = useSports();

  // Fetch value bets on mount
  useEffect(() => {
    fetchValueBets();
  }, []);

  // Intersection observer for animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const fetchValueBets = async () => {
    setLoading(true);
    setError(null);
    try {
      const bets = await api.getValueBets();
      setValueBets(bets);
    } catch (err) {
      console.error('Failed to fetch value bets:', err);
      setError('Nie udało się pobrać zakładów. Sprawdź połączenie z serwerem.');
      // Use demo data as fallback
      setValueBets([
        {
          rank: 1,
          match: 'Sinner J. vs Alcaraz C.',
          league: 'Australian Open',
          selection: 'home',
          odds: 2.15,
          bookmaker: 'Fortuna',
          edge: 0.042,
          quality_score: 78,
          stake_recommendation: '1.5-2% bankroll',
          confidence: 0.72,
          reasoning: ['Ranking advantage', 'Better recent form'],
        },
        {
          rank: 2,
          match: 'Sabalenka A. vs Swiatek I.',
          league: 'Australian Open',
          selection: 'away',
          odds: 1.95,
          bookmaker: 'STS',
          edge: 0.038,
          quality_score: 82,
          stake_recommendation: '1-1.5% bankroll',
          confidence: 0.68,
          reasoning: ['H2H advantage', 'Tournament form'],
        },
        {
          rank: 3,
          match: 'LA Lakers vs Boston Celtics',
          league: 'NBA',
          selection: 'home',
          odds: 2.10,
          bookmaker: 'Betclic',
          edge: 0.031,
          quality_score: 75,
          stake_recommendation: '1% bankroll',
          confidence: 0.65,
          reasoning: ['Home advantage', 'Rest days favor'],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Filter bets by selected sport
  const filteredBets = useMemo(() => {
    if (selectedSport === 'all') return valueBets;
    // Simple filtering based on league names containing sport keywords
    const sportKeywords: Record<SportId, string[]> = {
      tennis: ['ATP', 'WTA', 'Open', 'Grand Slam', 'Wimbledon', 'Roland Garros'],
      basketball: ['NBA', 'Euroleague', 'Basketball', 'NCAA'],
      greyhound: ['Greyhound', 'Racing', 'Dog'],
      handball: ['Handball', 'EHF', 'Champions League Handball'],
      table_tennis: ['Table Tennis', 'ITTF', 'Ping Pong'],
    };
    const keywords = sportKeywords[selectedSport] || [];
    return valueBets.filter(bet =>
      keywords.some(kw => bet.league.toLowerCase().includes(kw.toLowerCase())) ||
      (bet as any).sport === selectedSport
    );
  }, [valueBets, selectedSport]);

  return (
    <section ref={sectionRef} id="value-bets" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 mb-12">
          <div>
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-full mb-4 transition-all duration-700 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              <Trophy className="w-4 h-4 text-violet-400" />
              <span className="text-sm text-violet-400 font-medium">Value Bets</span>
            </div>
            <h2
              className={`text-3xl sm:text-4xl font-bold text-white mb-4 transition-all duration-700 delay-100 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Top Value Bets
            </h2>
            <p
              className={`text-gray-400 max-w-xl transition-all duration-700 delay-200 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              AI skanuje rynki bukmacherskie i znajduje zakłady z najwyższym edge × quality × confidence
            </p>
          </div>
          <div
            className={`flex flex-col sm:flex-row items-end gap-4 transition-all duration-700 delay-300 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            {/* Sport Filter */}
            <div className="flex flex-wrap gap-2">
              <Button
                variant={selectedSport === 'all' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedSport('all')}
                className={selectedSport === 'all' ? 'bg-gradient-primary' : 'bg-white/5 border-white/10 text-white'}
              >
                Wszystkie
              </Button>
              {sports.map((s) => (
                <Button
                  key={s.id}
                  variant={selectedSport === s.id ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedSport(s.id)}
                  className={selectedSport === s.id ? 'bg-gradient-primary' : 'bg-white/5 border-white/10 text-white'}
                >
                  {s.icon} {s.name}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-2xl font-bold text-white">{filteredBets.length}</div>
                <div className="text-sm text-gray-400">value bets</div>
              </div>
              <Button
                onClick={fetchValueBets}
                disabled={loading}
                className="bg-gradient-primary hover:opacity-90 text-white whitespace-nowrap"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4 mr-2" />
                )}
                Odśwież
              </Button>
            </div>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Value Bets Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {filteredBets.map((bet, index) => (
            <Card
              key={`${bet.match}-${bet.rank}`}
              className={`bg-glass-card border-white/5 overflow-hidden transition-all duration-500 hover:scale-[1.02] hover:border-violet-500/30 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
              style={{ transitionDelay: `${index * 100 + 200}ms` }}
            >
              <CardContent className="p-6">
                {/* Rank Badge */}
                <div className="flex items-center justify-between mb-4">
                  <div
                    className={`w-10 h-10 rounded-full bg-gradient-to-br ${getRankColor(
                      bet.rank
                    )} flex items-center justify-center font-bold text-white text-lg`}
                  >
                    #{bet.rank}
                  </div>
                  <Badge variant="secondary" className="bg-white/5 text-gray-400">
                    {bet.league}
                  </Badge>
                </div>

                {/* Match */}
                <div className="mb-4">
                  <div className="text-lg font-semibold text-white">{bet.match}</div>
                  <div className="text-sm text-gray-400 mt-1">
                    Typ: <span className="text-violet-400 font-medium">{bet.selection.toUpperCase()}</span>
                  </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-3 bg-white/5 rounded-lg">
                    <div className="text-xs text-gray-400 mb-1">Kurs</div>
                    <div className="text-lg font-bold text-white">{bet.odds.toFixed(2)}</div>
                    <div className="text-xs text-gray-500">{bet.bookmaker}</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg">
                    <div className="text-xs text-gray-400 mb-1">Edge</div>
                    <div className="text-lg font-bold text-green-400">{formatEdge(bet.edge)}</div>
                    <div className="text-xs text-gray-500">Value</div>
                  </div>
                </div>

                {/* Quality & Confidence */}
                <div className="flex items-center justify-between mb-4 text-sm">
                  <div>
                    <span className="text-gray-400">Jakość: </span>
                    <span className={`font-semibold ${getQualityColor(bet.quality_score)}`}>
                      {bet.quality_score}/100
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Pewność: </span>
                    <span className="font-semibold text-blue-400">{(bet.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Stake Recommendation */}
                <div className="p-3 bg-violet-500/10 rounded-lg mb-4">
                  <div className="text-xs text-violet-300 mb-1">Rekomendacja</div>
                  <div className="text-sm font-medium text-white">{bet.stake_recommendation}</div>
                </div>

                {/* Reasoning */}
                {bet.reasoning && bet.reasoning.length > 0 && (
                  <div className="text-xs text-gray-400">
                    {bet.reasoning.slice(0, 2).map((r, i) => (
                      <div key={i} className="flex items-center gap-1">
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        {r}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Empty State */}
        {filteredBets.length === 0 && !loading && (
          <div className="text-center py-12">
            <Trophy className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Brak value bets</h3>
            <p className="text-gray-400 mb-4">
              Uruchom analizę, aby znaleźć najlepsze zakłady
            </p>
            <Button
              className="bg-gradient-primary hover:opacity-90 text-white"
              onClick={() => window.location.href = '#live-predictions'}
            >
              Uruchom analizę
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        )}

        {/* Run Analysis CTA */}
        {filteredBets.length > 0 && (
          <div
            className={`text-center transition-all duration-700 delay-500 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            <Button
              size="lg"
              className="bg-gradient-primary hover:opacity-90 text-white font-semibold px-8"
              onClick={() => window.location.href = '#live-predictions'}
            >
              <TrendingUp className="w-5 h-5 mr-2" />
              Uruchom nową analizę
            </Button>
          </div>
        )}
      </div>
    </section>
  );
}
