// pages/ValueBetsPage.tsx
/**
 * Value Bets Page - Comprehensive value bets listing with filtering
 */

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Trophy,
  Clock,
  RefreshCw,
  Zap,
  Grid3X3,
  List,
  ExternalLink,
  AlertCircle,
} from 'lucide-react';
import { Top3ValueBets } from '@/components/Top3ValueBets';
import { AdvancedFilters, type FilterValues } from '@/components/AdvancedFilters';
import { SportSelector } from '@/components/SportSelector';
import { MatchDetailsModal } from '@/components/MatchDetailsModal';
import api, { type SportId, type ValueBet, type Sport, formatEdge, getQualityColor } from '@/lib/api';

const defaultFilters: FilterValues = {
  sports: [],
  minEdge: 0,
  maxEdge: 30,
  minQuality: 0,
  minConfidence: 0,
  minOdds: 1.0,
  maxOdds: 10.0,
  bookmakers: [],
  timeRange: 'all',
  onlyTopPicks: false,
  hideLowQuality: false,
  sortBy: 'edge',
  sortOrder: 'desc',
};

export function ValueBetsPage() {
  const [selectedSport, setSelectedSport] = useState<SportId>('tennis');
  const [sports, setSports] = useState<Sport[]>([]);
  const [valueBets, setValueBets] = useState<ValueBet[]>([]);
  const [filters, setFilters] = useState<FilterValues>(defaultFilters);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [selectedBet, setSelectedBet] = useState<ValueBet | null>(null);
  const [, setIsLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    loadData();
  }, [selectedSport]);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [sportsData, betsData] = await Promise.all([
        api.getAvailableSports().catch(() => ({ sports: [], default: 'tennis' as SportId, total: 0 })),
        api.getValueBets().catch(() => []),
      ]);
      if (sportsData.sports.length > 0) setSports(sportsData.sports);
      setValueBets(betsData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      await api.runAnalysis({ sport: selectedSport });
      api.connectWebSocket((data) => {
        if (data.type === 'complete' && data.data) {
          setValueBets(data.data);
          setIsAnalyzing(false);
          setLastUpdate(new Date());
        }
      });
    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
    }
  };

  // Apply filters
  const filteredBets = valueBets.filter((bet) => {
    if (filters.sports.length > 0 && !filters.sports.includes(selectedSport)) return false;
    if (bet.edge * 100 < filters.minEdge || bet.edge * 100 > filters.maxEdge) return false;
    if (bet.quality_score < filters.minQuality) return false;
    if (bet.confidence * 100 < filters.minConfidence) return false;
    if (bet.odds < filters.minOdds || bet.odds > filters.maxOdds) return false;
    if (filters.bookmakers.length > 0 && !filters.bookmakers.includes(bet.bookmaker)) return false;
    if (filters.onlyTopPicks && (bet.rank > 3 || bet.quality_score < 70)) return false;
    if (filters.hideLowQuality && bet.quality_score < 50) return false;
    return true;
  }).sort((a, b) => {
    const order = filters.sortOrder === 'desc' ? -1 : 1;
    switch (filters.sortBy) {
      case 'edge': return (b.edge - a.edge) * order;
      case 'quality': return (b.quality_score - a.quality_score) * order;
      case 'confidence': return (b.confidence - a.confidence) * order;
      case 'odds': return (b.odds - a.odds) * order;
      default: return 0;
    }
  });

  const availableBookmakers = [...new Set(valueBets.map(b => b.bookmaker))];
  const remainingBets = filteredBets.slice(3);

  return (
    <div className="min-h-screen bg-background pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-yellow-500 to-amber-600 flex items-center justify-center">
                <Trophy className="w-6 h-6 text-white" />
              </div>
              Value Bets
            </h1>
            <p className="text-gray-400 mt-1">
              Najlepsze zakłady z dodatnią wartością oczekiwaną
            </p>
          </div>

          <div className="flex items-center gap-4">
            <SportSelector
              value={selectedSport}
              onChange={setSelectedSport}
            />
            <Button
              onClick={runAnalysis}
              disabled={isAnalyzing}
              className="bg-gradient-to-r from-yellow-500 to-amber-600 hover:opacity-90"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Szukam...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Znajdź value bets
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="text-sm text-gray-400">Znalezione</div>
              <div className="text-2xl font-bold text-white">{filteredBets.length}</div>
            </CardContent>
          </Card>
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="text-sm text-gray-400">Top Picks</div>
              <div className="text-2xl font-bold text-yellow-400">{Math.min(3, filteredBets.length)}</div>
            </CardContent>
          </Card>
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="text-sm text-gray-400">Avg Edge</div>
              <div className="text-2xl font-bold text-green-400">
                +{filteredBets.length > 0
                  ? (filteredBets.reduce((sum, b) => sum + b.edge, 0) / filteredBets.length * 100).toFixed(1)
                  : '0.0'}%
              </div>
            </CardContent>
          </Card>
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="text-sm text-gray-400">Avg Quality</div>
              <div className="text-2xl font-bold text-blue-400">
                {filteredBets.length > 0
                  ? Math.round(filteredBets.reduce((sum, b) => sum + b.quality_score, 0) / filteredBets.length)
                  : 0}%
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Last Update */}
        {lastUpdate && (
          <div className="flex items-center gap-2 mb-6 text-sm text-gray-400">
            <Clock className="w-4 h-4" />
            Ostatnia aktualizacja: {lastUpdate.toLocaleTimeString('pl-PL')}
            <Button
              variant="ghost"
              size="sm"
              onClick={loadData}
              className="text-gray-400 hover:text-white ml-2"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        )}

        {/* Filters */}
        <AdvancedFilters
          filters={filters}
          onChange={setFilters}
          availableSports={sports.map(s => ({ id: s.id, name: s.name, icon: s.icon }))}
          availableBookmakers={availableBookmakers}
          resultCount={filteredBets.length}
        />

        {/* View Toggle */}
        <div className="flex items-center justify-between mb-6 mt-6">
          <h2 className="text-xl font-semibold text-white">
            {filteredBets.length} value bet{filteredBets.length !== 1 ? 's' : ''} znaleziono
          </h2>
          <div className="flex items-center gap-2">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('grid')}
              className={viewMode === 'grid' ? 'bg-violet-500' : 'bg-white/5 border-white/10'}
            >
              <Grid3X3 className="w-4 h-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('list')}
              className={viewMode === 'list' ? 'bg-violet-500' : 'bg-white/5 border-white/10'}
            >
              <List className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {filteredBets.length === 0 ? (
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-12 text-center">
              <AlertCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Brak value bets</h3>
              <p className="text-gray-400 mb-6">
                Uruchom nową analizę lub zmień filtry, aby zobaczyć dostępne zakłady.
              </p>
              <Button
                onClick={runAnalysis}
                disabled={isAnalyzing}
                className="bg-gradient-to-r from-yellow-500 to-amber-600"
              >
                <Zap className="w-4 h-4 mr-2" />
                Uruchom analizę
              </Button>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Top 3 Section */}
            <Card className="bg-glass-card border-white/5 mb-6">
              <CardContent className="p-6">
                <Top3ValueBets
                  bets={filteredBets}
                  onViewDetails={(bet) => setSelectedBet(bet)}
                  onSave={(bet) => console.log('Save:', bet)}
                  onShare={(bet) => console.log('Share:', bet)}
                />
              </CardContent>
            </Card>

            {/* Remaining Bets */}
            {remainingBets.length > 0 && (
              <>
                <h3 className="text-lg font-semibold text-white mb-4">Pozostałe value bets</h3>
                {viewMode === 'grid' ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {remainingBets.map((bet, index) => (
                      <Card
                        key={`${bet.match}-${index}`}
                        className="bg-glass-card border-white/5 hover:border-violet-500/30 transition-all cursor-pointer"
                        onClick={() => setSelectedBet(bet)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between mb-3">
                            <Badge className="bg-violet-500/20 text-violet-300">
                              #{bet.rank}
                            </Badge>
                            <Badge className={`${bet.quality_score >= 70 ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                              {bet.quality_score}%
                            </Badge>
                          </div>
                          <h4 className="text-white font-semibold mb-1">{bet.match}</h4>
                          <p className="text-sm text-gray-400 mb-3">{bet.league}</p>
                          <div className="grid grid-cols-2 gap-2 text-sm mb-3">
                            <div>
                              <span className="text-gray-400">Typ:</span>
                              <span className="text-white ml-1">{bet.selection}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Kurs:</span>
                              <span className="text-white ml-1">{bet.odds.toFixed(2)}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Edge:</span>
                              <span className="text-green-400 ml-1">{formatEdge(bet.edge)}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Pewność:</span>
                              <span className="text-blue-400 ml-1">{(bet.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                          <div className="text-xs text-gray-500">{bet.bookmaker}</div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <Card className="bg-glass-card border-white/5">
                    <CardContent className="p-0">
                      <div className="divide-y divide-white/5">
                        {remainingBets.map((bet, index) => (
                          <div
                            key={`${bet.match}-${index}`}
                            className="p-4 hover:bg-white/5 cursor-pointer transition-colors flex items-center justify-between"
                            onClick={() => setSelectedBet(bet)}
                          >
                            <div className="flex items-center gap-4">
                              <Badge className="bg-violet-500/20 text-violet-300">
                                #{bet.rank}
                              </Badge>
                              <div>
                                <h4 className="text-white font-medium">{bet.match}</h4>
                                <p className="text-sm text-gray-400">{bet.league}</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-6 text-sm">
                              <div className="text-center">
                                <div className="text-gray-400">Typ</div>
                                <div className="text-white">{bet.selection}</div>
                              </div>
                              <div className="text-center">
                                <div className="text-gray-400">Kurs</div>
                                <div className="text-white">{bet.odds.toFixed(2)}</div>
                              </div>
                              <div className="text-center">
                                <div className="text-gray-400">Edge</div>
                                <div className="text-green-400">{formatEdge(bet.edge)}</div>
                              </div>
                              <div className="text-center">
                                <div className="text-gray-400">Jakość</div>
                                <div className={getQualityColor(bet.quality_score)}>{bet.quality_score}%</div>
                              </div>
                              <ExternalLink className="w-4 h-4 text-gray-400" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </>
        )}

        {/* Match Details Modal */}
        <MatchDetailsModal
          bet={selectedBet}
          open={selectedBet !== null}
          onClose={() => setSelectedBet(null)}
        />
      </div>
    </div>
  );
}

export default ValueBetsPage;
