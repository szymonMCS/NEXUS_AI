// pages/AnalysisPage.tsx
/**
 * Analysis Page - Detailed match analysis with AI reasoning
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Activity,
  Search,
  Brain,
  Zap,
  RefreshCw,
  Clock,
  ChevronRight,
} from 'lucide-react';
import { SportSelector } from '@/components/SportSelector';
import api, { type SportId, type ValueBet } from '@/lib/api';

// Local type for match analysis data used in this page
interface MatchAnalysisData {
  match: string;
  league: string;
  sport: string;
  matchTime: string;
  homeTeam: {
    name: string;
    stats: Record<string, unknown>;
  };
  awayTeam: {
    name: string;
    stats: Record<string, unknown>;
  };
  valueBet: {
    selection: string;
    selectionName: string;
    probability: number;
    odds: number;
    fairOdds: number;
    edge: number;
    confidence: number;
    qualityScore: number;
    bookmaker: string;
    stakeRecommendation: string;
    kellyStake: number;
  };
  aiAnalysis: {
    summary: string;
    reasoning: string[];
    keyFactors: Array<{ factor: string; impact: string; description: string }>;
    warnings: string[];
    confidenceBreakdown: {
      dataQuality: number;
      modelAgreement: number;
      marketEfficiency: number;
    };
  };
  dataQuality: {
    overall: number;
    components: Array<{ name: string; score: number; description: string }>;
  };
}

// Sample match analysis data
const sampleAnalysisData: MatchAnalysisData = {
  match: 'Novak Djokovic vs Carlos Alcaraz',
  league: 'Australian Open - Final',
  sport: 'tennis',
  matchTime: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),

  homeTeam: {
    name: 'Novak Djokovic',
    stats: {
      rank: 1,
      winRate: 0.89,
      recentForm: 'WWWWW',
      avgPointsScored: 6.2,
      avgPointsConceded: 4.1,
      h2hRecord: '7-4',
      surfaceWinRate: 0.92,
      fatigue: 'Low',
      injuryStatus: 'Healthy',
    },
  },
  awayTeam: {
    name: 'Carlos Alcaraz',
    stats: {
      rank: 2,
      winRate: 0.85,
      recentForm: 'WWLWW',
      avgPointsScored: 5.8,
      avgPointsConceded: 4.3,
      h2hRecord: '4-7',
      surfaceWinRate: 0.78,
      fatigue: 'Medium',
      injuryStatus: 'Minor concern',
    },
  },

  valueBet: {
    selection: 'Djokovic ML',
    selectionName: 'Djokovic to Win',
    probability: 0.62,
    odds: 1.75,
    fairOdds: 1.61,
    edge: 0.087,
    confidence: 0.78,
    qualityScore: 85,
    bookmaker: 'Pinnacle',
    stakeRecommendation: '3.5% bankroll (Kelly: 4.2%)',
    kellyStake: 4.2,
  },

  aiAnalysis: {
    summary: 'Djokovic ma wyraźną przewagę na nawierzchni twardej i lepszy bilans H2H. Alcaraz pokazuje oznaki zmęczenia po ciężkim półfinale.',
    reasoning: [
      'Djokovic ma 92% win rate na nawierzchniach twardych w tym sezonie',
      'Bilans H2H 7-4 na korzyść Djokovica, w tym 3 ostatnie spotkania',
      'Alcaraz zagrał 5-setowy mecz w półfinale, Djokovic wygrał w 3 setach',
      'Djokovic nie przegrał seta w całym turnieju',
      'Forma Djokovica: 15 wygranych z rzędu na Australian Open',
    ],
    keyFactors: [
      { factor: 'Doświadczenie w finałach', impact: 'high', description: 'Djokovic: 10 finałów AO, Alcaraz: pierwszy' },
      { factor: 'Nawierzchnia', impact: 'high', description: '92% vs 78% win rate na hard court' },
      { factor: 'Zmęczenie', impact: 'medium', description: 'Alcaraz po 5-setowym półfinale' },
      { factor: 'H2H', impact: 'medium', description: '7-4 dla Djokovica' },
      { factor: 'Forma', impact: 'low', description: 'Obaj w świetnej formie' },
    ],
    warnings: [
      'Alcaraz może być niedoceniony przez bukmacherów',
      'Mecze finałowe często są nieprzewidywalne',
    ],
    confidenceBreakdown: {
      dataQuality: 0.90,
      modelAgreement: 0.82,
      marketEfficiency: 0.75,
    },
  },

  dataQuality: {
    overall: 85,
    components: [
      { name: 'Dane historyczne', score: 92, description: 'Pełne dane z ostatnich 2 lat' },
      { name: 'Dane H2H', score: 88, description: '11 spotkań w bazie' },
      { name: 'Aktualne statystyki', score: 85, description: 'Dane z bieżącego turnieju' },
      { name: 'Kursy bukmacherów', score: 78, description: 'Dane z 3 bukmacherów' },
      { name: 'Czynniki zewnętrzne', score: 72, description: 'Brak danych o pogodzie (kort kryty)' },
    ],
  },
};

export function AnalysisPage() {
  const [selectedSport, setSelectedSport] = useState<SportId>('tennis');
  const [valueBets, setValueBets] = useState<ValueBet[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMatch, setSelectedMatch] = useState<MatchAnalysisData | null>(sampleAnalysisData);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);

  useEffect(() => {
    loadData();
  }, [selectedSport]);

  const loadData = async () => {
    try {
      const betsData = await api.getValueBets().catch(() => []);
      setValueBets(betsData);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const runFullAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);

    // Simulate analysis progress
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          return 100;
        }
        return prev + 10;
      });
    }, 500);

    try {
      await api.runAnalysis({ sport: selectedSport });
      api.connectWebSocket((data) => {
        if (data.progress) {
          setAnalysisProgress(data.progress);
        }
        if (data.type === 'complete') {
          setIsAnalyzing(false);
          loadData();
        }
      });
    } catch (error) {
      console.error('Analysis failed:', error);
      clearInterval(interval);
      setIsAnalyzing(false);
    }
  };

  const filteredBets = valueBets.filter(bet =>
    bet.match.toLowerCase().includes(searchQuery.toLowerCase()) ||
    bet.league.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              Analiza AI
            </h1>
            <p className="text-gray-400 mt-1">
              Szczegółowa analiza meczów z wykorzystaniem sztucznej inteligencji
            </p>
          </div>

          <div className="flex items-center gap-4">
            <SportSelector
              value={selectedSport}
              onChange={setSelectedSport}
            />
            <Button
              onClick={runFullAnalysis}
              disabled={isAnalyzing}
              className="bg-gradient-to-r from-blue-500 to-cyan-600 hover:opacity-90"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Analizuję... {analysisProgress}%
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Pełna analiza
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Analysis Progress */}
        {isAnalyzing && (
          <Card className="bg-glass-card border-white/5 mb-6">
            <CardContent className="p-4">
              <div className="flex items-center gap-4">
                <Activity className="w-5 h-5 text-blue-400 animate-pulse" />
                <div className="flex-1">
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-white">Analiza w toku...</span>
                    <span className="text-sm text-gray-400">{analysisProgress}%</span>
                  </div>
                  <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300"
                      style={{ width: `${analysisProgress}%` }}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Match List */}
          <div className="lg:col-span-1 space-y-4">
            <Card className="bg-glass-card border-white/5">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg text-white">Mecze do analizy</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    placeholder="Szukaj meczu..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 bg-white/5 border-white/10 text-white"
                  />
                </div>

                {/* Match List */}
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {/* Sample Match Card */}
                  <div
                    className={`p-3 rounded-lg cursor-pointer transition-all ${
                      selectedMatch?.match === sampleAnalysisData.match
                        ? 'bg-blue-500/20 border border-blue-500/50'
                        : 'bg-white/5 hover:bg-white/10 border border-transparent'
                    }`}
                    onClick={() => setSelectedMatch(sampleAnalysisData)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <Badge className="bg-yellow-500/20 text-yellow-300 text-xs">
                        #1 TOP PICK
                      </Badge>
                      <span className="text-xs text-gray-400">
                        <Clock className="w-3 h-3 inline mr-1" />
                        2h
                      </span>
                    </div>
                    <h4 className="text-white font-medium text-sm mb-1">
                      {sampleAnalysisData.match}
                    </h4>
                    <p className="text-xs text-gray-400 mb-2">{sampleAnalysisData.league}</p>
                    <div className="flex items-center justify-between">
                      <Badge className="bg-green-500/20 text-green-300 text-xs">
                        +{(sampleAnalysisData.valueBet.edge * 100).toFixed(1)}% edge
                      </Badge>
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>

                  {/* Value Bets from API */}
                  {filteredBets.map((bet, index) => (
                    <div
                      key={`${bet.match}-${index}`}
                      className="p-3 rounded-lg cursor-pointer transition-all bg-white/5 hover:bg-white/10 border border-transparent"
                      onClick={() => {
                        // Convert ValueBet to MatchAnalysisData format
                        const analysisData: MatchAnalysisData = {
                          match: bet.match,
                          league: bet.league,
                          sport: selectedSport,
                          matchTime: new Date(Date.now() + Math.random() * 24 * 60 * 60 * 1000).toISOString(),
                          homeTeam: {
                            name: bet.match.split(' vs ')[0] || 'Team A',
                            stats: { winRate: 0.65, recentForm: 'WWLWW' },
                          },
                          awayTeam: {
                            name: bet.match.split(' vs ')[1] || 'Team B',
                            stats: { winRate: 0.58, recentForm: 'WLWLW' },
                          },
                          valueBet: {
                            selection: bet.selection,
                            selectionName: bet.selection,
                            probability: bet.confidence,
                            odds: bet.odds,
                            fairOdds: bet.odds / (1 + bet.edge),
                            edge: bet.edge,
                            confidence: bet.confidence,
                            qualityScore: bet.quality_score,
                            bookmaker: bet.bookmaker,
                            stakeRecommendation: bet.stake_recommendation,
                            kellyStake: 3.5,
                          },
                          aiAnalysis: {
                            summary: 'Analiza oparta na danych historycznych i aktualnej formie.',
                            reasoning: bet.reasoning,
                            keyFactors: [],
                            warnings: [],
                            confidenceBreakdown: {
                              dataQuality: bet.quality_score / 100,
                              modelAgreement: bet.confidence,
                              marketEfficiency: 0.75,
                            },
                          },
                          dataQuality: {
                            overall: bet.quality_score,
                            components: [],
                          },
                        };
                        setSelectedMatch(analysisData);
                      }}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <Badge className={`text-xs ${bet.rank <= 3 ? 'bg-yellow-500/20 text-yellow-300' : 'bg-violet-500/20 text-violet-300'}`}>
                          #{bet.rank}
                        </Badge>
                        <span className="text-xs text-gray-400">{bet.bookmaker}</span>
                      </div>
                      <h4 className="text-white font-medium text-sm mb-1">{bet.match}</h4>
                      <p className="text-xs text-gray-400 mb-2">{bet.league}</p>
                      <div className="flex items-center justify-between">
                        <Badge className="bg-green-500/20 text-green-300 text-xs">
                          +{(bet.edge * 100).toFixed(1)}% edge
                        </Badge>
                        <ChevronRight className="w-4 h-4 text-gray-400" />
                      </div>
                    </div>
                  ))}

                  {filteredBets.length === 0 && !searchQuery && (
                    <div className="text-center py-8">
                      <Activity className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                      <p className="text-gray-400 text-sm">
                        Uruchom analizę, aby zobaczyć mecze
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Analysis Report */}
          <div className="lg:col-span-2">
            {selectedMatch ? (
              <Card className="bg-glass-card border-white/5">
                <CardHeader>
                  <CardTitle className="text-xl text-white flex items-center gap-2">
                    <Brain className="w-5 h-5 text-violet-400" />
                    {selectedMatch.match}
                  </CardTitle>
                  <Badge className="w-fit bg-violet-500/20 text-violet-300">{selectedMatch.league}</Badge>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Value Bet Summary */}
                  <div className="grid grid-cols-3 gap-4 p-4 bg-gradient-to-br from-green-500/10 to-emerald-500/10 rounded-lg border border-green-500/20">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-white">{selectedMatch.valueBet.odds.toFixed(2)}</div>
                      <div className="text-xs text-gray-400">Kurs</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">+{(selectedMatch.valueBet.edge * 100).toFixed(1)}%</div>
                      <div className="text-xs text-gray-400">Edge</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-400">{(selectedMatch.valueBet.confidence * 100).toFixed(0)}%</div>
                      <div className="text-xs text-gray-400">Pewność</div>
                    </div>
                  </div>

                  {/* AI Analysis */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-300 mb-2">Analiza AI</h4>
                    <p className="text-gray-400 text-sm">{selectedMatch.aiAnalysis.summary}</p>
                  </div>

                  {/* Reasoning */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-300 mb-2">Uzasadnienie</h4>
                    <div className="space-y-2">
                      {selectedMatch.aiAnalysis.reasoning.map((reason, i) => (
                        <div key={i} className="flex items-start gap-2 text-sm">
                          <div className="w-5 h-5 rounded-full bg-violet-500/20 flex items-center justify-center text-xs text-violet-300 mt-0.5">{i + 1}</div>
                          <span className="text-gray-300">{reason}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Data Quality */}
                  <div className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-300">Jakość danych</span>
                      <span className="text-sm font-bold text-white">{selectedMatch.dataQuality.overall}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-violet-500 to-blue-500" style={{ width: `${selectedMatch.dataQuality.overall}%` }} />
                    </div>
                  </div>

                  {/* Stake Recommendation */}
                  <div className="p-4 bg-violet-500/10 rounded-lg border border-violet-500/20">
                    <h4 className="text-sm font-semibold text-violet-300 mb-1">Rekomendowana stawka</h4>
                    <p className="text-lg font-bold text-white">{selectedMatch.valueBet.stakeRecommendation}</p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-glass-card border-white/5">
                <CardContent className="p-12 text-center">
                  <Brain className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">
                    Wybierz mecz do analizy
                  </h3>
                  <p className="text-gray-400">
                    Kliknij na mecz z listy po lewej stronie, aby zobaczyć szczegółową analizę AI.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalysisPage;
