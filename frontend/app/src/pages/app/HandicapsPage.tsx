/**
 * HandicapsPage - Handicap Analysis & Prediction
 *
 * Connected to:
 * - GET /api/matches?sport=...
 * - GET /api/handicap/markets?sport=...
 * - POST /api/handicap
 * - GET /api/match/{matchId} (for stats)
 *
 * Features:
 * - Match selector from real API
 * - Available markets from backend
 * - Handicap prediction via API
 * - Odds calculator (frontend-only)
 */

import { useState, useEffect, useCallback } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Calculator,
  DollarSign,
  Loader2,
  Percent,
  Scale,
  Target,
  TrendingUp,
  Trophy,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api, {
  type ApiMatch,
  type HandicapMarket,
  type HandicapResponse,
  type SportId,
} from '@/lib/api';

const SPORTS: { id: SportId; label: string }[] = [
  { id: 'tennis', label: 'Tennis' },
  { id: 'basketball', label: 'Basketball' },
  { id: 'handball', label: 'Handball' },
  { id: 'table_tennis', label: 'Table Tennis' },
];

export function HandicapsPage() {
  const [activeTab, setActiveTab] = useState('predict');
  const [loading, setLoading] = useState(false);
  const [matchesLoading, setMatchesLoading] = useState(false);

  // Sport & match selection
  const [selectedSport, setSelectedSport] = useState<SportId>('tennis');
  const [matches, setMatches] = useState<ApiMatch[]>([]);
  const [selectedMatchId, setSelectedMatchId] = useState<string>('');

  // Markets
  const [markets, setMarkets] = useState<HandicapMarket[]>([]);
  const [selectedMarket, setSelectedMarket] = useState<string>('');
  const [lineValue, setLineValue] = useState('1.5');

  // Prediction result
  const [prediction, setPrediction] = useState<HandicapResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string>('');
  const [predicting, setPredicting] = useState(false);

  // Calculator
  const [calcStake, setCalcStake] = useState('100');
  const [calcOdds, setCalcOdds] = useState('1.88');

  const stake = parseFloat(calcStake) || 0;
  const odds = parseFloat(calcOdds) || 0;
  const potentialReturn = stake * odds;
  const potentialProfit = potentialReturn - stake;
  const impliedProbability = odds > 0 ? (1 / odds) * 100 : 0;

  // Fetch matches for selected sport
  const fetchMatches = useCallback(async (sport: SportId) => {
    setMatchesLoading(true);
    try {
      const res = await api.getMatches(sport);
      setMatches(res.matches || []);
      if (res.matches?.length > 0) {
        setSelectedMatchId(String(res.matches[0].id));
      } else {
        setSelectedMatchId('');
      }
    } catch (e) {
      console.error('Failed to fetch matches:', e);
      setMatches([]);
    } finally {
      setMatchesLoading(false);
    }
  }, []);

  // Fetch available markets for sport
  const fetchMarkets = useCallback(async (sport: SportId) => {
    try {
      const res = await api.getHandicapMarkets(sport);
      setMarkets(res.markets || []);
      if (res.markets?.length > 0) {
        setSelectedMarket(res.markets[0].type);
        if (res.markets[0].example_lines?.length > 0) {
          setLineValue(String(res.markets[0].example_lines[0]));
        }
      }
    } catch (e) {
      console.error('Failed to fetch markets:', e);
      setMarkets([]);
    }
  }, []);

  // Load matches and markets when sport changes
  useEffect(() => {
    fetchMatches(selectedSport);
    fetchMarkets(selectedSport);
  }, [selectedSport, fetchMatches, fetchMarkets]);

  // Run handicap prediction
  const runPrediction = async () => {
    const match = matches.find((m) => String(m.id) === selectedMatchId);
    if (!match || !selectedMarket) return;

    setPredicting(true);
    setPredictionError('');
    setPrediction(null);

    try {
      // Fetch match details for stats
      const matchName = `${match.home_team} vs ${match.away_team}`;
      let homeStats: Record<string, unknown> = {};
      let awayStats: Record<string, unknown> = {};

      try {
        const details = await api.getMatchDetails(matchName);
        homeStats = details.home_team.stats;
        awayStats = details.away_team.stats;
      } catch {
        // If match details unavailable, use empty stats
        homeStats = {};
        awayStats = {};
      }

      const result = await api.predictHandicap({
        sport: selectedSport,
        market_type: selectedMarket as 'match_handicap' | 'first_half' | 'second_half' | 'total_over' | 'total_under' | 'first_half_total',
        home_stats: homeStats,
        away_stats: awayStats,
        line: parseFloat(lineValue) || 1.5,
      });

      setPrediction(result);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Prediction failed';
      setPredictionError(msg);
    } finally {
      setPredicting(false);
    }
  };

  const selectedMatch = matches.find((m) => String(m.id) === selectedMatchId);
  const selectedMarketInfo = markets.find((m) => m.type === selectedMarket);

  return (
    <PageLayout
      title="Handicap Analysis"
      description="Handicap predictions and odds calculator"
      breadcrumbs={[{ label: 'Handicaps' }]}
    >
      <div className="space-y-6">
        {/* Sport & Match Selector */}
        <Card className="bg-[#0f1623]/80 border-white/10">
          <CardContent className="p-4">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
              <div className="flex items-center gap-3">
                <Label className="text-gray-400 text-sm whitespace-nowrap">Sport</Label>
                <Select
                  value={selectedSport}
                  onValueChange={(v) => setSelectedSport(v as SportId)}
                >
                  <SelectTrigger className="w-[160px] bg-white/5 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0f1623] border-white/10">
                    {SPORTS.map((s) => (
                      <SelectItem key={s.id} value={s.id} className="text-white">
                        {s.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-3 flex-1">
                <Label className="text-gray-400 text-sm whitespace-nowrap">Match</Label>
                {matchesLoading ? (
                  <div className="flex items-center gap-2 text-gray-500 text-sm">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading...
                  </div>
                ) : (
                  <Select
                    value={selectedMatchId}
                    onValueChange={setSelectedMatchId}
                  >
                    <SelectTrigger className="w-full max-w-[400px] bg-white/5 border-white/10 text-white">
                      <SelectValue placeholder="Select a match" />
                    </SelectTrigger>
                    <SelectContent className="bg-[#0f1623] border-white/10">
                      {matches.map((m) => (
                        <SelectItem
                          key={m.id}
                          value={String(m.id)}
                          className="text-white"
                        >
                          {m.home_team} vs {m.away_team}
                        </SelectItem>
                      ))}
                      {matches.length === 0 && (
                        <SelectItem value="none" disabled className="text-gray-500">
                          No matches available
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Match Header */}
        {selectedMatch && (
          <div className="flex items-center justify-between p-6 bg-gradient-to-r from-[#00d4ff]/5 to-[#00ff88]/5 border border-white/10 rounded-xl">
            <div className="flex items-center gap-8">
              <div className="text-center">
                <div className="text-lg font-bold text-white">
                  {selectedMatch.home_team}
                </div>
                <div className="text-xs text-gray-500">Home</div>
              </div>
              <div className="text-2xl font-bold text-gray-600">VS</div>
              <div className="text-center">
                <div className="text-lg font-bold text-white">
                  {selectedMatch.away_team}
                </div>
                <div className="text-xs text-gray-500">Away</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500 mb-1">
                {selectedMatch.league || 'Unknown League'}
              </div>
              <div className="text-sm text-gray-400">
                {selectedMatch.start_time
                  ? new Date(selectedMatch.start_time).toLocaleString()
                  : ''}
              </div>
              {selectedMatch.is_live && (
                <Badge className="bg-[#ff3860]/20 text-[#ff3860] border-0 mt-1">
                  LIVE
                </Badge>
              )}
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Prediction & Results (2 cols) */}
          <div className="lg:col-span-2">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="bg-[#0f1623]/80 border border-white/10">
                <TabsTrigger
                  value="predict"
                  className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
                >
                  <Target className="w-4 h-4 mr-2" />
                  Predict
                </TabsTrigger>
                <TabsTrigger
                  value="markets"
                  className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
                >
                  <Scale className="w-4 h-4 mr-2" />
                  Markets
                </TabsTrigger>
              </TabsList>

              {/* Predict Tab */}
              <TabsContent value="predict" className="mt-6 space-y-6">
                {/* Prediction Form */}
                <Card className="bg-[#0f1623]/80 border-white/10">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Zap className="w-5 h-5 text-[#00d4ff]" />
                      Handicap Prediction
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                      Select market type and line to generate a prediction
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-gray-400 text-xs">Market Type</Label>
                        <Select
                          value={selectedMarket}
                          onValueChange={setSelectedMarket}
                        >
                          <SelectTrigger className="bg-white/5 border-white/10 text-white">
                            <SelectValue placeholder="Select market" />
                          </SelectTrigger>
                          <SelectContent className="bg-[#0f1623] border-white/10">
                            {markets.map((m) => (
                              <SelectItem
                                key={m.type}
                                value={m.type}
                                className="text-white"
                              >
                                {m.name}
                              </SelectItem>
                            ))}
                            {markets.length === 0 && (
                              <SelectItem
                                value="none"
                                disabled
                                className="text-gray-500"
                              >
                                No markets available
                              </SelectItem>
                            )}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label className="text-gray-400 text-xs">Line</Label>
                        <div className="flex gap-2">
                          <Input
                            type="number"
                            step="0.5"
                            value={lineValue}
                            onChange={(e) => setLineValue(e.target.value)}
                            className="bg-white/5 border-white/10 text-white"
                          />
                          {selectedMarketInfo?.example_lines && (
                            <div className="flex gap-1">
                              {selectedMarketInfo.example_lines.slice(0, 4).map((l) => (
                                <Button
                                  key={l}
                                  variant="outline"
                                  size="sm"
                                  className={cn(
                                    'text-xs h-9 px-2 border-white/10',
                                    lineValue === String(l)
                                      ? 'bg-[#00d4ff]/10 text-[#00d4ff] border-[#00d4ff]/30'
                                      : 'text-gray-400'
                                  )}
                                  onClick={() => setLineValue(String(l))}
                                >
                                  {l}
                                </Button>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <Button
                      onClick={runPrediction}
                      disabled={
                        predicting || !selectedMatchId || !selectedMarket
                      }
                      className="w-full bg-[#00d4ff] hover:bg-[#00d4ff]/80 text-black font-medium"
                    >
                      {predicting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Predicting...
                        </>
                      ) : (
                        <>
                          <Target className="w-4 h-4 mr-2" />
                          Run Prediction
                        </>
                      )}
                    </Button>

                    {predictionError && (
                      <div className="p-3 bg-[#ff3860]/10 border border-[#ff3860]/20 rounded-lg text-[#ff3860] text-sm">
                        {predictionError}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Prediction Results */}
                {prediction && (
                  <Card className="bg-[#0f1623]/80 border-[#00ff88]/20">
                    <CardHeader>
                      <CardTitle className="text-white flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-[#00ff88]" />
                        Prediction Result
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Key metrics */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-1">
                            Cover Probability
                          </div>
                          <div className="text-lg font-bold text-[#00d4ff]">
                            {(prediction.cover_probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-1">
                            Fair Odds
                          </div>
                          <div className="text-lg font-bold text-white font-mono">
                            {prediction.fair_odds.toFixed(2)}
                          </div>
                        </div>
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-1">
                            Expected Margin
                          </div>
                          <div
                            className={cn(
                              'text-lg font-bold',
                              prediction.expected_margin > 0
                                ? 'text-[#00ff88]'
                                : 'text-[#ff3860]'
                            )}
                          >
                            {prediction.expected_margin > 0 ? '+' : ''}
                            {prediction.expected_margin.toFixed(2)}
                          </div>
                        </div>
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-1">
                            Confidence
                          </div>
                          <div className="text-lg font-bold text-[#ff9500]">
                            {(prediction.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>

                      {/* Reasoning */}
                      {prediction.reasoning && prediction.reasoning.length > 0 && (
                        <div className="space-y-2">
                          <div className="text-sm font-medium text-white">
                            Analysis
                          </div>
                          <ul className="space-y-1">
                            {prediction.reasoning.map((r, i) => (
                              <li
                                key={i}
                                className="text-sm text-gray-400 flex items-start gap-2"
                              >
                                <span className="text-[#00d4ff] mt-1">•</span>
                                {r}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Half patterns */}
                      {prediction.half_patterns && (
                        <div className="space-y-2">
                          <div className="text-sm font-medium text-white">
                            Half Patterns
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 bg-white/5 rounded-lg">
                              <div className="text-xs text-gray-500 mb-2">
                                First Half
                              </div>
                              <div className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-gray-400">
                                    Avg Scored
                                  </span>
                                  <span className="text-white">
                                    {prediction.half_patterns.first_half.avg_scored.toFixed(
                                      2
                                    )}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">
                                    Avg Conceded
                                  </span>
                                  <span className="text-white">
                                    {prediction.half_patterns.first_half.avg_conceded.toFixed(
                                      2
                                    )}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">
                                    Avg Margin
                                  </span>
                                  <span
                                    className={cn(
                                      prediction.half_patterns.first_half
                                        .avg_margin > 0
                                        ? 'text-[#00ff88]'
                                        : 'text-[#ff3860]'
                                    )}
                                  >
                                    {prediction.half_patterns.first_half.avg_margin.toFixed(
                                      2
                                    )}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="p-3 bg-white/5 rounded-lg">
                              <div className="text-xs text-gray-500 mb-2">
                                Second Half
                              </div>
                              <div className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-gray-400">
                                    Avg Scored
                                  </span>
                                  <span className="text-white">
                                    {prediction.half_patterns.second_half.avg_scored.toFixed(
                                      2
                                    )}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">
                                    Avg Conceded
                                  </span>
                                  <span className="text-white">
                                    {prediction.half_patterns.second_half.avg_conceded.toFixed(
                                      2
                                    )}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">
                                    Avg Margin
                                  </span>
                                  <span
                                    className={cn(
                                      prediction.half_patterns.second_half
                                        .avg_margin > 0
                                        ? 'text-[#00ff88]'
                                        : 'text-[#ff3860]'
                                    )}
                                  >
                                    {prediction.half_patterns.second_half.avg_margin.toFixed(
                                      2
                                    )}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Value bets from prediction */}
                      {prediction.value_bets &&
                        prediction.value_bets.length > 0 && (
                          <div className="space-y-2">
                            <div className="text-sm font-medium text-white flex items-center gap-2">
                              <Zap className="w-4 h-4 text-[#00ff88]" />
                              Value Bets Found
                            </div>
                            {prediction.value_bets.map((vb, i) => (
                              <div
                                key={i}
                                className="p-3 bg-[#00ff88]/5 border border-[#00ff88]/20 rounded-lg"
                              >
                                <div className="flex items-center justify-between">
                                  <div>
                                    <span className="text-white font-medium">
                                      Line {vb.line} — {vb.side.toUpperCase()}
                                    </span>
                                    <div className="text-xs text-gray-400 mt-1">
                                      Prob: {(vb.probability * 100).toFixed(1)}%
                                      | Fair: {vb.fair_odds.toFixed(2)} |
                                      Confidence:{' '}
                                      {(vb.confidence * 100).toFixed(0)}%
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-[#00ff88] font-bold">
                                      +{(vb.edge * 100).toFixed(1)}% edge
                                    </div>
                                    <div className="text-white font-mono">
                                      {vb.odds.toFixed(2)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                    </CardContent>
                  </Card>
                )}

                {/* Empty state */}
                {!prediction && !predicting && !predictionError && (
                  <Card className="bg-[#0f1623]/80 border-white/10">
                    <CardContent className="p-12 text-center text-gray-500">
                      <Target className="w-10 h-10 mx-auto mb-3 opacity-30" />
                      <p className="text-sm">
                        Select a match and market, then click "Run Prediction"
                      </p>
                      <p className="text-xs mt-1">
                        Results will appear here with cover probability, fair
                        odds, and value analysis.
                      </p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              {/* Markets Tab */}
              <TabsContent value="markets" className="mt-6">
                <Card className="bg-[#0f1623]/80 border-white/10">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Scale className="w-5 h-5 text-[#00d4ff]" />
                      Available Markets — {selectedSport}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {markets.length > 0 ? (
                      <div className="space-y-3">
                        {markets.map((m) => (
                          <div
                            key={m.type}
                            className="p-4 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-colors"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <Trophy className="w-4 h-4 text-[#00d4ff]" />
                                <span className="text-white font-medium">
                                  {m.name}
                                </span>
                              </div>
                              <Badge
                                variant="outline"
                                className="border-white/20 text-gray-400 text-xs"
                              >
                                {m.type}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2 mt-2">
                              <span className="text-xs text-gray-500">
                                Example lines:
                              </span>
                              {m.example_lines.map((l) => (
                                <Badge
                                  key={l}
                                  className="bg-white/10 text-gray-300 border-0 text-xs cursor-pointer hover:bg-[#00d4ff]/20"
                                  onClick={() => {
                                    setSelectedMarket(m.type);
                                    setLineValue(String(l));
                                    setActiveTab('predict');
                                  }}
                                >
                                  {l}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="p-12 text-center text-gray-500">
                        <Scale className="w-10 h-10 mx-auto mb-3 opacity-30" />
                        <p className="text-sm">
                          No markets available for {selectedSport}.
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Calculator (1 col) */}
          <Card className="bg-[#0f1623]/80 border-white/10 h-fit">
            <CardHeader className="pb-3">
              <CardTitle className="text-white text-base flex items-center gap-2">
                <Calculator className="w-4 h-4 text-[#00d4ff]" />
                Odds Calculator
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label className="text-gray-400 text-xs">Stake</Label>
                <div className="relative">
                  <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    type="number"
                    value={calcStake}
                    onChange={(e) => setCalcStake(e.target.value)}
                    className="pl-9 bg-white/5 border-white/10 text-white"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label className="text-gray-400 text-xs">Decimal Odds</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={calcOdds}
                  onChange={(e) => setCalcOdds(e.target.value)}
                  className="bg-white/5 border-white/10 text-white"
                />
              </div>
              <div className="pt-3 border-t border-white/10 space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Potential Return</span>
                  <span className="text-white font-bold font-mono">
                    ${potentialReturn.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Potential Profit</span>
                  <span
                    className={cn(
                      'font-bold font-mono',
                      potentialProfit > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                    )}
                  >
                    {potentialProfit > 0 ? '+' : ''}${potentialProfit.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Implied Probability</span>
                  <span className="text-[#00d4ff] font-bold">
                    {impliedProbability.toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#00d4ff] rounded-full"
                    style={{
                      width: `${Math.min(impliedProbability, 100)}%`,
                    }}
                  />
                </div>
              </div>
              <div className="pt-3 border-t border-white/10">
                <div className="text-xs text-gray-500 mb-2">
                  Quick Select Odds
                </div>
                <div className="flex flex-wrap gap-1">
                  {['1.50', '1.75', '1.88', '2.00', '2.25', '2.50', '3.00'].map(
                    (o) => (
                      <Button
                        key={o}
                        variant="outline"
                        size="sm"
                        onClick={() => setCalcOdds(o)}
                        className={cn(
                          'text-xs h-7 px-2 border-white/10',
                          calcOdds === o
                            ? 'bg-[#00d4ff]/10 text-[#00d4ff] border-[#00d4ff]/30'
                            : 'text-gray-400'
                        )}
                      >
                        {o}
                      </Button>
                    )
                  )}
                </div>
              </div>

              {/* Quick apply from prediction */}
              {prediction && (
                <div className="pt-3 border-t border-white/10">
                  <div className="text-xs text-gray-500 mb-2">
                    From Prediction
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full border-[#00ff88]/30 text-[#00ff88] text-xs"
                    onClick={() =>
                      setCalcOdds(prediction.fair_odds.toFixed(2))
                    }
                  >
                    Use Fair Odds ({prediction.fair_odds.toFixed(2)})
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </PageLayout>
  );
}

export default HandicapsPage;
