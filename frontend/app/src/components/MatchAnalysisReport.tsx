// components/MatchAnalysisReport.tsx
/**
 * Detailed Match Analysis Report - Shows comprehensive AI analysis
 * Inspired by nerdytips.com and sports-ai.dev
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  TrendingUp,
  TrendingDown,
  Target,
  Brain,
  BarChart3,
  History,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  ChevronDown,
  ChevronUp,
  Zap,
  Shield,
  Activity
} from 'lucide-react';

// Types
interface PlayerStats {
  name: string;
  ranking?: number;
  recentForm: number; // Win rate last 10
  surfaceForm?: number;
  h2hWins: number;
  fatigue?: number; // Matches last 30 days
  injuries?: string[];
}

interface AnalysisFactor {
  name: string;
  weight: number;
  score: number;
  description: string;
  impact: 'positive' | 'negative' | 'neutral';
}

interface MatchAnalysis {
  matchId: string;
  homePlayer: PlayerStats;
  awayPlayer: PlayerStats;
  tournament: string;
  surface?: string;
  matchTime: string;

  // Prediction
  prediction: {
    winner: 'home' | 'away';
    probability: number;
    confidence: number;
    expectedSets?: string;
  };

  // Value Bet
  valueBet: {
    selection: string;
    odds: number;
    bookmaker: string;
    edge: number;
    fairOdds: number;
    kellyStake: number;
    recommendation: string;
  };

  // Analysis factors
  factors: AnalysisFactor[];

  // Quality metrics
  dataQuality: {
    overall: number;
    sourceAgreement: number;
    freshness: number;
    coverage: number;
  };

  // AI Reasoning
  aiReasoning: string[];
  warnings: string[];
  keyInsights: string[];
}

interface MatchAnalysisReportProps {
  analysis: MatchAnalysis;
  onPlaceBet?: () => void;
  onSave?: () => void;
}

// Factor impact icon
const ImpactIcon = ({ impact }: { impact: 'positive' | 'negative' | 'neutral' }) => {
  if (impact === 'positive') return <TrendingUp className="w-4 h-4 text-green-400" />;
  if (impact === 'negative') return <TrendingDown className="w-4 h-4 text-red-400" />;
  return <Activity className="w-4 h-4 text-gray-400" />;
};

export function MatchAnalysisReport({ analysis, onPlaceBet, onSave }: MatchAnalysisReportProps) {
  const [expandedFactor, setExpandedFactor] = useState<string | null>(null);

  const { homePlayer, awayPlayer, prediction, valueBet, factors, dataQuality, aiReasoning, warnings, keyInsights } = analysis;

  const isHomeWinner = prediction.winner === 'home';
  const predictedWinner = isHomeWinner ? homePlayer : awayPlayer;

  return (
    <div className="space-y-6">
      {/* Header with Match Info */}
      <Card className="bg-gradient-to-br from-violet-500/10 to-blue-500/10 border-violet-500/30">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
            {/* Match Title */}
            <div className="text-center lg:text-left">
              <Badge className="mb-2 bg-violet-500/20 text-violet-300">
                {analysis.tournament}
              </Badge>
              <h2 className="text-2xl font-bold text-white mb-1">
                {homePlayer.name} vs {awayPlayer.name}
              </h2>
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <Clock className="w-4 h-4" />
                {analysis.matchTime}
                {analysis.surface && (
                  <>
                    <span>|</span>
                    <span className="capitalize">{analysis.surface}</span>
                  </>
                )}
              </div>
            </div>

            {/* Prediction Summary */}
            <div className="bg-black/30 rounded-xl p-4 text-center">
              <div className="text-sm text-gray-400 mb-1">AI Prediction</div>
              <div className="text-xl font-bold text-white mb-1">{predictedWinner.name}</div>
              <div className="flex items-center justify-center gap-4">
                <div>
                  <span className="text-2xl font-bold text-green-400">{(prediction.probability * 100).toFixed(0)}%</span>
                  <div className="text-xs text-gray-400">Probability</div>
                </div>
                <div className="w-px h-10 bg-white/20" />
                <div>
                  <span className="text-2xl font-bold text-blue-400">{(prediction.confidence * 100).toFixed(0)}%</span>
                  <div className="text-xs text-gray-400">Confidence</div>
                </div>
              </div>
            </div>

            {/* Value Bet Card */}
            <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl p-4 border border-green-500/30">
              <div className="text-sm text-green-300 mb-1">Value Bet Found</div>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-white">{valueBet.odds.toFixed(2)}</span>
                <Badge className="bg-green-500/30 text-green-300">+{(valueBet.edge * 100).toFixed(1)}% Edge</Badge>
              </div>
              <div className="text-xs text-gray-400 mt-1">@ {valueBet.bookmaker}</div>
              <div className="mt-3 text-sm">
                <span className="text-gray-400">Stake: </span>
                <span className="text-white font-medium">{valueBet.recommendation}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Analysis Tabs */}
      <Tabs defaultValue="analysis" className="space-y-4">
        <TabsList className="bg-white/5 border-white/10 p-1">
          <TabsTrigger value="analysis" className="data-[state=active]:bg-violet-500/20">
            <Brain className="w-4 h-4 mr-2" /> AI Analysis
          </TabsTrigger>
          <TabsTrigger value="stats" className="data-[state=active]:bg-violet-500/20">
            <BarChart3 className="w-4 h-4 mr-2" /> Statistics
          </TabsTrigger>
          <TabsTrigger value="factors" className="data-[state=active]:bg-violet-500/20">
            <Target className="w-4 h-4 mr-2" /> Key Factors
          </TabsTrigger>
          <TabsTrigger value="quality" className="data-[state=active]:bg-violet-500/20">
            <Shield className="w-4 h-4 mr-2" /> Data Quality
          </TabsTrigger>
        </TabsList>

        {/* AI Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          {/* Key Insights */}
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Key Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {keyInsights.map((insight, i) => (
                <div key={i} className="flex items-start gap-2 p-2 bg-yellow-500/10 rounded-lg">
                  <CheckCircle className="w-4 h-4 text-yellow-400 mt-0.5" />
                  <span className="text-sm text-gray-200">{insight}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* AI Reasoning */}
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                <Brain className="w-5 h-5 text-violet-400" />
                AI Reasoning
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {aiReasoning.map((reason, i) => (
                <div key={i} className="flex items-start gap-3 p-3 bg-white/5 rounded-lg">
                  <div className="w-6 h-6 rounded-full bg-violet-500/20 flex items-center justify-center text-xs font-bold text-violet-300">
                    {i + 1}
                  </div>
                  <p className="text-sm text-gray-300 flex-1">{reason}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Warnings */}
          {warnings.length > 0 && (
            <Card className="bg-orange-500/10 border-orange-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-semibold text-orange-300 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Warnings & Risks
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {warnings.map((warning, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-orange-400 mt-0.5" />
                    <span className="text-sm text-orange-200">{warning}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Statistics Tab */}
        <TabsContent value="stats" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Home Player Stats */}
            <Card className="bg-glass-card border-white/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-semibold text-white flex items-center justify-between">
                  <span>{homePlayer.name}</span>
                  {homePlayer.ranking && (
                    <Badge variant="secondary" className="bg-white/10">#{homePlayer.ranking}</Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Recent Form (Last 10)</span>
                    <span className="text-white font-medium">{(homePlayer.recentForm * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={homePlayer.recentForm * 100} className="h-2" />
                </div>
                {homePlayer.surfaceForm !== undefined && (
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Surface Form</span>
                      <span className="text-white font-medium">{(homePlayer.surfaceForm * 100).toFixed(0)}%</span>
                    </div>
                    <Progress value={homePlayer.surfaceForm * 100} className="h-2" />
                  </div>
                )}
                <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/10">
                  <div>
                    <div className="text-xs text-gray-400">H2H Wins</div>
                    <div className="text-xl font-bold text-white">{homePlayer.h2hWins}</div>
                  </div>
                  {homePlayer.fatigue !== undefined && (
                    <div>
                      <div className="text-xs text-gray-400">Matches (30d)</div>
                      <div className="text-xl font-bold text-white">{homePlayer.fatigue}</div>
                    </div>
                  )}
                </div>
                {homePlayer.injuries && homePlayer.injuries.length > 0 && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-xs text-red-400 mb-1">Injury Concerns</div>
                    {homePlayer.injuries.map((injury, i) => (
                      <Badge key={i} variant="destructive" className="mr-1 text-xs">{injury}</Badge>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Away Player Stats */}
            <Card className="bg-glass-card border-white/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-semibold text-white flex items-center justify-between">
                  <span>{awayPlayer.name}</span>
                  {awayPlayer.ranking && (
                    <Badge variant="secondary" className="bg-white/10">#{awayPlayer.ranking}</Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Recent Form (Last 10)</span>
                    <span className="text-white font-medium">{(awayPlayer.recentForm * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={awayPlayer.recentForm * 100} className="h-2" />
                </div>
                {awayPlayer.surfaceForm !== undefined && (
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Surface Form</span>
                      <span className="text-white font-medium">{(awayPlayer.surfaceForm * 100).toFixed(0)}%</span>
                    </div>
                    <Progress value={awayPlayer.surfaceForm * 100} className="h-2" />
                  </div>
                )}
                <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/10">
                  <div>
                    <div className="text-xs text-gray-400">H2H Wins</div>
                    <div className="text-xl font-bold text-white">{awayPlayer.h2hWins}</div>
                  </div>
                  {awayPlayer.fatigue !== undefined && (
                    <div>
                      <div className="text-xs text-gray-400">Matches (30d)</div>
                      <div className="text-xl font-bold text-white">{awayPlayer.fatigue}</div>
                    </div>
                  )}
                </div>
                {awayPlayer.injuries && awayPlayer.injuries.length > 0 && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-xs text-red-400 mb-1">Injury Concerns</div>
                    {awayPlayer.injuries.map((injury, i) => (
                      <Badge key={i} variant="destructive" className="mr-1 text-xs">{injury}</Badge>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* H2H Summary */}
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                <History className="w-5 h-5 text-blue-400" />
                Head-to-Head Record
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center gap-8">
                <div className="text-center">
                  <div className="text-4xl font-bold text-green-400">{homePlayer.h2hWins}</div>
                  <div className="text-sm text-gray-400">{homePlayer.name}</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-500">vs</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-blue-400">{awayPlayer.h2hWins}</div>
                  <div className="text-sm text-gray-400">{awayPlayer.name}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Key Factors Tab */}
        <TabsContent value="factors" className="space-y-3">
          {factors.map((factor, i) => (
            <Card key={i} className="bg-glass-card border-white/5">
              <CardContent className="p-4">
                <div
                  className="flex items-center justify-between cursor-pointer"
                  onClick={() => setExpandedFactor(expandedFactor === factor.name ? null : factor.name)}
                >
                  <div className="flex items-center gap-3">
                    <ImpactIcon impact={factor.impact} />
                    <div>
                      <div className="font-medium text-white">{factor.name}</div>
                      <div className="text-xs text-gray-400">Weight: {(factor.weight * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-32">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">Score</span>
                        <span className={factor.impact === 'positive' ? 'text-green-400' : factor.impact === 'negative' ? 'text-red-400' : 'text-gray-400'}>
                          {(factor.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <Progress
                        value={factor.score * 100}
                        className={`h-2 ${factor.impact === 'positive' ? '[&>div]:bg-green-500' : factor.impact === 'negative' ? '[&>div]:bg-red-500' : ''}`}
                      />
                    </div>
                    {expandedFactor === factor.name ? (
                      <ChevronUp className="w-4 h-4 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                </div>
                {expandedFactor === factor.name && (
                  <div className="mt-3 pt-3 border-t border-white/10">
                    <p className="text-sm text-gray-300">{factor.description}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        {/* Data Quality Tab */}
        <TabsContent value="quality" className="space-y-4">
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                <Shield className="w-5 h-5 text-violet-400" />
                Data Quality Assessment
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-center py-4">
                <div className="relative w-32 h-32">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="8"
                      className="text-white/10"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="url(#qualityGradient)"
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray={`${dataQuality.overall * 251.2} 251.2`}
                      transform="rotate(-90 50 50)"
                    />
                    <defs>
                      <linearGradient id="qualityGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor={dataQuality.overall >= 0.7 ? '#22c55e' : dataQuality.overall >= 0.5 ? '#eab308' : '#ef4444'} />
                        <stop offset="100%" stopColor={dataQuality.overall >= 0.8 ? '#10b981' : dataQuality.overall >= 0.6 ? '#f59e0b' : '#f97316'} />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold text-white">{(dataQuality.overall * 100).toFixed(0)}%</span>
                    <span className="text-xs text-gray-400">Overall</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-white/5 rounded-lg">
                  <div className="text-2xl font-bold text-white">{(dataQuality.sourceAgreement * 100).toFixed(0)}%</div>
                  <div className="text-xs text-gray-400">Source Agreement</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded-lg">
                  <div className="text-2xl font-bold text-white">{(dataQuality.freshness * 100).toFixed(0)}%</div>
                  <div className="text-xs text-gray-400">Data Freshness</div>
                </div>
                <div className="text-center p-3 bg-white/5 rounded-lg">
                  <div className="text-2xl font-bold text-white">{(dataQuality.coverage * 100).toFixed(0)}%</div>
                  <div className="text-xs text-gray-400">Coverage</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Action Buttons */}
      <div className="flex gap-4">
        {onPlaceBet && (
          <Button
            size="lg"
            className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600"
            onClick={onPlaceBet}
          >
            <DollarSign className="w-5 h-5 mr-2" />
            Place Bet @ {valueBet.odds.toFixed(2)}
          </Button>
        )}
        {onSave && (
          <Button
            size="lg"
            variant="outline"
            className="bg-white/5 border-white/20 hover:bg-white/10"
            onClick={onSave}
          >
            Save Analysis
          </Button>
        )}
      </div>
    </div>
  );
}

export default MatchAnalysisReport;
