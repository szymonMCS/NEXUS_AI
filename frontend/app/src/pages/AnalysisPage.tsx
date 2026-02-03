/**
 * AnalysisPage - Dynamic Match Analysis
 * 
 * Stadium atmosphere with:
 * - Live match visualization
 * - Dynamic probability bars
 * - Team statistics comparison
 * - Animated insights
 */

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SportSelector } from '@/components/SportSelector';
import type { SportId } from '@/lib/api';
import {
  Zap,
  RefreshCw,
  Brain,
  Target,
  Shield,
  Activity,
  ChevronRight,
  Trophy,
  BarChart3,
  Star,
  CheckCircle2,
} from 'lucide-react';

// Sample match data
const matchData = {
  league: 'NBA',
  leagueLogo: '🏀',
  home: {
    name: 'Lakers',
    color: '#FDB927',
    record: '24-15',
    rank: 3,
    form: ['W', 'W', 'L', 'W', 'W'],
    stats: { ppg: 118.5, defense: 112.3, fg: 48.2, threeP: 36.8 },
    injuries: ['A. Davis - Questionable'],
  },
  away: {
    name: 'Warriors',
    color: '#1D428A',
    record: '21-18',
    rank: 6,
    form: ['L', 'W', 'W', 'L', 'W'],
    stats: { ppg: 116.2, defense: 114.8, fg: 47.1, threeP: 38.5 },
    injuries: [],
  },
  prediction: {
    homeWin: 58,
    awayWin: 42,
    confidence: 76,
    method: 'Ensemble XGBoost',
  },
  odds: {
    home: 1.72,
    away: 2.15,
    edge: 8.5,
  },
  factors: [
    { name: 'Home Court', impact: 'high', value: '+12%', favor: 'home' },
    { name: 'Recent Form', impact: 'medium', value: '4-1', favor: 'home' },
    { name: '3PT Shooting', impact: 'high', value: '38.5%', favor: 'away' },
    { name: 'Defense', impact: 'medium', value: '112.3 Rating', favor: 'home' },
  ],
};

export function AnalysisPage() {
  const [selectedSport, setSelectedSport] = useState<SportId>('basketball');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState('prediction');

  const runAnalysis = () => {
    setIsAnalyzing(true);
    setTimeout(() => setIsAnalyzing(false), 1500);
  };

  return (
    <div className="min-h-screen bg-[#0a0f1c] text-white">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-[#0d1321]/95 backdrop-blur-md border-b border-blue-500/20">
        <div className="flex items-center justify-between h-16 px-4 lg:px-6">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
                Match Analysis
              </h1>
              <p className="text-xs text-slate-400">AI-Powered Predictions</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <SportSelector value={selectedSport} onChange={setSelectedSport} />
            <Button
              size="sm"
              onClick={runAnalysis}
              disabled={isAnalyzing}
              className="bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white font-semibold"
            >
              {isAnalyzing ? (
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              Analyze
            </Button>
          </div>
        </div>
      </header>

      <main className="p-4 lg:p-6">
        {/* Match Header Card */}
        <div className="sport-card p-6 mb-6 relative overflow-hidden">
          {/* Stadium Background Effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/5 via-transparent to-orange-500/5" />
          
          <div className="relative">
            {/* League Badge */}
            <div className="flex justify-center mb-6">
              <Badge className="bg-slate-800/80 border-slate-700 text-white px-4 py-1">
                <span className="mr-2">{matchData.leagueLogo}</span>
                {matchData.league}
              </Badge>
            </div>

            {/* Teams */}
            <div className="flex items-center justify-between max-w-3xl mx-auto">
              {/* Home Team */}
              <div className="text-center flex-1">
                <div 
                  className="w-24 h-24 mx-auto rounded-2xl flex items-center justify-center text-3xl font-bold mb-4 shadow-2xl"
                  style={{ 
                    background: `linear-gradient(135deg, ${matchData.home.color}22, ${matchData.home.color}44)`,
                    border: `2px solid ${matchData.home.color}`,
                  }}
                >
                  {matchData.home.name[0]}
                </div>
                <h2 className="text-2xl font-bold text-white">{matchData.home.name}</h2>
                <div className="flex items-center justify-center gap-2 mt-2">
                  <Badge variant="outline" className="border-slate-600 text-slate-400">
                    {matchData.home.record}
                  </Badge>
                  <span className="text-yellow-400">#{matchData.home.rank}</span>
                </div>
                {/* Form */}
                <div className="flex items-center justify-center gap-1 mt-3">
                  {matchData.home.form.map((result, i) => (
                    <span 
                      key={i}
                      className={cn(
                        "w-6 h-6 rounded text-xs font-bold flex items-center justify-center",
                        result === 'W' ? "bg-green-500 text-white" : "bg-red-500 text-white"
                      )}
                    >
                      {result}
                    </span>
                  ))}
                </div>
              </div>

              {/* VS & Prediction */}
              <div className="text-center px-8">
                <div className="text-4xl font-black text-slate-600 mb-4">VS</div>
                <div className="sport-border-gradient p-4">
                  <div className="text-3xl font-bold font-mono bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    {matchData.prediction.homeWin}%
                  </div>
                  <div className="text-xs text-slate-400 mt-1">Win Probability</div>
                </div>
              </div>

              {/* Away Team */}
              <div className="text-center flex-1">
                <div 
                  className="w-24 h-24 mx-auto rounded-2xl flex items-center justify-center text-3xl font-bold mb-4 shadow-2xl"
                  style={{ 
                    background: `linear-gradient(135deg, ${matchData.away.color}22, ${matchData.away.color}44)`,
                    border: `2px solid ${matchData.away.color}`,
                  }}
                >
                  {matchData.away.name[0]}
                </div>
                <h2 className="text-2xl font-bold text-white">{matchData.away.name}</h2>
                <div className="flex items-center justify-center gap-2 mt-2">
                  <Badge variant="outline" className="border-slate-600 text-slate-400">
                    {matchData.away.record}
                  </Badge>
                  <span className="text-yellow-400">#{matchData.away.rank}</span>
                </div>
                {/* Form */}
                <div className="flex items-center justify-center gap-1 mt-3">
                  {matchData.away.form.map((result, i) => (
                    <span 
                      key={i}
                      className={cn(
                        "w-6 h-6 rounded text-xs font-bold flex items-center justify-center",
                        result === 'W' ? "bg-green-500 text-white" : "bg-red-500 text-white"
                      )}
                    >
                      {result}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Value Bet Alert */}
            <div className="mt-6 p-4 bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-500 rounded-xl flex items-center justify-center">
                    <Trophy className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="text-sm text-green-400 font-semibold">Value Bet Detected</div>
                    <div className="text-white font-medium">{matchData.home.name} Moneyline @ {matchData.odds.home}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-green-400 font-mono">+{matchData.odds.edge}%</div>
                  <div className="text-xs text-slate-400">Edge</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="bg-slate-800/50 border border-slate-700 p-1">
            <TabsTrigger value="prediction" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-cyan-500 data-[state=active]:text-white">
              <Target className="w-4 h-4 mr-2" />
              Prediction
            </TabsTrigger>
            <TabsTrigger value="stats" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-orange-600 data-[state=active]:to-amber-500 data-[state=active]:text-white">
              <BarChart3 className="w-4 h-4 mr-2" />
              Statistics
            </TabsTrigger>
            <TabsTrigger value="factors" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-pink-500 data-[state=active]:text-white">
              <Star className="w-4 h-4 mr-2" />
              Key Factors
            </TabsTrigger>
          </TabsList>

          {/* Prediction Tab */}
          <TabsContent value="prediction" className="space-y-4 animate-slide-in-up">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Probability Bars */}
              <div className="sport-card p-5">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  Win Probability
                </h3>
                
                <div className="space-y-4">
                  {/* Home */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="font-semibold" style={{ color: matchData.home.color }}>
                        {matchData.home.name}
                      </span>
                      <span className="font-bold font-mono">{matchData.prediction.homeWin}%</span>
                    </div>
                    <div className="h-4 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{ 
                          width: `${matchData.prediction.homeWin}%`,
                          background: `linear-gradient(90deg, ${matchData.home.color}, ${matchData.home.color}88)`
                        }}
                      />
                    </div>
                  </div>

                  {/* Away */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="font-semibold" style={{ color: matchData.away.color }}>
                        {matchData.away.name}
                      </span>
                      <span className="font-bold font-mono">{matchData.prediction.awayWin}%</span>
                    </div>
                    <div className="h-4 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{ 
                          width: `${matchData.prediction.awayWin}%`,
                          background: `linear-gradient(90deg, ${matchData.away.color}, ${matchData.away.color}88)`
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <Brain className="w-4 h-4" />
                    <span>Method: <span className="text-white">{matchData.prediction.method}</span></span>
                  </div>
                </div>
              </div>

              {/* Confidence Gauge */}
              <div className="sport-card p-5">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-green-400" />
                  Model Confidence
                </h3>
                
                <div className="flex items-center justify-center py-6">
                  <div className="relative w-40 h-40">
                    {/* Gauge Background */}
                    <svg className="w-full h-full" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="40" fill="none" stroke="#1e293b" strokeWidth="10" />
                      <circle 
                        cx="50" 
                        cy="50" 
                        r="40" 
                        fill="none" 
                        stroke="url(#gaugeGradient)" 
                        strokeWidth="10"
                        strokeLinecap="round"
                        strokeDasharray={`${matchData.prediction.confidence * 2.51} 251.2`}
                        transform="rotate(-90 50 50)"
                      />
                      <defs>
                        <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#3b82f6" />
                          <stop offset="100%" stopColor="#22c55e" />
                        </linearGradient>
                      </defs>
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-4xl font-bold text-white">{matchData.prediction.confidence}%</span>
                      <span className="text-xs text-slate-400">Confidence</span>
                    </div>
                  </div>
                </div>

                <div className="text-center">
                  <Badge className="bg-green-500/20 text-green-400 border-0">
                    <CheckCircle2 className="w-3 h-3 mr-1" />
                    High Confidence Prediction
                  </Badge>
                </div>
              </div>
            </div>

            {/* Odds Comparison */}
            <div className="sport-card p-5">
              <h3 className="text-lg font-bold mb-4">Odds Comparison</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 bg-slate-800/50 rounded-xl text-center">
                  <div className="text-sm text-slate-400 mb-1">Market Odds</div>
                  <div className="text-2xl font-bold font-mono text-white">{matchData.odds.home}</div>
                </div>
                <div className="p-4 bg-blue-500/20 border border-blue-500/50 rounded-xl text-center">
                  <div className="text-sm text-blue-400 mb-1">Fair Odds (AI)</div>
                  <div className="text-2xl font-bold font-mono text-blue-400">1.58</div>
                </div>
                <div className="p-4 bg-green-500/20 border border-green-500/50 rounded-xl text-center">
                  <div className="text-sm text-green-400 mb-1">Edge</div>
                  <div className="text-2xl font-bold font-mono text-green-400">+{matchData.odds.edge}%</div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Stats Tab */}
          <TabsContent value="stats" className="animate-slide-in-up">
            <div className="sport-card p-6">
              <h3 className="text-lg font-bold mb-6">Team Statistics Comparison</h3>
              
              <div className="space-y-6">
                {[
                  { label: 'Points Per Game', home: matchData.home.stats.ppg, away: matchData.away.stats.ppg, max: 130 },
                  { label: 'Defensive Rating', home: matchData.home.stats.defense, away: matchData.away.stats.defense, max: 120 },
                  { label: 'Field Goal %', home: matchData.home.stats.fg, away: matchData.away.stats.fg, max: 55 },
                  { label: '3-Point %', home: matchData.home.stats.threeP, away: matchData.away.stats.threeP, max: 45 },
                ].map((stat) => (
                  <div key={stat.label}>
                    <div className="flex justify-between text-sm text-slate-400 mb-2">
                      <span>{stat.label}</span>
                      <span>{stat.home} vs {stat.away}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full rounded-full flex justify-end"
                          style={{ 
                            width: `${(stat.home / stat.max) * 50}%`,
                            backgroundColor: matchData.home.color
                          }}
                        />
                      </div>
                      <div className="w-16 text-center text-sm font-bold" style={{ color: matchData.home.color }}>
                        {stat.home}
                      </div>
                      <div className="w-16 text-center text-sm font-bold" style={{ color: matchData.away.color }}>
                        {stat.away}
                      </div>
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full rounded-full"
                          style={{ 
                            width: `${(stat.away / stat.max) * 50}%`,
                            backgroundColor: matchData.away.color
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Factors Tab */}
          <TabsContent value="factors" className="animate-slide-in-up">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {matchData.factors.map((factor, i) => (
                <div 
                  key={i}
                  className="sport-card p-4 flex items-center gap-4 group hover:border-blue-500/30 transition-colors"
                >
                  <div className={cn(
                    "w-12 h-12 rounded-xl flex items-center justify-center",
                    factor.impact === 'high' ? "bg-red-500/20 text-red-400" :
                    factor.impact === 'medium' ? "bg-yellow-500/20 text-yellow-400" :
                    "bg-blue-500/20 text-blue-400"
                  )}>
                    <Star className="w-6 h-6" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-white">{factor.name}</span>
                      <Badge variant="outline" className={cn(
                        "text-xs",
                        factor.impact === 'high' ? "border-red-500/30 text-red-400" :
                        factor.impact === 'medium' ? "border-yellow-500/30 text-yellow-400" :
                        "border-blue-500/30 text-blue-400"
                      )}>
                        {factor.impact}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-bold" style={{ 
                        color: factor.favor === 'home' ? matchData.home.color : matchData.away.color 
                      }}>
                        {factor.value}
                      </span>
                      <ChevronRight className="w-4 h-4 text-slate-600" />
                      <span className="text-sm" style={{ 
                        color: factor.favor === 'home' ? matchData.home.color : matchData.away.color 
                      }}>
                        Favors {factor.favor === 'home' ? matchData.home.name : matchData.away.name}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default AnalysisPage;
