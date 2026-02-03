/**
 * Dashboard - Professional Sports Betting Dashboard
 * 
 * Features:
 * - Animated floating background
 * - Forebet-style statistics tables
 * - Live match displays
 * - Odds comparison
 * - Predictions with confidence
 * - Team form indicators
 */

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FloatingBackground } from '@/components/FloatingBackground';
import { HandicapDisplay, HandicapCompact } from '@/components/HandicapDisplay';
import { ExtendedMarkets, ExtendedMarketStats } from '@/components/ExtendedMarkets';
import { 
  Activity, 
  Calendar,
  Trophy,
  Target,
  Star,
  Filter,
  Search,
  Bell,
  Settings,
  Flame,
  BarChart3,
  CheckCircle2,
  Scale,
  Plus,
  AlertTriangle,
  Shield,
  Swords
} from 'lucide-react';

// Navigation items
const navItems = [
  { icon: Activity, label: 'Live', count: 12, active: true },
  { icon: Calendar, label: 'Today', count: 48 },
  { icon: Trophy, label: 'Leagues', count: 15 },
  { icon: Star, label: 'Favorites', count: 8 },
  { icon: BarChart3, label: 'Statistics' },
];

// Live matches
const liveMatches = [
  {
    id: 1,
    league: 'Champions League',
    leagueIcon: '⚽',
    home: { name: 'Ajax', score: 1, color: '#D2122E' },
    away: { name: 'Olympiacos', score: 2, color: '#E00000' },
    time: "84'",
    odds: { home: 8.19, draw: 1.12, away: 2.88 },
    prediction: { result: '2', prob: 98, value: true },
    stats: { possession: [45, 55], shots: [8, 12] },
  },
  {
    id: 2,
    league: 'NBA',
    leagueIcon: '🏀',
    home: { name: 'Lakers', score: 98, color: '#FDB927' },
    away: { name: 'Warriors', score: 102, color: '#1D428A' },
    time: 'Q4 2:45',
    odds: { home: 2.15, away: 1.75 },
    prediction: { result: '2', prob: 65, value: false },
    stats: { possession: null, shots: null },
  },
];

// Upcoming matches with predictions
const upcomingMatches = [
  {
    id: 1,
    league: 'Premier League',
    leagueIcon: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    home: { name: 'Man City', form: ['W', 'W', 'D', 'W', 'W'], color: '#6CABDD' },
    away: { name: 'Arsenal', form: ['W', 'L', 'W', 'D', 'W'], color: '#EF0107' },
    time: 'Today 18:30',
    odds: { home: 1.65, draw: 3.80, away: 5.20 },
    prediction: { home: 58, draw: 24, away: 18 },
    confidence: 82,
    valueBet: { selection: 'Man City', edge: '+12%' },
  },
  {
    id: 2,
    league: 'La Liga',
    leagueIcon: '🇪🇸',
    home: { name: 'Real Madrid', form: ['W', 'W', 'W', 'D', 'W'], color: '#FEBE10' },
    away: { name: 'Barcelona', form: ['W', 'W', 'L', 'W', 'D'], color: '#A50044' },
    time: 'Today 21:00',
    odds: { home: 2.10, draw: 3.40, away: 3.30 },
    prediction: { home: 42, draw: 28, away: 30 },
    confidence: 76,
    valueBet: null,
  },
  {
    id: 3,
    league: 'NBA',
    leagueIcon: '🏀',
    home: { name: 'Celtics', form: ['W', 'W', 'W', 'L', 'W'], color: '#007A33' },
    away: { name: 'Heat', form: ['L', 'W', 'L', 'W', 'W'], color: '#98002E' },
    time: 'Tomorrow 01:30',
    odds: { home: 1.45, away: 2.80 },
    prediction: { home: 72, away: 28 },
    confidence: 78,
    valueBet: { selection: 'Celtics', edge: '+8%' },
  },
  {
    id: 4,
    league: 'ATP - Australian Open',
    leagueIcon: '🎾',
    home: { name: 'Alcaraz', form: ['W', 'W', 'W', 'W', 'L'], color: '#FFD700' },
    away: { name: 'Djokovic', form: ['W', 'W', 'W', 'W', 'W'], color: '#C0C0C0' },
    time: 'Tomorrow 09:00',
    odds: { home: 1.85, away: 1.95 },
    prediction: { home: 48, away: 52 },
    confidence: 71,
    valueBet: { selection: 'Djokovic', edge: '+15%' },
  },
];

// Statistics for sidebar
const topStats = [
  { label: 'Today\'s Predictions', value: '47', change: '+12%', positive: true },
  { label: 'Win Rate', value: '68.5%', change: '+5.2%', positive: true },
  { label: 'Avg Edge', value: '8.4%', change: '+1.2%', positive: true },
  { label: 'Profit Today', value: '+$245', change: '+18%', positive: true },
];

// Recent results
const recentResults = [
  { match: 'Man City vs Arsenal', result: 'WIN', odds: 1.65, profit: '+65%', time: '2h ago' },
  { match: 'Real Madrid vs Barcelona', result: 'LOSS', odds: 2.10, profit: '-100%', time: '5h ago' },
  { match: 'Celtics vs Heat', result: 'WIN', odds: 1.45, profit: '+45%', time: '8h ago' },
];

export function Dashboard() {
  const [activeTab, setActiveTab] = useState('live');
  const [, setSelectedMatch] = useState<number | null>(null);

  return (
    <div className="min-h-screen relative">
      {/* Animated Floating Background */}
      <FloatingBackground />
      
      {/* Content wrapper */}
      <div className="relative z-10">
        {/* Top Navigation */}
        <header className="sticky top-0 z-50 bg-[#0a0e1a]/80 backdrop-blur-xl border-b border-white/10">
          <div className="flex items-center justify-between h-16 px-4 lg:px-6">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center shadow-lg shadow-[#00d4ff]/30">
                  <Activity className="w-6 h-6 text-black" />
                </div>
                <span className="text-xl font-bold text-white">NEXUS<span className="text-[#00d4ff]">AI</span></span>
              </div>
              
              <nav className="hidden lg:flex items-center gap-1">
                {navItems.map((item) => (
                  <button
                    key={item.label}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      item.active 
                        ? 'bg-[#00d4ff]/10 text-[#00d4ff]' 
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                    {item.count && (
                      <span className="px-1.5 py-0.5 bg-white/10 rounded text-xs">{item.count}</span>
                    )}
                  </button>
                ))}
              </nav>
            </div>

            <div className="flex items-center gap-3">
              <div className="relative hidden md:block">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input 
                  type="text" 
                  placeholder="Search matches..."
                  className="w-64 pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-[#00d4ff]/50"
                />
              </div>
              <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white relative">
                <Bell className="w-5 h-5" />
                <span className="absolute top-1 right-1 w-2 h-2 bg-[#00ff88] rounded-full" />
              </Button>
              <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
                <Settings className="w-5 h-5" />
              </Button>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center text-black font-bold text-sm">
                G
              </div>
            </div>
          </div>
        </header>

        <div className="flex">
          {/* Sidebar */}
          <aside className="hidden lg:block w-64 border-r border-white/10 min-h-[calc(100vh-64px)] p-4">
            {/* Quick Stats */}
            <div className="space-y-3 mb-6">
              {topStats.map((stat, i) => (
                <div key={i} className="p-3 bg-[#0f1623]/80 border border-white/10 rounded-xl">
                  <div className="text-xs text-gray-500 mb-1">{stat.label}</div>
                  <div className="flex items-center justify-between">
                    <span className="text-lg font-bold text-white font-mono">{stat.value}</span>
                    <span className={`text-xs ${stat.positive ? 'text-[#00ff88]' : 'text-[#ff3860]'}`}>
                      {stat.change}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Hot Predictions */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                <Flame className="w-4 h-4 text-[#ff9500]" />
                Hot Predictions
              </h3>
              <div className="space-y-2">
                {[
                  { match: 'Man City vs Arsenal', confidence: 82 },
                  { match: 'Lakers vs Warriors', confidence: 75 },
                  { match: 'Alcaraz vs Djokovic', confidence: 71 },
                ].map((item, i) => (
                  <div key={i} className="flex items-center justify-between p-2 hover:bg-white/5 rounded-lg cursor-pointer transition-colors">
                    <span className="text-sm text-gray-300">{item.match}</span>
                    <Badge className="bg-[#00ff88]/20 text-[#00ff88] text-xs border-0">{item.confidence}%</Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Results */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#00ff88]" />
                Recent Results
              </h3>
              <div className="space-y-2">
                {recentResults.map((result, i) => (
                  <div key={i} className="p-2 bg-[#0f1623]/60 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400 truncate">{result.match}</span>
                      <Badge className={`text-xs border-0 ${
                        result.result === 'WIN' 
                          ? 'bg-[#00ff88]/20 text-[#00ff88]' 
                          : 'bg-[#ff3860]/20 text-[#ff3860]'
                      }`}>
                        {result.result}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">Odds: {result.odds}</span>
                      <span className={result.profit.startsWith('+') ? 'text-[#00ff88]' : 'text-[#ff3860]'}>
                        {result.profit}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Leagues */}
            <div>
              <h3 className="text-sm font-bold text-white mb-3">Top Leagues</h3>
              <div className="space-y-1">
                {['Champions League', 'Premier League', 'NBA', 'La Liga', 'ATP'].map((league, i) => (
                  <div key={i} className="flex items-center gap-2 p-2 hover:bg-white/5 rounded-lg cursor-pointer text-sm text-gray-400 hover:text-white transition-colors">
                    <div className="w-2 h-2 rounded-full bg-[#00d4ff]" />
                    {league}
                  </div>
                ))}
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1 p-4 lg:p-6">
            {/* Welcome Banner */}
            <div className="mb-6 p-6 rounded-xl bg-gradient-to-r from-[#00d4ff]/10 via-[#b829dd]/10 to-[#00ff88]/10 border border-white/10">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-bold text-white mb-1">Welcome back, Guest!</h1>
                  <p className="text-gray-400">You have 12 live matches and 48 upcoming predictions today.</p>
                </div>
                <div className="hidden md:flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-sm text-gray-500">Current Streak</div>
                    <div className="flex items-center gap-1 text-[#00ff88]">
                      <Flame className="w-4 h-4" />
                      <span className="font-bold">7 Wins</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-500">Total Profit</div>
                    <div className="font-bold text-white">+$12,450</div>
                  </div>
                </div>
              </div>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <div className="flex items-center justify-between">
                <TabsList className="bg-[#0f1623]/80 border border-white/10">
                  <TabsTrigger value="live" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
                    <Activity className="w-4 h-4 mr-2" />
                    Live
                  </TabsTrigger>
                  <TabsTrigger value="upcoming" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
                    <Calendar className="w-4 h-4 mr-2" />
                    Upcoming
                  </TabsTrigger>
                  <TabsTrigger value="handicaps" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
                    <Scale className="w-4 h-4 mr-2" />
                    Handicaps
                  </TabsTrigger>
                  <TabsTrigger value="markets" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
                    <AlertTriangle className="w-4 h-4 mr-2" />
                    Markets
                  </TabsTrigger>
                  <TabsTrigger value="predictions" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
                    <Target className="w-4 h-4 mr-2" />
                    My Predictions
                  </TabsTrigger>
                </TabsList>

                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" className="border-white/10 text-gray-400 hover:text-white">
                    <Filter className="w-4 h-4 mr-2" />
                    Filter
                  </Button>
                </div>
              </div>

              {/* Live Matches */}
              <TabsContent value="live" className="space-y-4">
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  {liveMatches.map((match) => (
                    <div key={match.id} className="match-card live">
                      <div className="flex items-center justify-between mb-4">
                        <Badge className="bg-white/5 text-gray-400 border-0">
                          <span className="mr-1">{match.leagueIcon}</span>
                          {match.league}
                        </Badge>
                        <div className="live-indicator">{match.time}</div>
                      </div>

                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-14 h-14 rounded-xl flex items-center justify-center text-xl font-bold"
                            style={{ background: `${match.home.color}30`, color: match.home.color }}
                          >
                            {match.home.name[0]}
                          </div>
                          <div>
                            <div className="text-lg font-bold text-white">{match.home.name}</div>
                            <div className="score-display">{match.home.score}</div>
                          </div>
                        </div>

                        <div className="text-center px-4">
                          <div className="text-gray-600 font-bold text-xl">VS</div>
                        </div>

                        <div className="flex items-center gap-3 text-right">
                          <div>
                            <div className="text-lg font-bold text-white">{match.away.name}</div>
                            <div className="score-display">{match.away.score}</div>
                          </div>
                          <div 
                            className="w-14 h-14 rounded-xl flex items-center justify-center text-xl font-bold"
                            style={{ background: `${match.away.color}30`, color: match.away.color }}
                          >
                            {match.away.name[0]}
                          </div>
                        </div>
                      </div>

                      {/* Match Stats */}
                      {match.stats.possession && (
                        <div className="grid grid-cols-2 gap-4 mb-4 p-3 bg-black/20 rounded-lg">
                          <div>
                            <div className="flex justify-between text-xs text-gray-500 mb-1">
                              <span>Possession</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold" style={{ color: match.home.color }}>{match.stats.possession[0]}%</span>
                              <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div 
                                  className="h-full rounded-full transition-all"
                                  style={{ 
                                    width: `${match.stats.possession[0]}%`,
                                    background: `linear-gradient(90deg, ${match.home.color}, ${match.away.color})`
                                  }}
                                />
                              </div>
                              <span className="text-sm font-bold" style={{ color: match.away.color }}>{match.stats.possession[1]}%</span>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-xs text-gray-500 mb-1">
                              <span>Shots</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold" style={{ color: match.home.color }}>{match.stats.shots[0]}</span>
                              <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div 
                                  className="h-full rounded-full"
                                  style={{ 
                                    width: `${(match.stats.shots[0] / (match.stats.shots[0] + match.stats.shots[1])) * 100}%`,
                                    background: match.home.color
                                  }}
                                />
                              </div>
                              <span className="text-sm font-bold" style={{ color: match.away.color }}>{match.stats.shots[1]}</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Odds */}
                      <div className="flex justify-center gap-3">
                        <div className="odds-box">
                          <span className="odds-value" style={{ color: match.home.color }}>{match.odds.home}</span>
                          <span className="odds-label">1</span>
                        </div>
                        {match.odds.draw && (
                          <div className="odds-box">
                            <span className="odds-value">{match.odds.draw}</span>
                            <span className="odds-label">X</span>
                          </div>
                        )}
                        <div className="odds-box">
                          <span className="odds-value" style={{ color: match.away.color }}>{match.odds.away}</span>
                          <span className="odds-label">2</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </TabsContent>

              {/* Upcoming Matches - Forebet Style */}
              <TabsContent value="upcoming">
                <div className="bg-[#0f1623]/80 border border-white/10 rounded-xl overflow-hidden">
                  <div className="grid grid-cols-12 gap-4 p-4 bg-white/5 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    <div className="col-span-4">Match</div>
                    <div className="col-span-2">Form</div>
                    <div className="col-span-3">Prediction</div>
                    <div className="col-span-2">Odds</div>
                    <div className="col-span-1">Value</div>
                  </div>

                  <div className="divide-y divide-white/5">
                    {upcomingMatches.map((match) => (
                      <div 
                        key={match.id} 
                        className="grid grid-cols-12 gap-4 p-4 items-center hover:bg-white/5 transition-colors cursor-pointer group"
                        onClick={() => setSelectedMatch(match.id)}
                      >
                        {/* Match Info */}
                        <div className="col-span-4">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">{match.leagueIcon}</span>
                            <span className="text-xs text-gray-500">{match.league}</span>
                            <span className="text-xs text-[#00d4ff]">{match.time}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div 
                                className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
                                style={{ background: `${match.home.color}30`, color: match.home.color }}
                              >
                                {match.home.name[0]}
                              </div>
                              <span className="font-semibold text-white">{match.home.name}</span>
                            </div>
                            <span className="text-gray-600 text-sm">vs</span>
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-white">{match.away.name}</span>
                              <div 
                                className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
                                style={{ background: `${match.away.color}30`, color: match.away.color }}
                              >
                                {match.away.name[0]}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Form */}
                        <div className="col-span-2">
                          <div className="flex items-center gap-4">
                            <div className="form-indicator">
                              {match.home.form.map((r, i) => (
                                <span key={i} className={`form-dot ${r.toLowerCase()}`}>{r}</span>
                              ))}
                            </div>
                            <span className="text-gray-600">-</span>
                            <div className="form-indicator">
                              {match.away.form.map((r, i) => (
                                <span key={i} className={`form-dot ${r.toLowerCase()}`}>{r}</span>
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* Prediction Bars */}
                        <div className="col-span-3 space-y-1">
                          <div className="prob-bar-container">
                            <div className="prob-bar">
                              <div 
                                className="prob-bar-fill home"
                                style={{ width: `${match.prediction.home}%` }}
                              />
                            </div>
                            <span className="prob-value text-[#00d4ff]">{match.prediction.home}%</span>
                          </div>
                          {match.prediction.draw && (
                            <div className="prob-bar-container">
                              <div className="prob-bar">
                                <div 
                                  className="prob-bar-fill draw"
                                  style={{ width: `${match.prediction.draw}%` }}
                                />
                              </div>
                              <span className="prob-value text-[#ff9500]">{match.prediction.draw}%</span>
                            </div>
                          )}
                          <div className="prob-bar-container">
                            <div className="prob-bar">
                              <div 
                                className="prob-bar-fill away"
                                style={{ width: `${match.prediction.away}%` }}
                              />
                            </div>
                            <span className="prob-value text-[#00ff88]">{match.prediction.away}%</span>
                          </div>
                        </div>

                        {/* Odds */}
                        <div className="col-span-2">
                          <div className="flex gap-2">
                            <div className="text-center">
                              <div className="text-sm font-bold text-white font-mono">{match.odds.home}</div>
                              <div className="text-[10px] text-gray-500">1</div>
                            </div>
                            {match.odds.draw && (
                              <div className="text-center">
                                <div className="text-sm font-bold text-gray-400 font-mono">{match.odds.draw}</div>
                                <div className="text-[10px] text-gray-500">X</div>
                              </div>
                            )}
                            <div className="text-center">
                              <div className="text-sm font-bold text-white font-mono">{match.odds.away}</div>
                              <div className="text-[10px] text-gray-500">2</div>
                            </div>
                          </div>
                        </div>

                        {/* Value */}
                        <div className="col-span-1">
                          {match.valueBet ? (
                            <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-[#00ff88]/30">
                              {match.valueBet.edge}
                            </Badge>
                          ) : (
                            <span className="text-gray-600">-</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>

              {/* Handicaps Tab */}
              <TabsContent value="handicaps" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Featured Handicap Match 1 */}
                  <HandicapDisplay 
                    handicaps={[
                      { type: 'asian', line: -1.5, homeOdds: 2.80, awayOdds: 1.40, homeProb: 32, awayProb: 68, homeEdge: -0.10, awayEdge: 0.05 },
                      { type: 'asian', line: -1, homeOdds: 2.20, awayOdds: 1.65, homeProb: 42, awayProb: 58, homeEdge: -0.08, awayEdge: 0.04 },
                      { type: 'asian', line: -0.5, homeOdds: 1.85, awayOdds: 1.95, homeProb: 52, awayProb: 48, homeEdge: -0.04, awayEdge: -0.06 },
                      { type: 'asian', line: 0, homeOdds: 1.60, awayOdds: 2.20, homeProb: 58, awayProb: 42, homeEdge: -0.07, awayEdge: -0.08 },
                      { type: 'asian', line: 0.5, homeOdds: 1.40, awayOdds: 2.80, homeProb: 68, awayProb: 32, homeEdge: -0.05, awayEdge: -0.10 },
                      { type: 'european', line: '0:0', homeOdds: 2.40, drawOdds: 3.20, awayOdds: 2.80, homeProb: 38, drawProb: 28, awayProb: 34 },
                      { type: 'european', line: '1:0', homeOdds: 1.55, drawOdds: 3.80, awayOdds: 5.50, homeProb: 62, drawProb: 18, awayProb: 20 },
                    ]}
                    sport="football"
                  />
                  
                  {/* Featured Handicap Match 2 */}
                  <HandicapDisplay 
                    handicaps={[
                      { type: 'asian', line: -0.5, homeOdds: 1.72, awayOdds: 2.10, homeProb: 58, awayProb: 42, homeEdge: 0.00, awayEdge: -0.12, recommendation: 'home' },
                      { type: 'asian', line: 0, homeOdds: 1.45, awayOdds: 2.60, homeProb: 65, awayProb: 35, homeEdge: -0.06, awayEdge: -0.09 },
                      { type: 'asian', line: 0.5, homeOdds: 1.25, awayOdds: 3.60, homeProb: 76, awayProb: 24, homeEdge: -0.05, awayEdge: -0.14 },
                      { type: 'asian', line: -1, homeOdds: 2.10, awayOdds: 1.70, homeProb: 45, awayProb: 55, homeEdge: -0.05, awayEdge: -0.06 },
                      { type: 'european', line: '0:0', homeOdds: 1.85, drawOdds: 3.40, awayOdds: 3.80, homeProb: 52, drawProb: 28, awayProb: 20 },
                      { type: 'european', line: '1:0', homeOdds: 1.25, drawOdds: 5.00, awayOdds: 9.00, homeProb: 76, drawProb: 15, awayProb: 9 },
                    ]}
                    sport="football"
                  />
                </div>
                
                {/* Compact Handicap Lines */}
                <div className="bg-[#0f1623]/80 border border-white/10 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Plus className="w-5 h-5 text-[#00ff88]" />
                    Quick Handicap Lines
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                    <div className="p-3 bg-white/5 rounded-lg text-center">
                      <div className="text-xs text-gray-500 mb-1">Man City -1.5</div>
                      <HandicapCompact line={-1.5} odds={2.80} probability={32} />
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg text-center">
                      <div className="text-xs text-gray-500 mb-1">Lakers -5.5</div>
                      <HandicapCompact line={-5.5} odds={1.90} probability={52} recommendation />
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg text-center">
                      <div className="text-xs text-gray-500 mb-1">Celtics -7.5</div>
                      <HandicapCompact line={-7.5} odds={1.85} probability={54} recommendation />
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg text-center">
                      <div className="text-xs text-gray-500 mb-1">Real +1.5</div>
                      <HandicapCompact line={1.5} odds={1.40} probability={68} />
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg text-center">
                      <div className="text-xs text-gray-500 mb-1">Arsenal +0.5</div>
                      <HandicapCompact line={0.5} odds={2.20} probability={45} />
                    </div>
                    <div className="p-3 bg-white/5 rounded-lg text-center">
                      <div className="text-xs text-gray-500 mb-1">Djokovic -1.5</div>
                      <HandicapCompact line={-1.5} odds={2.10} probability={48} />
                    </div>
                  </div>
                </div>
              </TabsContent>

              {/* Extended Markets Tab */}
              <TabsContent value="markets" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Football Extended Markets */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-2xl">⚽</span>
                      <h3 className="text-lg font-semibold text-white">Football Markets</h3>
                    </div>
                    <ExtendedMarkets 
                      sport="football"
                      cardMarkets={[
                        { type: 'total_cards', line: 3.5, overOdds: 1.85, underOdds: 1.95, overProb: 52, underProb: 48, expectedCards: 4.2 },
                        { type: 'total_cards', line: 4.5, overOdds: 2.20, underOdds: 1.65, overProb: 42, underProb: 58, expectedCards: 4.2 },
                        { type: 'asian_cards', line: 0.5, homeOdds: 1.90, awayOdds: 1.90, homeProb: 50, awayProb: 50 },
                      ]}
                      foulMarkets={[
                        { type: 'total_fouls', line: 21.5, overOdds: 1.85, underOdds: 1.95, overProb: 48, underProb: 52, expectedFouls: 21.0 },
                        { type: 'foul_handicap', line: 2, homeOdds: 1.75, awayOdds: 2.05, expectedFouls: 21.0 },
                      ]}
                      goalDiffMarkets={[
                        { type: 'exact_margin', selection: 'Home by 1', odds: 3.40, probability: 28, edge: 0.02 },
                        { type: 'win_by', selection: 'Home by 2+', odds: 4.20, probability: 22, edge: -0.05 },
                        { type: 'exact_margin', selection: 'Draw', odds: 3.60, probability: 25, edge: 0.01 },
                        { type: 'exact_margin', selection: 'Away by 1', odds: 5.00, probability: 18, edge: -0.08 },
                        { type: 'any_win', selection: 'Any Home Win', odds: 1.75, probability: 55, edge: 0.03 },
                      ]}
                    />
                    <ExtendedMarketStats 
                      sport="football"
                      stats={{ expectedCards: 4.2, expectedFouls: 21.0 }}
                    />
                  </div>

                  {/* Basketball Extended Markets */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-2xl">🏀</span>
                      <h3 className="text-lg font-semibold text-white">Basketball Markets</h3>
                    </div>
                    <ExtendedMarkets 
                      sport="basketball"
                      winMarginMarkets={[
                        { range: '1-2', homeOdds: 4.50, awayOdds: 5.00, homeProb: 22, awayProb: 18 },
                        { range: '3-6', homeOdds: 3.80, awayOdds: 4.20, homeProb: 26, awayProb: 22 },
                        { range: '7-9', homeOdds: 4.00, awayOdds: 4.50, homeProb: 24, awayProb: 20 },
                        { range: '10-14', homeOdds: 3.40, awayOdds: 3.80, homeProb: 28, awayProb: 24, recommendation: 'home' },
                        { range: '15+', homeOdds: 2.80, awayOdds: 3.20, homeProb: 35, awayProb: 30 },
                      ]}
                      foulMarkets={[
                        { type: 'team_fouls', line: 18.5, overOdds: 1.90, underOdds: 1.90, overProb: 50, underProb: 50, expectedFouls: 20.5 },
                        { type: 'team_fouls', line: 20.5, overOdds: 2.10, underOdds: 1.70, overProb: 45, underProb: 55, expectedFouls: 20.5 },
                      ]}
                    />
                    <ExtendedMarketStats 
                      sport="basketball"
                      stats={{ avgMargin: 8.5 }}
                    />
                  </div>

                  {/* Tennis Extended Markets */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-2xl">🎾</span>
                      <h3 className="text-lg font-semibold text-white">Tennis Markets</h3>
                    </div>
                    <ExtendedMarkets 
                      sport="tennis"
                      setBettingMarkets={[
                        { score: '3-0', homeOdds: 2.80, awayOdds: 6.00, homeProb: 35, awayProb: 15 },
                        { score: '3-1', homeOdds: 3.40, awayOdds: 4.50, homeProb: 28, awayProb: 20 },
                        { score: '3-2', homeOdds: 5.00, awayOdds: 3.80, homeProb: 18, awayProb: 24 },
                      ]}
                      gameHandicapMarkets={[
                        { line: -4.5, homeOdds: 1.85, awayOdds: 1.95, homeProb: 54, awayProb: 46 },
                        { line: -3.5, homeOdds: 1.75, awayOdds: 2.05, homeProb: 58, awayProb: 42, recommendation: 'home' },
                        { line: -2.5, homeOdds: 1.65, awayOdds: 2.20, homeProb: 62, awayProb: 38, recommendation: 'home' },
                        { line: 2.5, homeOdds: 1.35, awayOdds: 3.00, homeProb: 74, awayProb: 26 },
                      ]}
                    />
                    <ExtendedMarketStats 
                      sport="tennis"
                      stats={{ likelySetScore: '3-1' }}
                    />
                  </div>

                  {/* Quick Stats Summary */}
                  <div className="bg-[#0f1623]/80 border border-white/10 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Shield className="w-5 h-5 text-[#00d4ff]" />
                      Market Insights
                    </h3>
                    <div className="space-y-4">
                      <div className="p-3 bg-white/5 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="w-4 h-4 text-[#ff9500]" />
                          <span className="text-sm font-medium text-white">High Card Expectation</span>
                        </div>
                        <p className="text-xs text-gray-400">
                          Man City vs Arsenal: Derby atmosphere expected with 4.2 cards (avg: 3.5)
                        </p>
                      </div>
                      <div className="p-3 bg-white/5 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Swords className="w-4 h-4 text-[#00ff88]" />
                          <span className="text-sm font-medium text-white">Foul Intensity</span>
                        </div>
                        <p className="text-xs text-gray-400">
                          Lakers game: Physical matchup projected with 21+ fouls
                        </p>
                      </div>
                      <div className="p-3 bg-white/5 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Target className="w-4 h-4 text-[#b829dd]" />
                          <span className="text-sm font-medium text-white">Win Margin Value</span>
                        </div>
                        <p className="text-xs text-gray-400">
                          Celtics by 10-14 points offers best value at 3.40 odds
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="predictions">
                <div className="text-center py-20">
                  <Trophy className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-white mb-2">No Active Predictions</h3>
                  <p className="text-gray-400 mb-6">Start tracking your bets and view your history here</p>
                  <Button 
                    className="bg-gradient-to-r from-[#00d4ff] to-[#00ff88] text-black font-semibold"
                    onClick={() => setActiveTab('upcoming')}
                  >
                    Browse Matches
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </main>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
