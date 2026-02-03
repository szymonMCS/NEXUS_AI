/**
 * LandingPage - Professional Sports Betting Interface
 * 
 * Inspired by bitbet.io, forebet.com, softivuspro/oddsx
 * Features:
 * - Animated floating background (coins, trophies, chips)
 * - Stadium atmosphere hero
 * - Live match ticker
 * - Advanced statistics preview
 * - Model performance metrics
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ElegantSportBackground } from '@/components/ElegantSportBackground';
import { 
  TrendingUp, 
  Activity, 
  BarChart3, 
  Shield, 
  Zap,
  ChevronRight,
  Play,
  Trophy,
  Target,
  Users,
  Globe,
  Brain,
  Cpu,
  LineChart,
  Award,
  Flame,
  ArrowUpRight,
  Sparkles,
  Percent,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Circle,
  Hexagon,
  Triangle
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Sample live matches for ticker
const liveMatches = [
  { league: 'UCL', home: 'Ajax', away: 'Olympiacos', score: '1-2', time: "84'", status: 'live' },
  { league: 'EPL', home: 'Liverpool', away: 'Chelsea', score: '2-1', time: "67'", status: 'live' },
  { league: 'NBA', home: 'Lakers', away: 'Warriors', score: '98-102', time: 'Q4', status: 'live' },
  { league: 'UCL', home: 'Real Madrid', away: 'Bayern', score: '3-2', time: "91'", status: 'live' },
  { league: 'NBA', home: 'Celtics', away: 'Heat', score: '78-76', time: 'Q3', status: 'live' },
];

// Featured matches
const featuredMatches = [
  {
    league: 'Champions League',
    leagueIcon: '⚽',
    home: { name: 'Man City', odds: 1.65, color: '#6CABDD' },
    away: { name: 'Real Madrid', odds: 4.20, color: '#FEBE10' },
    draw: 3.80,
    prediction: '1',
    confidence: 78,
    time: 'Today 20:45',
  },
  {
    league: 'NBA',
    leagueIcon: '🏀',
    home: { name: 'Lakers', odds: 1.85, color: '#FDB927' },
    away: { name: 'Warriors', odds: 1.95, color: '#1D428A' },
    prediction: '1',
    confidence: 65,
    time: 'Today 23:00',
  },
  {
    league: 'ATP Masters',
    leagueIcon: '🎾',
    home: { name: 'Alcaraz', odds: 1.72, color: '#FFD700' },
    away: { name: 'Djokovic', odds: 2.10, color: '#C0C0C0' },
    prediction: '2',
    confidence: 72,
    time: 'Tomorrow 14:00',
  },
];

// Statistics
const stats = [
  { label: 'Active Predictions', value: '2,847', icon: Activity },
  { label: 'Win Rate', value: '68.5%', icon: Target },
  { label: 'Active Users', value: '12.5K', icon: Users },
  { label: 'Countries', value: '45+', icon: Globe },
];

// Model metrics data
const modelMetrics = [
  { name: 'Accuracy', value: 0.685, benchmark: 0.52, description: 'Overall prediction accuracy' },
  { name: 'Precision', value: 0.712, benchmark: 0.55, description: 'True positives ratio' },
  { name: 'Recall', value: 0.648, benchmark: 0.50, description: 'Coverage of winners' },
  { name: 'F1 Score', value: 0.678, benchmark: 0.52, description: 'Harmonic mean' },
];

// Insights data
const insights = [
  { 
    type: 'opportunity' as const,
    title: 'Value Bet Detected',
    description: 'Underdog showing strong momentum in recent matches',
    metric: { label: 'Expected Value', value: '+12.5%', change: 8.3 },
    confidence: 82 
  },
  { 
    type: 'positive' as const,
    title: 'Home Advantage Strong',
    description: 'Home team has 85% win rate in last 10 matches',
    metric: { label: 'Win Rate', value: '85%', change: 15 },
    confidence: 78 
  },
  { 
    type: 'warning' as const,
    title: 'Key Player Injured',
    description: 'Star striker out due to muscle injury',
    metric: { label: 'Impact', value: '-15%', change: -15 },
    confidence: 95 
  },
];

// Recent performance data
const recentPredictions = [
  { match: 'Man City vs Arsenal', result: 'WIN', odds: 1.75, profit: '+75%', date: '2h ago' },
  { match: 'Lakers vs Warriors', result: 'WIN', odds: 1.95, profit: '+95%', date: '5h ago' },
  { match: 'Barcelona vs Madrid', result: 'LOSS', odds: 2.10, profit: '-100%', date: '1d ago' },
  { match: 'Djokovic vs Alcaraz', result: 'WIN', odds: 1.85, profit: '+85%', date: '1d ago' },
  { match: 'Celtics vs Heat', result: 'WIN', odds: 1.65, profit: '+65%', date: '2d ago' },
];

// KPI Cards data
const kpiData = [
  { 
    label: 'Total Profit', 
    value: '+$12,450', 
    subValue: 'Last 30 days',
    change: 23.5, 
    changeType: 'positive' as const,
    sparklineData: [65, 68, 72, 70, 75, 78, 82, 80, 85, 88, 92, 95],
    icon: <TrendingUp className="w-5 h-5" />
  },
  { 
    label: 'Avg Odds', 
    value: '1.82', 
    subValue: 'Last 100 bets',
    change: 5.2, 
    changeType: 'positive' as const,
    sparklineData: [1.75, 1.78, 1.80, 1.79, 1.81, 1.82, 1.85, 1.83, 1.84, 1.82],
    icon: <Percent className="w-5 h-5" />
  },
  { 
    label: 'Win Streak', 
    value: '7', 
    subValue: 'Consecutive wins',
    change: 12, 
    changeType: 'positive' as const,
    sparklineData: [3, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9],
    icon: <Flame className="w-5 h-5" />
  },
  { 
    label: 'ROI', 
    value: '+18.5%', 
    subValue: 'Return on Investment',
    change: 3.2, 
    changeType: 'positive' as const,
    sparklineData: [12, 13, 14, 13, 15, 16, 15, 17, 18, 18.5],
    icon: <LineChart className="w-5 h-5" />
  },
];

// Sports categories - using abstract icons instead of emojis
const sportsCategories = [
  { name: 'Football', icon: Circle, active: 1247, accuracy: 71.2 },
  { name: 'Basketball', icon: Hexagon, active: 856, accuracy: 68.5 },
  { name: 'Tennis', icon: Triangle, active: 432, accuracy: 66.8 },
  { name: 'Esports', icon: Cpu, active: 312, accuracy: 64.2 },
];

export function LandingPage() {
  const navigate = useNavigate();
  const [animatedStats, setAnimatedStats] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedStats(true), 500);
    return () => clearTimeout(timer);
  }, []);

  // Helper for metric colors
  const getMetricColor = (value: number) => {
    if (value >= 0.75) return 'bg-[#00ff88]';
    if (value >= 0.60) return 'bg-[#ff9500]';
    return 'bg-[#ff3860]';
  };

  // Helper for sparkline
  const getSparkline = (data: number[], color: string) => {
    if (!data || data.length < 2) return null;
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const points = data.map((v, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 100 - ((v - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="w-full h-8" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  };

  return (
    <div className="min-h-screen relative bg-[#0a0e1a]">
      {/* Subtle Sport Background */}
      <ElegantSportBackground />
      
      {/* Content wrapper */}
      <div className="relative z-10">
        {/* Navigation */}
        <nav className="sticky top-0 z-50 bg-[#0a0e1a]/80 backdrop-blur-xl border-b border-white/10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <button 
                onClick={() => navigate('/')}
                className="flex items-center gap-3 hover:opacity-80 transition-opacity"
              >
                <div className="w-10 h-10 rounded-xl bg-[#00d4ff] flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-black" />
                </div>
                <span className="text-xl font-bold text-white">NEXUS<span className="text-[#00d4ff]">AI</span></span>
              </button>
              
              <div className="hidden md:flex items-center gap-8">
                <a href="#features" className="text-sm text-gray-400 hover:text-white transition-colors">Features</a>
                <a href="#predictions" className="text-sm text-gray-400 hover:text-white transition-colors">Predictions</a>
                <a href="#analytics" className="text-sm text-gray-400 hover:text-white transition-colors">Analytics</a>
                <a href="#stats" className="text-sm text-gray-400 hover:text-white transition-colors">Statistics</a>
                <a href="#pricing" className="text-sm text-gray-400 hover:text-white transition-colors">Pricing</a>
              </div>

              <div className="flex items-center gap-3">
                <Button 
                  variant="outline"
                  className="border-[#00d4ff]/50 text-[#00d4ff] hover:bg-[#00d4ff]/10"
                  onClick={() => window.location.href = '/app'}
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Launch App
                </Button>
                <Button 
                  variant="ghost" 
                  className="text-gray-400 hover:text-white hidden sm:flex"
                  onClick={() => window.location.href = '/sign-in'}
                >
                  Log in
                </Button>
                <Button 
                  className="bg-gradient-to-r from-[#00d4ff] to-[#00ff88] text-black font-semibold hover:opacity-90 hidden sm:flex"
                  onClick={() => window.location.href = '/sign-up'}
                >
                  Get Started
                </Button>
              </div>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="relative pt-20 pb-32 overflow-hidden">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
            <div className="text-center max-w-3xl mx-auto">
              <Badge className="mb-6 bg-white/5 text-gray-300 border-white/10 hover:bg-white/10">
                AI-Powered Sports Analytics
              </Badge>
              
              <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight tracking-tight">
                <span className="text-white">Predict The</span>
                <br />
                <span className="text-[#00d4ff]">Future of Sports</span>
              </h1>
              
              <p className="text-lg text-gray-400 mb-10 leading-relaxed max-w-2xl mx-auto">
                Advanced machine learning models analyzing millions of data points 
                to deliver accurate predictions and actionable insights for sports analytics.
              </p>
              
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button 
                  size="lg"
                  className="bg-[#00d4ff] text-black font-medium text-base px-8 hover:bg-[#00d4ff]/90"
                  onClick={() => window.location.href = '/app'}
                >
                  Launch Analytics Platform
                </Button>
                <Button 
                  size="lg"
                  variant="outline" 
                  className="border-white/20 text-white hover:bg-white/5 text-base px-8"
                >
                  View Documentation
                </Button>
              </div>

              {/* Stats Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-16">
                {stats.map((stat, i) => (
                  <div key={i} className="text-center">
                    <div className="flex items-center justify-center gap-2 text-[#00d4ff] mb-2">
                      <stat.icon className="w-5 h-5" />
                    </div>
                    <div className="text-3xl font-bold text-white font-mono">{stat.value}</div>
                    <div className="text-sm text-gray-500">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Featured Predictions */}
        <section id="predictions" className="py-20 bg-black/20 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="section-header">
              <h2 className="section-title">Featured Predictions</h2>
              <Button variant="ghost" className="text-[#00d4ff] hover:text-[#00ff88]">
                View All <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {featuredMatches.map((match, i) => (
                <div 
                  key={i}
                  className="match-card"
                >
                  {/* Header */}
                  <div className="flex items-center justify-between mb-4">
                    <Badge className="bg-white/5 text-gray-400 border-0">
                      <span className="mr-1">{match.leagueIcon}</span>
                      {match.league}
                    </Badge>
                    <span className="text-sm text-gray-500">{match.time}</span>
                  </div>

                  {/* Teams */}
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-12 h-12 rounded-xl flex items-center justify-center text-lg font-bold"
                        style={{ background: `${match.home.color}20`, color: match.home.color }}
                      >
                        {match.home.name[0]}
                      </div>
                      <div>
                        <div className="font-bold text-white">{match.home.name}</div>
                        <div className="text-sm text-gray-500">Home</div>
                      </div>
                    </div>
                    <div className="text-gray-600 font-bold">VS</div>
                    <div className="flex items-center gap-3 text-right">
                      <div>
                        <div className="font-bold text-white">{match.away.name}</div>
                        <div className="text-sm text-gray-500">Away</div>
                      </div>
                      <div 
                        className="w-12 h-12 rounded-xl flex items-center justify-center text-lg font-bold"
                        style={{ background: `${match.away.color}20`, color: match.away.color }}
                      >
                        {match.away.name[0]}
                      </div>
                    </div>
                  </div>

                  {/* Odds */}
                  <div className="flex justify-center gap-3 mb-4">
                    <div className="odds-box">
                      <span className="odds-value" style={{ color: match.home.color }}>{match.home.odds}</span>
                      <span className="odds-label">1</span>
                    </div>
                    {match.draw && (
                      <div className="odds-box">
                        <span className="odds-value text-gray-400">{match.draw}</span>
                        <span className="odds-label">X</span>
                      </div>
                    )}
                    <div className="odds-box">
                      <span className="odds-value" style={{ color: match.away.color }}>{match.away.odds}</span>
                      <span className="odds-label">2</span>
                    </div>
                  </div>

                  {/* Prediction */}
                  <div className="p-3 bg-[#00ff88]/10 border border-[#00ff88]/30 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-[#00ff88]" />
                        <span className="text-sm text-gray-300">AI Prediction</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-[#00ff88] font-bold">
                          {match.prediction === '1' ? match.home.name : match.prediction === '2' ? match.away.name : 'Draw'}
                        </span>
                        <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0">
                          {match.confidence}%
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Analytics Dashboard Preview */}
        <section id="analytics" className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <Badge className="mb-4 bg-[#b829dd]/10 text-[#b829dd] border-[#b829dd]/30">
                <Brain className="w-3 h-3 mr-1" />
                Advanced Analytics
              </Badge>
              <h2 className="text-4xl font-bold text-white mb-4">Model Performance</h2>
              <p className="text-gray-400 max-w-2xl mx-auto">
                Our machine learning models are trained on millions of historical matches 
                and continuously optimized for maximum accuracy.
              </p>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {kpiData.map((kpi, i) => (
                <div 
                  key={i}
                  className={cn(
                    "bg-[#0f1623]/80 border border-white/10 rounded-xl p-4",
                    "hover:border-[#00d4ff]/30 hover:bg-[#0f1623] transition-all duration-300",
                    "group cursor-pointer"
                  )}
                >
                  <div className="flex items-start justify-between mb-2">
                    <span className="text-xs font-medium uppercase tracking-wider text-gray-500">
                      {kpi.label}
                    </span>
                    {kpi.change !== undefined && (
                      <div className={cn(
                        "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                        kpi.changeType === 'positive' ? 'text-[#00ff88] bg-[#00ff88]/10' : 
                        'text-[#ff3860] bg-[#ff3860]/10'
                      )}>
                        <TrendingUp className="w-3 h-3" />
                        <span>+{kpi.change}%</span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-white/5 text-[#00d4ff] group-hover:text-[#00ff88] transition-colors">
                      {kpi.icon}
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-white font-mono">
                        {kpi.value}
                      </div>
                      {kpi.subValue && (
                        <div className="text-xs text-gray-500">
                          {kpi.subValue}
                        </div>
                      )}
                    </div>
                  </div>

                  {kpi.sparklineData && (
                    <div className="mt-2 opacity-50 group-hover:opacity-100 transition-opacity">
                      {getSparkline(kpi.sparklineData, kpi.changeType === 'positive' ? '#00ff88' : '#00d4ff')}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Model Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-[#0f1623]/80 border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                  <Brain className="w-5 h-5 text-[#00d4ff]" />
                  Model Performance Metrics
                </h3>
                
                <div className="space-y-5">
                  {modelMetrics.map((metric, i) => (
                    <div key={i} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-gray-400">{metric.name}</span>
                          <span className="text-xs text-gray-600" title={metric.description}>ⓘ</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-white tabular-nums">
                            {(metric.value * 100).toFixed(1)}%
                          </span>
                          <span className="text-xs text-gray-500">
                            (bench: {(metric.benchmark * 100).toFixed(0)}%)
                          </span>
                        </div>
                      </div>
                      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className={cn("h-full rounded-full transition-all duration-1000", getMetricColor(metric.value))}
                          style={{ width: animatedStats ? `${metric.value * 100}%` : '0%' }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* AI Insights */}
              <div className="bg-[#0f1623]/80 border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-[#ff9500]" />
                  AI Insights & Alerts
                </h3>
                
                <div className="space-y-4">
                  {insights.map((insight, i) => {
                    const config = {
                      opportunity: { icon: Zap, color: 'border-[#00d4ff]/30 bg-[#00d4ff]/10 text-[#00d4ff]' },
                      positive: { icon: TrendingUp, color: 'border-[#00ff88]/30 bg-[#00ff88]/10 text-[#00ff88]' },
                      warning: { icon: AlertTriangle, color: 'border-[#ff9500]/30 bg-[#ff9500]/10 text-[#ff9500]' },
                    }[insight.type];
                    const Icon = config.icon;
                    
                    return (
                      <div key={i} className={cn("rounded-lg border p-4", config.color)}>
                        <div className="flex items-start gap-3">
                          <div className="p-1.5 rounded-md bg-white/10">
                            <Icon className="w-4 h-4" />
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-medium uppercase tracking-wide opacity-70">
                                {insight.type}
                              </span>
                              {insight.confidence !== undefined && (
                                <span className={cn(
                                  "text-xs px-1.5 py-0.5 rounded-full bg-white/10",
                                  insight.confidence >= 75 ? 'text-[#00ff88]' :
                                  insight.confidence >= 50 ? 'text-[#ff9500]' :
                                  'text-[#ff3860]'
                                )}>
                                  {insight.confidence}% conf
                                </span>
                              )}
                            </div>
                            
                            <h4 className="font-semibold text-sm mb-1">{insight.title}</h4>
                            <p className="text-sm opacity-80 leading-relaxed">{insight.description}</p>
                            
                            {insight.metric && (
                              <div className="mt-2 flex items-center gap-2 text-sm">
                                <span className="opacity-70">{insight.metric.label}:</span>
                                <span className="font-semibold tabular-nums">{insight.metric.value}</span>
                                {insight.metric.change !== undefined && (
                                  <span className={cn(
                                    "text-xs",
                                    insight.metric.change > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                                  )}>
                                    {insight.metric.change > 0 ? '+' : ''}{insight.metric.change}%
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Recent Predictions Table */}
        <section id="stats" className="py-20 bg-black/20 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="section-header mb-8">
              <div>
                <h2 className="text-2xl font-bold text-white">Recent Predictions</h2>
                <p className="text-sm text-gray-500">Last 7 days performance</p>
              </div>
              <Button variant="ghost" className="text-[#00d4ff] hover:text-[#00ff88]">
                View History <ArrowUpRight className="w-4 h-4 ml-1" />
              </Button>
            </div>

            <div className="bg-[#0f1623]/80 border border-white/10 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10 bg-white/5">
                    <th className="py-3 px-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">Match</th>
                    <th className="py-3 px-4 text-center text-xs font-medium uppercase tracking-wider text-gray-500">Result</th>
                    <th className="py-3 px-4 text-center text-xs font-medium uppercase tracking-wider text-gray-500">Odds</th>
                    <th className="py-3 px-4 text-center text-xs font-medium uppercase tracking-wider text-gray-500">Profit/Loss</th>
                    <th className="py-3 px-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {recentPredictions.map((pred, i) => (
                    <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="py-3 px-4">
                        <span className="text-white font-medium">{pred.match}</span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <Badge className={cn(
                          "border-0",
                          pred.result === 'WIN' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#ff3860]/20 text-[#ff3860]'
                        )}>
                          {pred.result === 'WIN' ? <CheckCircle2 className="w-3 h-3 mr-1" /> : null}
                          {pred.result}
                        </Badge>
                      </td>
                      <td className="py-3 px-4 text-center text-gray-400">{pred.odds}</td>
                      <td className={cn(
                        "py-3 px-4 text-center font-medium tabular-nums",
                        pred.profit.startsWith('+') ? 'text-[#00ff88]' : 'text-[#ff3860]'
                      )}>
                        {pred.profit}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-500 text-xs">{pred.date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Sports Categories */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-white mb-4">Sports Coverage</h2>
              <p className="text-gray-400">Comprehensive analysis across all major sports</p>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {sportsCategories.map((sport, i) => {
                const Icon = sport.icon;
                return (
                  <div 
                    key={i}
                    className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-6 text-center hover:border-[#00d4ff]/30 hover:bg-[#0f1623] transition-all group cursor-pointer"
                  >
                    <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-white/5 flex items-center justify-center group-hover:bg-[#00d4ff]/10 transition-colors">
                      <Icon className="w-6 h-6 text-[#00d4ff]" />
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">{sport.name}</h3>
                    <div className="space-y-1">
                      <div className="text-sm text-gray-500">{sport.active} active</div>
                      <div className="text-sm text-[#00ff88]">{sport.accuracy}% accuracy</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="py-20 bg-black/20 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-4">Why Choose NEXUS AI?</h2>
              <p className="text-gray-400 max-w-2xl mx-auto">
                Our advanced algorithms analyze millions of data points to provide you with 
                accurate predictions and valuable betting insights.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { icon: Activity, title: 'Real-Time Data', desc: 'Live odds and statistics updated every second' },
                { icon: Shield, title: '99.9% Uptime', desc: 'Reliable predictions when you need them most' },
                { icon: BarChart3, title: 'Advanced Analytics', desc: 'Machine learning models trained on 10+ years of data' },
                { icon: Zap, title: 'Instant Alerts', desc: 'Get notified when high-value opportunities appear' },
                { icon: Brain, title: 'AI Predictions', desc: 'Neural networks analyzing match patterns' },
                { icon: Cpu, title: 'Edge Computing', desc: 'Sub-second latency for live betting' },
                { icon: Award, title: 'Proven Results', desc: 'Verified track record of profitability' },
                { icon: Clock, title: '24/7 Support', desc: 'Expert team available around the clock' },
              ].map((feature, i) => (
                <div key={i} className="card-glow p-6 text-center group cursor-pointer">
                  <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-[#00d4ff]/20 to-[#00ff88]/20 flex items-center justify-center group-hover:from-[#00d4ff]/30 group-hover:to-[#00ff88]/30 transition-all">
                    <feature.icon className="w-7 h-7 text-[#00d4ff]" />
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">{feature.title}</h3>
                  <p className="text-sm text-gray-400">{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="relative p-8 rounded-2xl bg-gradient-to-br from-[#00d4ff]/10 via-[#b829dd]/10 to-[#00ff88]/10 border border-white/10">
              {/* Glow effect */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-[#00d4ff]/20 to-[#00ff88]/20 blur-xl opacity-50" />
              
              <div className="relative">
                <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                  Ready to Start Winning?
                </h2>
                <p className="text-xl text-gray-400 mb-8">
                  Join thousands of analysts and sports professionals who trust NEXUS AI for data-driven predictions.
                  Start your free trial today.
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                  <Button 
                    size="lg"
                    className="bg-gradient-to-r from-[#00d4ff] to-[#00ff88] text-black font-bold text-lg px-10 hover:opacity-90 shadow-lg shadow-[#00d4ff]/25"
                    onClick={() => window.location.href = '/sign-up'}
                  >
                    Start Free Trial
                  </Button>
                  <Button 
                    size="lg"
                    variant="outline" 
                    className="border-white/20 text-white hover:bg-white/10 text-lg px-10"
                    onClick={() => window.location.href = '/app'}
                  >
                    Try Demo
                  </Button>
                </div>
                <p className="text-sm text-gray-500 mt-4">No credit card required • Cancel anytime</p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-white/10 py-12 bg-black/40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center">
                  <BarChart3 className="w-5 h-5 text-black" />
                </div>
                <span className="text-lg font-bold text-white">NEXUS<span className="text-[#00d4ff]">AI</span></span>
              </div>
              <div className="flex items-center gap-6 text-sm text-gray-500">
                <a href="#" className="hover:text-white transition-colors">Privacy</a>
                <a href="#" className="hover:text-white transition-colors">Terms</a>
                <a href="#" className="hover:text-white transition-colors">Contact</a>
              </div>
              <p className="text-sm text-gray-600">
                © 2026 NEXUS AI. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default LandingPage;
