/**
 * PredictionsPage - Comprehensive prediction analysis and value betting
 * 
 * Features:
 * - Live and upcoming predictions with detailed analysis
 * - Value bet identification with edge calculation
 * - Confidence scoring with factor breakdown
 * - Odds comparison across bookmakers
 * - Prediction filters and sorting
 * - Similar historical matches
 * - Risk assessment
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Activity,
  ArrowUpRight,
  Bell,
  BookOpen,
  Calendar,
  ChevronDown,
  ChevronUp,
  Clock,
  Filter,
  Flame,
  Layers,
  LineChart,
  Percent,
  Search,
  Shield,
  Star,
  Target,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Types
type Sport = 'football' | 'basketball' | 'tennis';

interface Prediction {
  id: string;
  match: string;
  league: string;
  sport: Sport;
  startTime: string;
  prediction: string;
  confidence: number;
  odds: number;
  modelProb: number;
  bookieProb: number;
  edge: number;
  isValue: boolean;
  riskLevel: 'low' | 'medium' | 'high';
  factors: { name: string; impact: 'positive' | 'negative' | 'neutral'; weight: number }[];
  bookmakers: { name: string; odds: number; best?: boolean }[];
  similarMatches: { match: string; result: string; similarity: number }[];
  status: 'live' | 'upcoming' | 'finished';
  liveScore?: { home: number; away: number; minute: string };
}

const sportConfig: Record<Sport | 'all', { label: string; color: string }> = {
  all: { label: 'All Sports', color: '#00d4ff' },
  football: { label: 'Football', color: '#00ff88' },
  basketball: { label: 'Basketball', color: '#ff9500' },
  tennis: { label: 'Tennis', color: '#00d4ff' },
};

// Mock data
const predictions: Prediction[] = [
  {
    id: '1',
    match: 'Man City vs Arsenal',
    league: 'Premier League',
    sport: 'football',
    startTime: '2026-01-29T18:30:00',
    prediction: 'Man City Win',
    confidence: 82,
    odds: 1.65,
    modelProb: 65,
    bookieProb: 61,
    edge: 4.2,
    isValue: true,
    riskLevel: 'low',
    factors: [
      { name: 'Home Advantage', impact: 'positive', weight: 15 },
      { name: 'Recent Form', impact: 'positive', weight: 18 },
      { name: 'ELO Rating', impact: 'positive', weight: 12 },
      { name: 'Injury Impact', impact: 'negative', weight: -8 },
      { name: 'H2H Record', impact: 'positive', weight: 10 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 1.68, best: true },
      { name: 'Bet365', odds: 1.65 },
      { name: 'Unibet', odds: 1.63 },
      { name: 'William Hill', odds: 1.62 },
    ],
    similarMatches: [
      { match: 'Man City vs Liverpool', result: '2-1', similarity: 88 },
      { match: 'Man City vs Chelsea', result: '3-0', similarity: 82 },
    ],
    status: 'upcoming',
  },
  {
    id: '2',
    match: 'Real Madrid vs Barcelona',
    league: 'La Liga',
    sport: 'football',
    startTime: '2026-01-29T21:00:00',
    prediction: 'Draw',
    confidence: 58,
    odds: 3.40,
    modelProb: 32,
    bookieProb: 29,
    edge: 3.1,
    isValue: true,
    riskLevel: 'high',
    factors: [
      { name: 'Home Advantage', impact: 'neutral', weight: 5 },
      { name: 'Recent Form', impact: 'neutral', weight: 2 },
      { name: 'ELO Rating', impact: 'neutral', weight: 0 },
      { name: 'Pressure Index', impact: 'negative', weight: -5 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 3.50, best: true },
      { name: 'Bet365', odds: 3.40 },
      { name: 'Unibet', odds: 3.35 },
    ],
    similarMatches: [
      { match: 'Real vs Atletico', result: '1-1', similarity: 75 },
    ],
    status: 'upcoming',
  },
  {
    id: '3',
    match: 'Celtics vs Heat',
    league: 'NBA',
    sport: 'basketball',
    startTime: '2026-01-29T01:30:00',
    prediction: 'Celtics Win',
    confidence: 78,
    odds: 1.45,
    modelProb: 72,
    bookieProb: 69,
    edge: 3.0,
    isValue: true,
    riskLevel: 'low',
    factors: [
      { name: 'Rest Days', impact: 'positive', weight: 12 },
      { name: 'Home Court', impact: 'positive', weight: 15 },
      { name: 'Player Form', impact: 'positive', weight: 18 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 1.47, best: true },
      { name: 'DraftKings', odds: 1.45 },
      { name: 'FanDuel', odds: 1.43 },
    ],
    similarMatches: [
      { match: 'Celtics vs Bucks', result: '112-105', similarity: 85 },
    ],
    status: 'upcoming',
  },
  {
    id: '4',
    match: 'Ajax vs Olympiacos',
    league: 'Champions League',
    sport: 'football',
    startTime: '2026-01-28T20:45:00',
    prediction: 'Olympiacos Win',
    confidence: 65,
    odds: 2.88,
    modelProb: 42,
    bookieProb: 35,
    edge: 7.0,
    isValue: true,
    riskLevel: 'medium',
    factors: [
      { name: 'Away Form', impact: 'positive', weight: 15 },
      { name: 'Counter Attack', impact: 'positive', weight: 12 },
      { name: 'Ajax Defense', impact: 'positive', weight: 18 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 2.95, best: true },
      { name: 'Bet365', odds: 2.88 },
    ],
    similarMatches: [],
    status: 'live',
    liveScore: { home: 1, away: 2, minute: "84'" },
  },
  {
    id: '5',
    match: 'Alcaraz vs Djokovic',
    league: 'ATP Finals',
    sport: 'tennis',
    startTime: '2026-01-30T09:00:00',
    prediction: 'Djokovic Win',
    confidence: 71,
    odds: 1.95,
    modelProb: 55,
    bookieProb: 51,
    edge: 3.9,
    isValue: true,
    riskLevel: 'medium',
    factors: [
      { name: 'Surface Expertise', impact: 'positive', weight: 20 },
      { name: 'Experience', impact: 'positive', weight: 15 },
      { name: 'Recent Fatigue', impact: 'negative', weight: -10 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 1.98, best: true },
      { name: 'Bet365', odds: 1.95 },
    ],
    similarMatches: [
      { match: 'Djokovic vs Sinner', result: '2-1', similarity: 80 },
    ],
    status: 'upcoming',
  },
  {
    id: '6',
    match: 'Bayern Munich vs Dortmund',
    league: 'Bundesliga',
    sport: 'football',
    startTime: '2026-01-30T15:30:00',
    prediction: 'Over 2.5',
    confidence: 74,
    odds: 1.72,
    modelProb: 62,
    bookieProb: 58,
    edge: 4.0,
    isValue: true,
    riskLevel: 'low',
    factors: [
      { name: 'H2H Goals Avg', impact: 'positive', weight: 20 },
      { name: 'Both Teams Score Rate', impact: 'positive', weight: 16 },
      { name: 'Defensive Injuries', impact: 'positive', weight: 10 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 1.75, best: true },
      { name: 'Bet365', odds: 1.72 },
    ],
    similarMatches: [
      { match: 'Bayern vs Leipzig', result: '3-2', similarity: 78 },
    ],
    status: 'upcoming',
  },
  {
    id: '7',
    match: 'Lakers vs Warriors',
    league: 'NBA',
    sport: 'basketball',
    startTime: '2026-01-29T03:00:00',
    prediction: 'Warriors Win',
    confidence: 62,
    odds: 2.10,
    modelProb: 52,
    bookieProb: 48,
    edge: 4.2,
    isValue: true,
    riskLevel: 'medium',
    factors: [
      { name: 'Home Court', impact: 'positive', weight: 14 },
      { name: 'Back-to-Back', impact: 'negative', weight: -8 },
      { name: '3PT Shooting', impact: 'positive', weight: 12 },
    ],
    bookmakers: [
      { name: 'DraftKings', odds: 2.12, best: true },
      { name: 'FanDuel', odds: 2.10 },
    ],
    similarMatches: [
      { match: 'Warriors vs Nuggets', result: '118-112', similarity: 72 },
    ],
    status: 'live',
    liveScore: { home: 88, away: 94, minute: 'Q3 4:22' },
  },
  {
    id: '8',
    match: 'Sinner vs Medvedev',
    league: 'Australian Open',
    sport: 'tennis',
    startTime: '2026-01-30T04:00:00',
    prediction: 'Sinner Win',
    confidence: 76,
    odds: 1.55,
    modelProb: 68,
    bookieProb: 65,
    edge: 3.5,
    isValue: false,
    riskLevel: 'low',
    factors: [
      { name: 'Hard Court Record', impact: 'positive', weight: 18 },
      { name: 'Head-to-Head', impact: 'positive', weight: 14 },
      { name: 'Tournament Form', impact: 'positive', weight: 12 },
    ],
    bookmakers: [
      { name: 'Pinnacle', odds: 1.58, best: true },
      { name: 'Bet365', odds: 1.55 },
    ],
    similarMatches: [
      { match: 'Sinner vs Rublev', result: '3-1', similarity: 82 },
    ],
    status: 'upcoming',
  },
];

// Helper components
const ConfidenceBar = ({ value }: { value: number }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
      <div
        className={cn(
          'h-full rounded-full transition-all duration-500',
          value >= 75 ? 'bg-[#00ff88]' : value >= 60 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
        )}
        style={{ width: `${value}%` }}
      />
    </div>
    <span className={cn(
      'text-sm font-medium w-10',
      value >= 75 ? 'text-[#00ff88]' : value >= 60 ? 'text-[#00d4ff]' : 'text-[#ff9500]'
    )}>
      {value}%
    </span>
  </div>
);

const EdgeBadge = ({ edge }: { edge: number }) => (
  <Badge className={cn(
    'border-0',
    edge >= 5 ? 'bg-[#00ff88]/20 text-[#00ff88]' :
    edge >= 3 ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
    edge >= 0 ? 'bg-[#ff9500]/20 text-[#ff9500]' :
    'bg-[#ff3860]/20 text-[#ff3860]'
  )}>
    {edge > 0 ? '+' : ''}{edge.toFixed(1)}%
  </Badge>
);

const FactorBadge = ({ factor }: { factor: Prediction['factors'][0] }) => (
  <div className="flex items-center gap-1.5 text-xs">
    <div className={cn(
      'w-1.5 h-1.5 rounded-full',
      factor.impact === 'positive' ? 'bg-[#00ff88]' :
      factor.impact === 'negative' ? 'bg-[#ff3860]' :
      'bg-gray-500'
    )} />
    <span className="text-gray-400">{factor.name}</span>
    <span className={cn(
      'font-medium',
      factor.impact === 'positive' ? 'text-[#00ff88]' :
      factor.impact === 'negative' ? 'text-[#ff3860]' :
      'text-gray-500'
    )}>
      {factor.weight > 0 ? '+' : ''}{factor.weight}%
    </span>
  </div>
);

const PredictionCard = ({ prediction, expanded, onToggle }: { 
  prediction: Prediction; 
  expanded: boolean;
  onToggle: () => void;
}) => {
  const isLive = prediction.status === 'live';
  
  return (
    <Card className={cn(
      'bg-[#0f1623]/80 border-white/10 transition-all duration-200',
      isLive && 'border-[#ff3860]/30',
      expanded && 'ring-1 ring-[#00d4ff]/30'
    )}>
      <CardContent className="p-0">
        {/* Main Row */}
        <div 
          className="p-4 cursor-pointer hover:bg-white/5 transition-colors"
          onClick={onToggle}
        >
          <div className="flex items-center gap-4">
            {/* Status / Time */}
            <div className="w-16 shrink-0">
              {isLive ? (
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#ff3860] animate-pulse" />
                  <span className="text-sm text-[#ff3860] font-medium">{prediction.liveScore?.minute}</span>
                </div>
              ) : (
                <div className="text-sm text-gray-500">
                  {new Date(prediction.startTime).toLocaleTimeString('en-US', { 
                    hour: '2-digit', 
                    minute: '2-digit',
                    hour12: false 
                  })}
                </div>
              )}
            </div>

            {/* League */}
            <Badge variant="outline" className="shrink-0 text-xs border-white/20 text-gray-400 min-w-[100px] justify-center">
              {prediction.league}
            </Badge>

            {/* Match */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3">
                <span className="text-white font-medium truncate">{prediction.match}</span>
                {isLive && prediction.liveScore && (
                  <span className="text-lg font-bold text-[#ff3860]">
                    {prediction.liveScore.home} - {prediction.liveScore.away}
                  </span>
                )}
              </div>
            </div>

            {/* Prediction */}
            <div className="w-32 shrink-0">
              <div className="text-sm text-[#00d4ff]">{prediction.prediction}</div>
            </div>

            {/* Confidence */}
            <div className="w-32 shrink-0">
              <ConfidenceBar value={prediction.confidence} />
            </div>

            {/* Odds */}
            <div className="w-16 shrink-0 text-right">
              <span className="font-mono text-white">{prediction.odds.toFixed(2)}</span>
            </div>

            {/* Edge */}
            <div className="w-20 shrink-0 flex justify-end">
              <EdgeBadge edge={prediction.edge} />
            </div>

            {/* Risk Level */}
            <div className="w-16 shrink-0 flex justify-end">
              <Badge className={cn(
                'border-0 text-[10px]',
                prediction.riskLevel === 'low' ? 'bg-[#00ff88]/10 text-[#00ff88]' :
                prediction.riskLevel === 'medium' ? 'bg-[#ff9500]/10 text-[#ff9500]' :
                'bg-[#ff3860]/10 text-[#ff3860]'
              )}>
                {prediction.riskLevel === 'low' ? 'Low' : prediction.riskLevel === 'medium' ? 'Med' : 'High'}
              </Badge>
            </div>

            {/* Value Badge */}
            <div className="w-20 shrink-0 flex justify-end">
              {prediction.isValue ? (
                <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0">
                  <Zap className="w-3 h-3 mr-1" />
                  Value
                </Badge>
              ) : (
                <Badge variant="outline" className="border-white/10 text-gray-500">
                  Standard
                </Badge>
              )}
            </div>

            {/* Expand Icon */}
            <div className="w-6 shrink-0 flex justify-end">
              {expanded ? (
                <ChevronUp className="w-4 h-4 text-gray-500" />
              ) : (
                <ChevronDown className="w-4 h-4 text-gray-500" />
              )}
            </div>
          </div>
        </div>

        {/* Expanded Details */}
        {expanded && (
          <div className="px-4 pb-4 border-t border-white/10">
            <div className="pt-4 grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Factors */}
              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Key Factors
                </h4>
                <div className="space-y-2">
                  {prediction.factors.map((factor, idx) => (
                    <FactorBadge key={idx} factor={factor} />
                  ))}
                </div>
                <div className="mt-4 p-3 bg-white/5 rounded-lg">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Model Probability</span>
                    <span className="text-[#00d4ff] font-medium">{prediction.modelProb}%</span>
                  </div>
                  <div className="flex items-center justify-between text-sm mt-2">
                    <span className="text-gray-400">Bookmaker Implied</span>
                    <span className="text-gray-300">{prediction.bookieProb}%</span>
                  </div>
                  <div className="flex items-center justify-between text-sm mt-2 pt-2 border-t border-white/10">
                    <span className="text-gray-400">Edge</span>
                    <span className={cn(
                      'font-medium',
                      prediction.edge > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                    )}>
                      {prediction.edge > 0 ? '+' : ''}{prediction.edge}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Bookmaker Comparison */}
              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                  <Percent className="w-4 h-4" />
                  Odds Comparison
                </h4>
                <div className="space-y-2">
                  {prediction.bookmakers.map((bookie, idx) => (
                    <div 
                      key={idx} 
                      className={cn(
                        'flex items-center justify-between p-2 rounded-lg',
                        bookie.best ? 'bg-[#00ff88]/10 border border-[#00ff88]/20' : 'bg-white/5'
                      )}
                    >
                      <span className="text-sm text-gray-300">{bookie.name}</span>
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          'font-mono font-medium',
                          bookie.best ? 'text-[#00ff88]' : 'text-white'
                        )}>
                          {bookie.odds.toFixed(2)}
                        </span>
                        {bookie.best && (
                          <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-xs">
                            Best
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Similar Matches */}
              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Similar Historical Matches
                </h4>
                {prediction.similarMatches.length > 0 ? (
                  <div className="space-y-2">
                    {prediction.similarMatches.map((match, idx) => (
                      <div key={idx} className="p-3 bg-white/5 rounded-lg">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-white">{match.match}</span>
                          <Badge variant="outline" className="text-xs border-white/20 text-gray-400">
                            {match.similarity}% match
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-xs text-gray-500">Result: {match.result}</span>
                          <span className={cn(
                            'text-xs',
                            match.result.includes(prediction.prediction.split(' ')[0]) 
                              ? 'text-[#00ff88]' 
                              : 'text-gray-500'
                          )}>
                            Prediction would: WIN
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 bg-white/5 rounded-lg text-center">
                    <p className="text-sm text-gray-500">No similar matches found</p>
                  </div>
                )}
                
                <div className="mt-4 flex gap-2">
                  <Button variant="outline" size="sm" className="flex-1 border-white/10 text-gray-400">
                    <BookOpen className="w-4 h-4 mr-2" />
                    Full Analysis
                  </Button>
                  <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                    <Bell className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export function PredictionsPage() {
  const [activeTab, setActiveTab] = useState('all');
  const [sportFilter, setSportFilter] = useState<Sport | 'all'>('all');
  const [expandedId, setExpandedId] = useState<string | null>('1');
  const [searchQuery, setSearchQuery] = useState('');
  const [confidenceFilter, setConfidenceFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [valueOnly, setValueOnly] = useState(false);

  const filteredPredictions = predictions.filter(p => {
    if (sportFilter !== 'all' && p.sport !== sportFilter) return false;
    if (activeTab === 'live') return p.status === 'live';
    if (activeTab === 'upcoming') return p.status === 'upcoming';
    if (activeTab === 'value') return p.isValue;
    return true;
  }).filter(p => {
    if (searchQuery && !p.match.toLowerCase().includes(searchQuery.toLowerCase()) && !p.league.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    if (confidenceFilter === 'high' && p.confidence < 75) return false;
    if (confidenceFilter === 'medium' && (p.confidence < 60 || p.confidence >= 75)) return false;
    if (confidenceFilter === 'low' && p.confidence >= 60) return false;
    if (riskFilter !== 'all' && p.riskLevel !== riskFilter) return false;
    if (valueOnly && !p.isValue) return false;
    return true;
  });

  // Sport counts
  const sportCounts = {
    all: predictions.length,
    football: predictions.filter(p => p.sport === 'football').length,
    basketball: predictions.filter(p => p.sport === 'basketball').length,
    tennis: predictions.filter(p => p.sport === 'tennis').length,
  };

  const stats = {
    total: filteredPredictions.length,
    live: predictions.filter(p => p.status === 'live' && (sportFilter === 'all' || p.sport === sportFilter)).length,
    value: predictions.filter(p => p.isValue && (sportFilter === 'all' || p.sport === sportFilter)).length,
    avgConfidence: filteredPredictions.length > 0 ? Math.round(filteredPredictions.reduce((acc, p) => acc + p.confidence, 0) / filteredPredictions.length) : 0,
    avgEdge: filteredPredictions.filter(p => p.isValue).length > 0 ? (filteredPredictions.filter(p => p.isValue).reduce((acc, p) => acc + p.edge, 0) / filteredPredictions.filter(p => p.isValue).length).toFixed(1) : '0.0',
  };

  return (
    <PageLayout
      title="Predictions"
      description="Match predictions with confidence scores, value analysis, and edge detection"
      breadcrumbs={[{ label: 'Predictions' }]}
    >
      <div className="space-y-6">
        {/* Sport Sub-tabs */}
        <div className="flex items-center gap-2">
          {(Object.entries(sportConfig) as [Sport | 'all', { label: string; color: string }][]).map(([key, cfg]) => (
            <Button
              key={key}
              variant="outline"
              size="sm"
              onClick={() => setSportFilter(key)}
              className={cn(
                'border-white/10 transition-all',
                sportFilter === key
                  ? 'text-black border-transparent'
                  : 'text-gray-400 hover:text-white'
              )}
              style={sportFilter === key ? { backgroundColor: cfg.color } : undefined}
            >
              {cfg.label}
              <Badge
                variant="secondary"
                className={cn(
                  'ml-2 border-0 text-[10px]',
                  sportFilter === key
                    ? 'bg-black/20 text-black'
                    : 'bg-white/10 text-gray-500'
                )}
              >
                {sportCounts[key]}
              </Badge>
            </Button>
          ))}
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Total Predictions</p>
                  <p className="text-2xl font-bold text-white">{stats.total}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
                  <Target className="w-5 h-5 text-[#00d4ff]" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Live Matches</p>
                  <p className="text-2xl font-bold text-[#ff3860]">{stats.live}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#ff3860]/10 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-[#ff3860]" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Value Bets</p>
                  <p className="text-2xl font-bold text-[#00ff88]">{stats.value}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#00ff88]/10 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-[#00ff88]" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Avg Edge</p>
                  <p className="text-2xl font-bold text-[#00d4ff]">+{stats.avgEdge}%</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#00d4ff]/10 flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-[#00d4ff]" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <Card className="bg-[#0f1623]/80 border-white/10">
          <CardContent className="p-4">
            <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-3 flex-1 w-full lg:w-auto">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    placeholder="Search matches, teams, leagues..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 bg-white/5 border-white/10 text-white"
                  />
                </div>
                <Select value={confidenceFilter} onValueChange={setConfidenceFilter}>
                  <SelectTrigger className="w-[160px] bg-white/5 border-white/10 text-white">
                    <Filter className="w-4 h-4 mr-2" />
                    <SelectValue placeholder="Confidence" />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0f1623] border-white/10">
                    <SelectItem value="all" className="text-white">All Confidence</SelectItem>
                    <SelectItem value="high" className="text-white">High (75%+)</SelectItem>
                    <SelectItem value="medium" className="text-white">Medium (60-74%)</SelectItem>
                    <SelectItem value="low" className="text-white">Low (&lt;60%)</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={riskFilter} onValueChange={setRiskFilter}>
                  <SelectTrigger className="w-[140px] bg-white/5 border-white/10 text-white">
                    <Shield className="w-4 h-4 mr-2" />
                    <SelectValue placeholder="Risk" />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0f1623] border-white/10">
                    <SelectItem value="all" className="text-white">All Risk</SelectItem>
                    <SelectItem value="low" className="text-white">Low Risk</SelectItem>
                    <SelectItem value="medium" className="text-white">Medium Risk</SelectItem>
                    <SelectItem value="high" className="text-white">High Risk</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch 
                    id="value-only" 
                    checked={valueOnly}
                    onCheckedChange={setValueOnly}
                  />
                  <Label htmlFor="value-only" className="text-sm text-gray-400 cursor-pointer">
                    Value bets only
                  </Label>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tabs and List */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="all" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Layers className="w-4 h-4 mr-2" />
              All
              <Badge variant="secondary" className="ml-2 bg-white/10 text-gray-400 border-0">
                {stats.total}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="live" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Activity className="w-4 h-4 mr-2" />
              Live
              <Badge variant="secondary" className="ml-2 bg-[#ff3860]/20 text-[#ff3860] border-0">
                {stats.live}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="value" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Zap className="w-4 h-4 mr-2" />
              Value
              <Badge variant="secondary" className="ml-2 bg-[#00ff88]/20 text-[#00ff88] border-0">
                {stats.value}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="upcoming" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Calendar className="w-4 h-4 mr-2" />
              Upcoming
            </TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab} className="mt-6 space-y-3">
            {filteredPredictions.length > 0 ? (
              filteredPredictions.map((prediction) => (
                <PredictionCard
                  key={prediction.id}
                  prediction={prediction}
                  expanded={expandedId === prediction.id}
                  onToggle={() => setExpandedId(expandedId === prediction.id ? null : prediction.id)}
                />
              ))
            ) : (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center">
                  <Search className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No predictions match your filters</p>
                  <Button 
                    variant="link" 
                    className="text-[#00d4ff] mt-2"
                    onClick={() => {
                      setSearchQuery('');
                      setConfidenceFilter('all');
                      setRiskFilter('all');
                      setValueOnly(false);
                      setSportFilter('all');
                    }}
                  >
                    Clear filters
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default PredictionsPage;
