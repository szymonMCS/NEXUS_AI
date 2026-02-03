/**
 * HandicapsPage - Comprehensive handicap analysis and selection
 * 
 * Features:
 * - Asian handicap matrix with all lines
 * - European handicap display
 * - Line movement tracking
 * - ROI per handicap type
 * - Historical performance
 * - Odds comparison across bookmakers
 * - Value identification
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
} from 'recharts';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  ArrowRightLeft,
  Calculator,
  ChevronDown,
  ChevronUp,
  DollarSign,
  Filter,
  History,
  Info,
  Minus,
  Percent,
  Scale,
  TrendingUp,
  Trophy,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Types
interface AsianHandicapLine {
  line: number;
  homeOdds: number;
  awayOdds: number;
  homeProb: number;
  awayProb: number;
  homeEdge: number;
  awayEdge: number;
  lineMovement: 'up' | 'down' | 'stable';
  volume: number;
}

interface EuropeanHandicapLine {
  line: string;
  homeOdds: number;
  drawOdds: number;
  awayOdds: number;
  homeProb: number;
  drawProb: number;
  awayProb: number;
}

interface HandicapMatch {
  id: string;
  league: string;
  homeTeam: string;
  awayTeam: string;
  matchTime: string;
  asianLines: AsianHandicapLine[];
  europeanLines: EuropeanHandicapLine[];
  totalLines: TotalLine[];
}

interface TotalLine {
  line: number;
  overOdds: number;
  underOdds: number;
  overProb: number;
  underProb: number;
  overEdge: number;
  underEdge: number;
}

// Mock data for handicap matches
const handicapMatches: HandicapMatch[] = [
  {
    id: '1',
    league: 'Premier League',
    homeTeam: 'Manchester City',
    awayTeam: 'Arsenal',
    matchTime: 'Today 18:30',
    asianLines: [
      { line: -1.5, homeOdds: 2.85, awayOdds: 1.42, homeProb: 32, awayProb: 68, homeEdge: -0.08, awayEdge: -0.03, lineMovement: 'down', volume: 1250 },
      { line: -1, homeOdds: 2.25, awayOdds: 1.68, homeProb: 42, awayProb: 58, homeEdge: -0.05, awayEdge: -0.02, lineMovement: 'stable', volume: 2100 },
      { line: -0.5, homeOdds: 1.88, awayOdds: 1.98, homeProb: 51, awayProb: 49, homeEdge: -0.04, awayEdge: -0.03, lineMovement: 'up', volume: 3400 },
      { line: 0, homeOdds: 1.62, awayOdds: 2.35, homeProb: 58, awayProb: 42, homeEdge: -0.06, awayEdge: -0.01, lineMovement: 'stable', volume: 2800 },
      { line: 0.5, homeOdds: 1.42, awayOdds: 2.85, homeProb: 68, awayProb: 32, homeEdge: -0.03, awayEdge: -0.08, lineMovement: 'up', volume: 1900 },
    ],
    europeanLines: [
      { line: '0:0', homeOdds: 2.40, drawOdds: 3.20, awayOdds: 2.80, homeProb: 38, drawProb: 28, awayProb: 34 },
      { line: '1:0', homeOdds: 1.55, drawOdds: 3.80, awayOdds: 5.50, homeProb: 62, drawProb: 18, awayProb: 20 },
      { line: '0:1', homeOdds: 5.00, drawOdds: 3.60, awayOdds: 1.62, homeProb: 19, drawProb: 26, awayProb: 55 },
    ],
    totalLines: [
      { line: 2, overOdds: 1.75, underOdds: 2.05, overProb: 56, underProb: 44, overEdge: -0.02, underEdge: -0.10, },
      { line: 2.5, overOdds: 1.95, underOdds: 1.90, overProb: 51, underProb: 49, overEdge: -0.05, underEdge: -0.07, },
      { line: 3, overOdds: 2.35, underOdds: 1.60, overProb: 42, underProb: 58, overEdge: -0.01, underEdge: -0.07, },
    ],
  },
  {
    id: '2',
    league: 'La Liga',
    homeTeam: 'Real Madrid',
    awayTeam: 'Barcelona',
    matchTime: 'Today 21:00',
    asianLines: [
      { line: -0.5, homeOdds: 1.95, awayOdds: 1.92, homeProb: 50, awayProb: 50, homeEdge: -0.02, awayEdge: -0.04, lineMovement: 'stable', volume: 4200 },
      { line: 0, homeOdds: 1.48, awayOdds: 2.70, homeProb: 65, awayProb: 35, homeEdge: -0.04, awayEdge: -0.05, lineMovement: 'down', volume: 3100 },
      { line: 0.5, homeOdds: 1.28, awayOdds: 3.70, homeProb: 75, awayProb: 25, homeEdge: -0.04, awayEdge: -0.07, lineMovement: 'stable', volume: 2200 },
    ],
    europeanLines: [
      { line: '0:0', homeOdds: 2.10, drawOdds: 3.40, awayOdds: 3.25, homeProb: 45, drawProb: 28, awayProb: 27 },
      { line: '1:0', homeOdds: 1.42, drawOdds: 4.20, awayOdds: 7.00, homeProb: 68, drawProb: 16, awayProb: 16 },
    ],
    totalLines: [
      { line: 2.5, overOdds: 1.72, underOdds: 2.10, overProb: 58, underProb: 42, overEdge: 0, underEdge: -0.12, },
      { line: 3, overOdds: 2.10, underOdds: 1.72, overProb: 48, underProb: 52, overEdge: 0.01, underEdge: -0.10, },
    ],
  },
];

// Line movement history data
const lineMovementData = [
  { time: '24h ago', open: -0.75, current: -0.5, peak: -0.25, low: -1 },
  { time: '18h ago', open: -0.75, current: -0.5, peak: -0.25, low: -1 },
  { time: '12h ago', open: -0.75, current: -0.5, peak: -0.5, low: -1 },
  { time: '6h ago', open: -0.75, current: -0.5, peak: -0.5, low: -0.75 },
  { time: '3h ago', open: -0.75, current: -0.5, peak: -0.5, low: -0.75 },
  { time: '1h ago', open: -0.75, current: -0.5, peak: -0.5, low: -0.75 },
  { time: 'Now', open: -0.75, current: -0.5, peak: -0.5, low: -0.75 },
];

// Historical ROI data
const roiData = [
  { handicap: 'AH -1.5', roi: 12.5, wins: 45, losses: 32, total: 77 },
  { handicap: 'AH -1', roi: 8.2, wins: 52, losses: 38, total: 90 },
  { handicap: 'AH -0.5', roi: -2.1, wins: 48, losses: 52, total: 100 },
  { handicap: 'AH 0', roi: 5.8, wins: 58, losses: 42, total: 100 },
  { handicap: 'AH +0.5', roi: 3.4, wins: 62, losses: 38, total: 100 },
  { handicap: 'EH 0:0', roi: 6.2, wins: 55, losses: 45, total: 100 },
  { handicap: 'EH 1:0', roi: -5.3, wins: 42, losses: 58, total: 100 },
  { handicap: 'O/U 2.5', roi: 4.1, wins: 60, losses: 40, total: 100 },
];

// Bookmaker odds comparison
const bookmakerOdds = [
  { name: 'Bet365', ahHome: 1.88, ahAway: 1.98, ou25Over: 1.95, ou25Under: 1.90, margin: 3.2 },
  { name: 'Pinnacle', ahHome: 1.92, ahAway: 1.96, ou25Over: 1.98, ou25Under: 1.92, margin: 2.1 },
  { name: 'Betfair', ahHome: 1.90, ahAway: 1.95, ou25Over: 1.96, ou25Under: 1.88, margin: 2.8 },
  { name: '1xBet', ahHome: 1.94, ahAway: 1.94, ou25Over: 2.00, ou25Under: 1.86, margin: 2.5 },
  { name: 'Unibet', ahHome: 1.85, ahAway: 2.00, ou25Over: 1.92, ou25Under: 1.92, margin: 3.5 },
  { name: 'William Hill', ahHome: 1.87, ahAway: 1.97, ou25Over: 1.91, ou25Under: 1.91, margin: 3.4 },
];

// Value opportunities
const valueOpportunities = [
  { market: 'AH -1.5 Home', bookmaker: 'Pinnacle', odds: 2.92, modelProb: 38, impliedProb: 34.2, edge: 3.8, confidence: 72 },
  { market: 'O/U 2.5 Over', bookmaker: '1xBet', odds: 2.00, modelProb: 54, impliedProb: 50, edge: 4.0, confidence: 68 },
  { market: 'AH 0 Home', bookmaker: '1xBet', odds: 1.62, modelProb: 65, impliedProb: 61.7, edge: 3.3, confidence: 75 },
];

// Helper components
const ProbabilityBar = ({ value, color = 'cyan' }: { value: number; color?: 'cyan' | 'green' | 'amber' | 'red' }) => {
  const colorClasses = {
    cyan: 'bg-[#00d4ff]',
    green: 'bg-[#00ff88]',
    amber: 'bg-[#ff9500]',
    red: 'bg-[#ff3860]',
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', colorClasses[color])}
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{value}%</span>
    </div>
  );
};

const EdgeIndicator = ({ edge }: { edge: number }) => {
  if (edge >= 0.05) {
    return (
      <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-xs">
        <TrendingUp className="w-3 h-3 mr-1" />
        +{(edge * 100).toFixed(1)}%
      </Badge>
    );
  }
  if (edge >= 0) {
    return <span className="text-xs text-gray-500">+{(edge * 100).toFixed(1)}%</span>;
  }
  return <span className="text-xs text-[#ff3860]">{(edge * 100).toFixed(1)}%</span>;
};

const LineMovementIndicator = ({ movement }: { movement: 'up' | 'down' | 'stable' }) => {
  if (movement === 'up') {
    return <ChevronUp className="w-4 h-4 text-[#00ff88]" />;
  }
  if (movement === 'down') {
    return <ChevronDown className="w-4 h-4 text-[#ff3860]" />;
  }
  return <Minus className="w-4 h-4 text-gray-500" />;
};

export function HandicapsPage() {
  const [selectedMatch, setSelectedMatch] = useState<string>(handicapMatches[0].id);
  const [activeTab, setActiveTab] = useState('asian');
  const [selectedLine, setSelectedLine] = useState<string | null>(null);
  const [calcStake, setCalcStake] = useState('100');
  const [calcOdds, setCalcOdds] = useState('1.88');

  const match = handicapMatches.find(m => m.id === selectedMatch) || handicapMatches[0];

  // Calculator logic
  const stake = parseFloat(calcStake) || 0;
  const odds = parseFloat(calcOdds) || 0;
  const potentialReturn = stake * odds;
  const potentialProfit = potentialReturn - stake;
  const impliedProbability = odds > 0 ? (1 / odds) * 100 : 0;

  return (
    <PageLayout
      title="Handicap Analysis"
      description="Asian and European handicaps with line movement tracking and value identification"
      breadcrumbs={[{ label: 'Handicaps' }]}
    >
      <div className="space-y-6">
        {/* Match Selector */}
        <Card className="bg-[#0f1623]/80 border-white/10">
          <CardContent className="p-4">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <div className="flex items-center gap-4">
                <Select value={selectedMatch} onValueChange={setSelectedMatch}>
                  <SelectTrigger className="w-[320px] bg-white/5 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0f1623] border-white/10">
                    {handicapMatches.map((m) => (
                      <SelectItem key={m.id} value={m.id} className="text-white">
                        <div className="flex flex-col">
                          <span>{m.homeTeam} vs {m.awayTeam}</span>
                          <span className="text-xs text-gray-500">{m.league} • {m.matchTime}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                
                <div className="hidden md:flex items-center gap-2 text-sm text-gray-400">
                  <span>{match.league}</span>
                  <span>•</span>
                  <span>{match.matchTime}</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                  <History className="w-4 h-4 mr-2" />
                  History
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Match Header */}
        <div className="flex items-center justify-between p-6 bg-gradient-to-r from-[#00d4ff]/5 to-[#00ff88]/5 border border-white/10 rounded-xl">
          <div className="flex items-center gap-8">
            <div className="text-center">
              <div className="text-lg font-bold text-white">{match.homeTeam}</div>
              <div className="text-xs text-gray-500">Home</div>
            </div>
            <div className="text-2xl font-bold text-gray-600">VS</div>
            <div className="text-center">
              <div className="text-lg font-bold text-white">{match.awayTeam}</div>
              <div className="text-xs text-gray-500">Away</div>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="text-xs text-gray-500 mb-1">Best Value</div>
              <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0">
                AH 0 @ 1.62
              </Badge>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500 mb-1">Line Movement</div>
              <div className="flex items-center gap-1 text-[#00d4ff]">
                <ArrowRightLeft className="w-4 h-4" />
                <span className="text-sm font-medium">Stable</span>
              </div>
            </div>
          </div>
        </div>

        {/* Value Opportunities & Calculator Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Value Opportunities */}
          <div className="lg:col-span-2 space-y-3">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="w-4 h-4 text-[#00ff88]" />
              <span className="text-sm font-medium text-white">Value Opportunities</span>
              <Badge variant="outline" className="text-[10px] border-[#00ff88]/30 text-[#00ff88]">
                {valueOpportunities.length} found
              </Badge>
            </div>
            {valueOpportunities.map((opp, idx) => (
              <div key={idx} className="flex items-center gap-4 p-3 bg-[#0f1623]/80 border border-white/10 rounded-lg hover:border-[#00ff88]/30 transition-colors">
                <div className="p-2 bg-[#00ff88]/10 rounded-lg">
                  <TrendingUp className="w-4 h-4 text-[#00ff88]" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-white text-sm font-medium">{opp.market}</span>
                    <Badge variant="outline" className="text-[10px] border-white/20 text-gray-400">{opp.bookmaker}</Badge>
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
                    <span>Model: {opp.modelProb}%</span>
                    <span>Implied: {opp.impliedProb}%</span>
                    <span>Confidence: {opp.confidence}%</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[#00ff88] font-bold text-sm">+{opp.edge.toFixed(1)}% edge</div>
                  <div className="text-white font-mono text-sm">{opp.odds.toFixed(2)}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Handicap Calculator */}
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-white text-base flex items-center gap-2">
                <Calculator className="w-4 h-4 text-[#00d4ff]" />
                Handicap Calculator
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
                  <span className="text-white font-bold font-mono">${potentialReturn.toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Potential Profit</span>
                  <span className={cn('font-bold font-mono', potentialProfit > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]')}>
                    {potentialProfit > 0 ? '+' : ''}${potentialProfit.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Implied Probability</span>
                  <span className="text-[#00d4ff] font-bold">{impliedProbability.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-[#00d4ff] rounded-full" style={{ width: `${Math.min(impliedProbability, 100)}%` }} />
                </div>
              </div>
              <div className="pt-3 border-t border-white/10">
                <div className="text-xs text-gray-500 mb-2">Quick Select Odds</div>
                <div className="flex flex-wrap gap-1">
                  {['1.50', '1.75', '1.88', '2.00', '2.25', '2.50', '3.00'].map((o) => (
                    <Button
                      key={o}
                      variant="outline"
                      size="sm"
                      onClick={() => setCalcOdds(o)}
                      className={cn(
                        'text-xs h-7 px-2 border-white/10',
                        calcOdds === o ? 'bg-[#00d4ff]/10 text-[#00d4ff] border-[#00d4ff]/30' : 'text-gray-400'
                      )}
                    >
                      {o}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="asian" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Trophy className="w-4 h-4 mr-2" />
              Asian Handicap
            </TabsTrigger>
            <TabsTrigger value="european" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Calculator className="w-4 h-4 mr-2" />
              European Handicap
            </TabsTrigger>
            <TabsTrigger value="totals" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Percent className="w-4 h-4 mr-2" />
              Totals
            </TabsTrigger>
            <TabsTrigger value="movement" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <ArrowRightLeft className="w-4 h-4 mr-2" />
              Line Movement
            </TabsTrigger>
            <TabsTrigger value="roi" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <TrendingUp className="w-4 h-4 mr-2" />
              Historical ROI
            </TabsTrigger>
            <TabsTrigger value="bookmakers" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Scale className="w-4 h-4 mr-2" />
              Odds Comparison
            </TabsTrigger>
          </TabsList>

          {/* Asian Handicap Tab */}
          <TabsContent value="asian" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-[#00d4ff]" />
                  Asian Handicap Lines
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Line</TableHead>
                      <TableHead className="text-gray-500">Home</TableHead>
                      <TableHead className="text-gray-500">Probability</TableHead>
                      <TableHead className="text-gray-500">Edge</TableHead>
                      <TableHead className="text-gray-500">Away</TableHead>
                      <TableHead className="text-gray-500">Probability</TableHead>
                      <TableHead className="text-gray-500">Edge</TableHead>
                      <TableHead className="text-gray-500">Movement</TableHead>
                      <TableHead className="text-gray-500">Volume</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {match.asianLines.map((line, idx) => (
                      <TableRow
                        key={idx}
                        className={cn(
                          'border-white/5 cursor-pointer transition-colors',
                          selectedLine === `${line.line}` ? 'bg-[#00d4ff]/10' : 'hover:bg-white/5'
                        )}
                        onClick={() => setSelectedLine(`${line.line}`)}
                      >
                        <TableCell className="font-medium text-white">
                          {line.line > 0 ? `+${line.line}` : line.line}
                        </TableCell>
                        <TableCell className="font-mono text-[#00d4ff]">{line.homeOdds.toFixed(2)}</TableCell>
                        <TableCell>
                          <ProbabilityBar value={line.homeProb} color="cyan" />
                        </TableCell>
                        <TableCell><EdgeIndicator edge={line.homeEdge} /></TableCell>
                        <TableCell className="font-mono text-[#00ff88]">{line.awayOdds.toFixed(2)}</TableCell>
                        <TableCell>
                          <ProbabilityBar value={line.awayProb} color="green" />
                        </TableCell>
                        <TableCell><EdgeIndicator edge={line.awayEdge} /></TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <LineMovementIndicator movement={line.lineMovement} />
                            <span className="text-xs text-gray-500">
                              {line.lineMovement === 'up' ? 'Rising' : line.lineMovement === 'down' ? 'Falling' : 'Stable'}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="text-gray-400 text-sm">
                          {line.volume.toLocaleString()}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Line Explanation */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Info className="w-4 h-4 text-[#00d4ff]" />
                    <span className="text-sm font-medium text-white">AH -0.5 (Must Win)</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    Home team must win the match. If draw or loss, bet is lost. No refund.
                  </p>
                </CardContent>
              </Card>
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Info className="w-4 h-4 text-[#00d4ff]" />
                    <span className="text-sm font-medium text-white">AH 0 (Draw No Bet)</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    If match ends in draw, stake is refunded. Home must win for profit.
                  </p>
                </CardContent>
              </Card>
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Info className="w-4 h-4 text-[#00d4ff]" />
                    <span className="text-sm font-medium text-white">AH +0.5 (Win or Draw)</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    Home team wins if they win or draw. Only loses if away team wins.
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* European Handicap Tab */}
          <TabsContent value="european" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Calculator className="w-5 h-5 text-[#00d4ff]" />
                  European Handicap Lines
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {match.europeanLines.map((line, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <Badge variant="outline" className="border-white/20 text-white">
                            EH {line.line}
                          </Badge>
                          <span className="text-sm text-gray-400">
                            {line.line === '0:0' ? 'Level start' : 
                             line.line.startsWith('1:') ? 'Away starts with +1' :
                             line.line.endsWith(':1') ? 'Home starts with +1' : 'Custom line'}
                          </span>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-2">Home Win</div>
                          <div className="text-lg font-bold text-[#00d4ff] font-mono">{line.homeOdds.toFixed(2)}</div>
                          <div className="mt-2">
                            <ProbabilityBar value={line.homeProb} color="cyan" />
                          </div>
                        </div>
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-2">Draw</div>
                          <div className="text-lg font-bold text-[#ff9500] font-mono">{line.drawOdds.toFixed(2)}</div>
                          <div className="mt-2">
                            <ProbabilityBar value={line.drawProb} color="amber" />
                          </div>
                        </div>
                        <div className="p-3 bg-white/5 rounded-lg text-center">
                          <div className="text-xs text-gray-500 mb-2">Away Win</div>
                          <div className="text-lg font-bold text-[#00ff88] font-mono">{line.awayOdds.toFixed(2)}</div>
                          <div className="mt-2">
                            <ProbabilityBar value={line.awayProb} color="green" />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Totals Tab */}
          <TabsContent value="totals" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Percent className="w-5 h-5 text-[#00d4ff]" />
                  Over/Under Lines
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Line</TableHead>
                      <TableHead className="text-gray-500">Over Odds</TableHead>
                      <TableHead className="text-gray-500">Over Prob</TableHead>
                      <TableHead className="text-gray-500">Over Edge</TableHead>
                      <TableHead className="text-gray-500">Under Odds</TableHead>
                      <TableHead className="text-gray-500">Under Prob</TableHead>
                      <TableHead className="text-gray-500">Under Edge</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {match.totalLines.map((line, idx) => (
                      <TableRow key={idx} className="border-white/5 hover:bg-white/5">
                        <TableCell className="font-medium text-white">{line.line}</TableCell>
                        <TableCell className="font-mono text-[#00d4ff]">{line.overOdds.toFixed(2)}</TableCell>
                        <TableCell>
                          <ProbabilityBar value={line.overProb} color="cyan" />
                        </TableCell>
                        <TableCell><EdgeIndicator edge={line.overEdge} /></TableCell>
                        <TableCell className="font-mono text-[#00ff88]">{line.underOdds.toFixed(2)}</TableCell>
                        <TableCell>
                          <ProbabilityBar value={line.underProb} color="green" />
                        </TableCell>
                        <TableCell><EdgeIndicator edge={line.underEdge} /></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Line Movement Tab */}
          <TabsContent value="movement" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <ArrowRightLeft className="w-5 h-5 text-[#00d4ff]" />
                  24-Hour Line Movement
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={lineMovementData}>
                      <defs>
                        <linearGradient id="colorCurrent" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="time" stroke="#64748b" fontSize={12} />
                      <YAxis stroke="#64748b" fontSize={12} domain={[-1.5, 0]} />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#0f1623',
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: '8px',
                        }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Area
                        type="monotone"
                        dataKey="current"
                        stroke="#00d4ff"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorCurrent)"
                      />
                      <Line type="monotone" dataKey="open" stroke="#64748b" strokeDasharray="5 5" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="mt-4 grid grid-cols-3 gap-4">
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Opening Line</div>
                    <div className="text-lg font-bold text-gray-400">-0.75</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Current Line</div>
                    <div className="text-lg font-bold text-[#00d4ff]">-0.50</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Line Movement</div>
                    <div className="text-lg font-bold text-[#00ff88]">+0.25</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ROI Tab */}
          <TabsContent value="roi" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-[#00d4ff]" />
                  Historical ROI by Handicap Type
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Handicap</TableHead>
                      <TableHead className="text-gray-500">ROI</TableHead>
                      <TableHead className="text-gray-500">Win Rate</TableHead>
                      <TableHead className="text-gray-500">Wins</TableHead>
                      <TableHead className="text-gray-500">Losses</TableHead>
                      <TableHead className="text-gray-500">Total Bets</TableHead>
                      <TableHead className="text-gray-500">Grade</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {roiData.map((row, idx) => (
                      <TableRow key={idx} className="border-white/5 hover:bg-white/5">
                        <TableCell className="font-medium text-white">{row.handicap}</TableCell>
                        <TableCell>
                          <span className={cn(
                            'font-bold',
                            row.roi > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                          )}>
                            {row.roi > 0 ? '+' : ''}{row.roi.toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  'h-full rounded-full',
                                  row.wins / row.total > 0.55 ? 'bg-[#00ff88]' : 'bg-[#00d4ff]'
                                )}
                                style={{ width: `${(row.wins / row.total) * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-400">{((row.wins / row.total) * 100).toFixed(0)}%</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-[#00ff88]">{row.wins}</TableCell>
                        <TableCell className="text-[#ff3860]">{row.losses}</TableCell>
                        <TableCell className="text-gray-400">{row.total}</TableCell>
                        <TableCell>
                          <Badge className={cn(
                            'border-0',
                            row.roi >= 10 ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                            row.roi >= 5 ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
                            row.roi >= 0 ? 'bg-[#ff9500]/20 text-[#ff9500]' :
                            'bg-[#ff3860]/20 text-[#ff3860]'
                          )}>
                            {row.roi >= 10 ? 'A+' : row.roi >= 5 ? 'A' : row.roi >= 0 ? 'B' : 'C'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Bookmaker Odds Comparison Tab */}
          <TabsContent value="bookmakers" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Scale className="w-5 h-5 text-[#00d4ff]" />
                  Bookmaker Odds Comparison — AH -0.5
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Bookmaker</TableHead>
                      <TableHead className="text-gray-500 text-center">AH Home</TableHead>
                      <TableHead className="text-gray-500 text-center">AH Away</TableHead>
                      <TableHead className="text-gray-500 text-center">O 2.5</TableHead>
                      <TableHead className="text-gray-500 text-center">U 2.5</TableHead>
                      <TableHead className="text-gray-500 text-center">Margin</TableHead>
                      <TableHead className="text-gray-500 text-center">Best</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {bookmakerOdds.map((bk, idx) => {
                      const bestAhHome = Math.max(...bookmakerOdds.map(b => b.ahHome));
                      const bestAhAway = Math.max(...bookmakerOdds.map(b => b.ahAway));
                      const bestOver = Math.max(...bookmakerOdds.map(b => b.ou25Over));
                      const bestUnder = Math.max(...bookmakerOdds.map(b => b.ou25Under));
                      const lowestMargin = Math.min(...bookmakerOdds.map(b => b.margin));
                      const isBestHome = bk.ahHome === bestAhHome;
                      const isBestAway = bk.ahAway === bestAhAway;
                      const isBestOver = bk.ou25Over === bestOver;
                      const isBestUnder = bk.ou25Under === bestUnder;
                      const isBestMargin = bk.margin === lowestMargin;

                      return (
                        <TableRow key={idx} className="border-white/5 hover:bg-white/5">
                          <TableCell className="font-medium text-white">{bk.name}</TableCell>
                          <TableCell className="text-center">
                            <span className={cn('font-mono', isBestHome ? 'text-[#00ff88] font-bold' : 'text-gray-300')}>
                              {bk.ahHome.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <span className={cn('font-mono', isBestAway ? 'text-[#00ff88] font-bold' : 'text-gray-300')}>
                              {bk.ahAway.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <span className={cn('font-mono', isBestOver ? 'text-[#00ff88] font-bold' : 'text-gray-300')}>
                              {bk.ou25Over.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <span className={cn('font-mono', isBestUnder ? 'text-[#00ff88] font-bold' : 'text-gray-300')}>
                              {bk.ou25Under.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <span className={cn(
                              'text-sm',
                              isBestMargin ? 'text-[#00ff88] font-bold' :
                              bk.margin <= 2.5 ? 'text-[#00d4ff]' :
                              'text-gray-400'
                            )}>
                              {bk.margin.toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            {(isBestHome || isBestAway || isBestOver || isBestUnder) && (
                              <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-[10px]">
                                Best {[
                                  isBestHome && 'H',
                                  isBestAway && 'A',
                                  isBestOver && 'O',
                                  isBestUnder && 'U',
                                ].filter(Boolean).join('/')}
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>

                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Best AH Home</div>
                    <div className="text-lg font-bold text-[#00ff88] font-mono">
                      {Math.max(...bookmakerOdds.map(b => b.ahHome)).toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">{bookmakerOdds.reduce((best, b) => b.ahHome > best.ahHome ? b : best).name}</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Best AH Away</div>
                    <div className="text-lg font-bold text-[#00ff88] font-mono">
                      {Math.max(...bookmakerOdds.map(b => b.ahAway)).toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">{bookmakerOdds.reduce((best, b) => b.ahAway > best.ahAway ? b : best).name}</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Best Over 2.5</div>
                    <div className="text-lg font-bold text-[#00ff88] font-mono">
                      {Math.max(...bookmakerOdds.map(b => b.ou25Over)).toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">{bookmakerOdds.reduce((best, b) => b.ou25Over > best.ou25Over ? b : best).name}</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg text-center">
                    <div className="text-xs text-gray-500 mb-1">Lowest Margin</div>
                    <div className="text-lg font-bold text-[#00d4ff] font-mono">
                      {Math.min(...bookmakerOdds.map(b => b.margin)).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500">{bookmakerOdds.reduce((best, b) => b.margin < best.margin ? b : best).name}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default HandicapsPage;
