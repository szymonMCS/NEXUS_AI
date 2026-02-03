/**
 * DashboardPage - Main analytical dashboard
 *
 * Features:
 * - KPI cards with sparkline trends
 * - Live matches section with tabs
 * - ROI performance chart (7d)
 * - System health / model status
 * - Activity timeline
 * - Top predictions
 * - Top leagues by accuracy
 * - Quick navigation links
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  Activity,
  ArrowRight,
  BarChart3,
  Calendar,
  CheckCircle2,
  ChevronRight,
  Clock,
  Cpu,
  Flame,
  Server,
  Target,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data
const kpiData = [
  { label: "Today's Predictions", value: '47', change: '+12%', positive: true, icon: Target, sparkline: [32, 35, 38, 42, 40, 44, 47] },
  { label: 'Active Live Matches', value: '12', change: '+3', positive: true, icon: Activity, sparkline: [8, 10, 7, 9, 11, 10, 12] },
  { label: 'Win Rate (30d)', value: '66.2%', change: '+5.2%', positive: true, icon: TrendingUp, sparkline: [58, 60, 61, 63, 64, 65, 66.2] },
  { label: 'Total ROI', value: '+14.8%', change: '+2.1%', positive: true, icon: Zap, sparkline: [8, 9.5, 10, 11.2, 12.5, 13.8, 14.8] },
];

const liveMatches = [
  { id: 1, league: 'Champions League', home: 'Ajax', away: 'Olympiacos', homeScore: 1, awayScore: 2, time: "84'", prediction: { result: '2', prob: 98 } },
  { id: 2, league: 'NBA', home: 'Lakers', away: 'Warriors', homeScore: 98, awayScore: 102, time: 'Q4 2:45', prediction: { result: '2', prob: 65 } },
];

const upcomingPredictions = [
  { id: 1, league: 'Premier League', match: 'Man City vs Arsenal', time: '18:30', prediction: 'Home Win', confidence: 82, odds: 1.65, value: true, edge: 4.2 },
  { id: 2, league: 'La Liga', match: 'Real Madrid vs Barcelona', time: '21:00', prediction: 'Draw', confidence: 76, odds: 3.40, value: false, edge: 1.5 },
  { id: 3, league: 'NBA', match: 'Celtics vs Heat', time: '01:30', prediction: 'Home Win', confidence: 78, odds: 1.45, value: true, edge: 3.0 },
  { id: 4, league: 'ATP', match: 'Alcaraz vs Djokovic', time: '09:00', prediction: 'Away Win', confidence: 71, odds: 1.95, value: true, edge: 3.9 },
];

const recentResults = [
  { match: 'Man City vs Arsenal', result: 'WIN', odds: 1.65, profit: '+6.5%', time: '2h ago' },
  { match: 'Real Madrid vs Barcelona', result: 'LOSS', odds: 2.10, profit: '-10.0%', time: '4h ago' },
  { match: 'Celtics vs Heat', result: 'WIN', odds: 1.45, profit: '+4.5%', time: '6h ago' },
  { match: 'Djokovic vs Sinner', result: 'WIN', odds: 1.75, profit: '+7.5%', time: '8h ago' },
  { match: 'Bayern vs Dortmund', result: 'WIN', odds: 1.50, profit: '+5.0%', time: '10h ago' },
];

const topLeagues = [
  { name: 'Premier League', matches: 12, accuracy: 68.5, roi: 15.2 },
  { name: 'La Liga', matches: 8, accuracy: 62.1, roi: 8.7 },
  { name: 'NBA', matches: 6, accuracy: 71.2, roi: 18.5 },
  { name: 'Champions League', matches: 4, accuracy: 75.0, roi: 22.1 },
  { name: 'ATP', matches: 5, accuracy: 70.0, roi: 14.3 },
];

// ROI performance data (7 days)
const roiChartData = [
  { day: 'Mon', roi: 3.2, predictions: 12, wins: 8 },
  { day: 'Tue', roi: -1.5, predictions: 15, wins: 7 },
  { day: 'Wed', roi: 5.8, predictions: 10, wins: 7 },
  { day: 'Thu', roi: 2.1, predictions: 18, wins: 11 },
  { day: 'Fri', roi: 8.4, predictions: 14, wins: 10 },
  { day: 'Sat', roi: 12.2, predictions: 22, wins: 16 },
  { day: 'Sun', roi: 14.8, predictions: 20, wins: 14 },
];

// System health
const systemStatus = [
  { name: 'Ensemble Model', status: 'active', accuracy: '88.2%', lastUpdate: '2m ago' },
  { name: 'MLP + PCA', status: 'active', accuracy: '86.7%', lastUpdate: '2m ago' },
  { name: 'Data Pipeline', status: 'active', accuracy: null, lastUpdate: '30s ago' },
  { name: 'Odds Feed', status: 'active', accuracy: null, lastUpdate: '15s ago' },
];

// Activity timeline
const activityTimeline = [
  { time: '16:42', event: 'Value bet identified', detail: 'Man City vs Arsenal - Home Win @ 1.65', type: 'value' as const },
  { time: '16:30', event: 'Match started', detail: 'Liverpool vs Chelsea - Premier League', type: 'match' as const },
  { time: '16:15', event: 'Model prediction', detail: 'Atletico vs Sevilla - Home Win (73%)', type: 'prediction' as const },
  { time: '15:55', event: 'Prediction settled', detail: 'Tottenham vs Aston Villa - LOSS', type: 'loss' as const },
  { time: '15:30', event: 'Prediction settled', detail: 'Bayern vs Dortmund - WIN (+5.0%)', type: 'win' as const },
  { time: '14:50', event: 'Line movement', detail: 'Man City AH -0.75 to -0.50', type: 'line' as const },
];

// Helper components
const MiniSparkline = ({ data, positive }: { data: number[]; positive: boolean }) => {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 80;
  const h = 24;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg width={w} height={h} className="opacity-60">
      <polyline
        points={points}
        fill="none"
        stroke={positive ? '#00ff88' : '#ff3860'}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

const KPICard = ({ item }: { item: (typeof kpiData)[0] }) => {
  const Icon = item.icon;
  return (
    <Card className="bg-[#0f1623]/80 border-white/10">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm text-gray-500">{item.label}</p>
            <p className="text-2xl font-bold text-white mt-1">{item.value}</p>
            <div className="flex items-center gap-3 mt-1">
              <p
                className={cn(
                  'text-sm',
                  item.positive ? 'text-[#00ff88]' : 'text-[#ff3860]'
                )}
              >
                {item.positive ? '↑' : '↓'} {item.change}
              </p>
              <MiniSparkline data={item.sparkline} positive={item.positive} />
            </div>
          </div>
          <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
            <Icon className="w-5 h-5 text-[#00d4ff]" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const ConfidenceBar = ({ value }: { value: number }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
      <div
        className={cn(
          'h-full rounded-full',
          value >= 75 ? 'bg-[#00ff88]' : value >= 60 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
        )}
        style={{ width: `${value}%` }}
      />
    </div>
    <span className="text-xs text-gray-400 w-8">{value}%</span>
  </div>
);

export function DashboardPage() {
  const [activeTab, setActiveTab] = useState('live');
  const navigate = useNavigate();

  return (
    <PageLayout
      title="Dashboard"
      description="Overview of matches, predictions, and performance"
    >
      <div className="space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {kpiData.map((item, idx) => (
            <KPICard key={idx} item={item} />
          ))}
        </div>

        {/* Row 2: ROI Chart + System Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ROI Performance Chart */}
          <Card className="lg:col-span-2 bg-[#0f1623]/80 border-white/10">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-[#00d4ff]" />
                  Cumulative ROI (7d)
                </CardTitle>
                <div className="flex items-center gap-3 text-sm">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-[#00ff88]" />
                    <span className="text-gray-500">ROI</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-[#00d4ff]/50" />
                    <span className="text-gray-500">Predictions</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={roiChartData}>
                    <defs>
                      <linearGradient id="roiGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00ff88" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.06)"
                    />
                    <XAxis dataKey="day" stroke="#64748b" fontSize={12} />
                    <YAxis
                      yAxisId="left"
                      stroke="#64748b"
                      fontSize={12}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="#64748b"
                      fontSize={12}
                    />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#0f1623',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: 12,
                      }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="roi"
                      stroke="#00ff88"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#roiGrad)"
                      name="ROI %"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="predictions"
                      stroke="rgba(0, 212, 255, 0.4)"
                      strokeWidth={1}
                      dot={false}
                      name="Predictions"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* System Status */}
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-white flex items-center gap-2 text-base">
                <Server className="w-4 h-4 text-[#00d4ff]" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {systemStatus.map((item) => (
                <div
                  key={item.name}
                  className="flex items-center justify-between p-2.5 bg-white/5 rounded-lg"
                >
                  <div className="flex items-center gap-2.5">
                    <div
                      className={cn(
                        'w-2 h-2 rounded-full',
                        item.status === 'active'
                          ? 'bg-[#00ff88]'
                          : item.status === 'warning'
                            ? 'bg-[#ff9500]'
                            : 'bg-[#ff3860]'
                      )}
                    />
                    <div>
                      <div className="text-sm text-white">{item.name}</div>
                      <div className="text-[10px] text-gray-600">{item.lastUpdate}</div>
                    </div>
                  </div>
                  {item.accuracy && (
                    <span className="text-xs text-[#00ff88] font-mono">
                      {item.accuracy}
                    </span>
                  )}
                </div>
              ))}
              <div className="pt-2 border-t border-white/10 flex items-center justify-between">
                <span className="text-xs text-gray-500">All systems operational</span>
                <div className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88]" />
                  <span className="text-xs text-[#00ff88]">Healthy</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Row 3: Matches + Sidebar */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Matches */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-[#00d4ff]" />
                    Matches
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-[#00d4ff] h-8"
                    onClick={() => navigate('/app/matches')}
                  >
                    View All <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="bg-white/5 border border-white/10 mb-4">
                    <TabsTrigger
                      value="live"
                      className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
                    >
                      <Activity className="w-4 h-4 mr-2" />
                      Live
                    </TabsTrigger>
                    <TabsTrigger
                      value="upcoming"
                      className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
                    >
                      <Calendar className="w-4 h-4 mr-2" />
                      Upcoming
                    </TabsTrigger>
                    <TabsTrigger
                      value="results"
                      className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
                    >
                      <Clock className="w-4 h-4 mr-2" />
                      Results
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="live" className="mt-0">
                    <div className="space-y-3">
                      {liveMatches.map((match) => (
                        <div
                          key={match.id}
                          className="p-4 bg-white/5 rounded-lg border border-white/10"
                        >
                          <div className="flex items-center justify-between mb-3">
                            <Badge
                              variant="outline"
                              className="text-xs border-white/20 text-gray-400"
                            >
                              {match.league}
                            </Badge>
                            <div className="flex items-center gap-1">
                              <span className="w-2 h-2 rounded-full bg-[#ff3860] animate-pulse" />
                              <span className="text-sm text-[#ff3860]">{match.time}</span>
                            </div>
                          </div>
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-white font-medium">{match.home}</span>
                                <span className="text-2xl font-bold text-white">
                                  {match.homeScore}
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span className="text-white font-medium">{match.away}</span>
                                <span className="text-2xl font-bold text-white">
                                  {match.awayScore}
                                </span>
                              </div>
                            </div>
                            <div className="ml-6 pl-6 border-l border-white/10">
                              <div className="text-xs text-gray-500 mb-1">Prediction</div>
                              <Badge className="bg-[#00d4ff]/20 text-[#00d4ff] border-0">
                                {match.prediction.prob}%{' '}
                                {match.prediction.result === '1'
                                  ? 'Home'
                                  : match.prediction.result === '2'
                                    ? 'Away'
                                    : 'Draw'}
                              </Badge>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="upcoming" className="mt-0">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-white/10 hover:bg-transparent">
                          <TableHead className="text-gray-500">Match</TableHead>
                          <TableHead className="text-gray-500">Time</TableHead>
                          <TableHead className="text-gray-500">Prediction</TableHead>
                          <TableHead className="text-gray-500">Conf.</TableHead>
                          <TableHead className="text-gray-500">Odds</TableHead>
                          <TableHead className="text-gray-500">Edge</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {upcomingPredictions.map((pred) => (
                          <TableRow
                            key={pred.id}
                            className="border-white/5 hover:bg-white/5"
                          >
                            <TableCell>
                              <div className="text-white font-medium">{pred.match}</div>
                              <div className="text-xs text-gray-500">{pred.league}</div>
                            </TableCell>
                            <TableCell className="text-gray-400">{pred.time}</TableCell>
                            <TableCell>
                              <span className="text-[#00d4ff]">{pred.prediction}</span>
                              {pred.value && (
                                <Badge className="ml-2 bg-[#00ff88]/20 text-[#00ff88] border-0 text-xs">
                                  Value
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell>
                              <ConfidenceBar value={pred.confidence} />
                            </TableCell>
                            <TableCell className="font-mono text-white">
                              {pred.odds.toFixed(2)}
                            </TableCell>
                            <TableCell>
                              <span
                                className={cn(
                                  'text-xs font-medium',
                                  pred.edge >= 3
                                    ? 'text-[#00ff88]'
                                    : pred.edge >= 1
                                      ? 'text-[#00d4ff]'
                                      : 'text-gray-500'
                                )}
                              >
                                +{pred.edge}%
                              </span>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TabsContent>

                  <TabsContent value="results" className="mt-0">
                    <div className="space-y-2">
                      {recentResults.map((result, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between p-3 bg-white/5 rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            {result.result === 'WIN' ? (
                              <CheckCircle2 className="w-4 h-4 text-[#00ff88]" />
                            ) : (
                              <XCircle className="w-4 h-4 text-[#ff3860]" />
                            )}
                            <div>
                              <div className="text-white font-medium text-sm">
                                {result.match}
                              </div>
                              <div className="text-xs text-gray-500">{result.time}</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <Badge
                              className={cn(
                                'border-0',
                                result.result === 'WIN'
                                  ? 'bg-[#00ff88]/20 text-[#00ff88]'
                                  : 'bg-[#ff3860]/20 text-[#ff3860]'
                              )}
                            >
                              {result.result}
                            </Badge>
                            <span
                              className={cn(
                                'text-sm font-medium w-16 text-right font-mono',
                                result.profit.startsWith('+')
                                  ? 'text-[#00ff88]'
                                  : 'text-[#ff3860]'
                              )}
                            >
                              {result.profit}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Sidebar Content */}
          <div className="space-y-6">
            {/* Activity Timeline */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2 text-base">
                  <Clock className="w-4 h-4 text-[#00d4ff]" />
                  Activity
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-0">
                  {activityTimeline.map((item, idx) => (
                    <div key={idx} className="flex gap-3 py-2">
                      {/* Timeline dot and line */}
                      <div className="flex flex-col items-center">
                        <div
                          className={cn(
                            'w-2 h-2 rounded-full mt-1.5',
                            item.type === 'value'
                              ? 'bg-[#00ff88]'
                              : item.type === 'win'
                                ? 'bg-[#00ff88]'
                                : item.type === 'loss'
                                  ? 'bg-[#ff3860]'
                                  : item.type === 'match'
                                    ? 'bg-[#ff3860]'
                                    : item.type === 'line'
                                      ? 'bg-[#ff9500]'
                                      : 'bg-[#00d4ff]'
                          )}
                        />
                        {idx < activityTimeline.length - 1 && (
                          <div className="w-px flex-1 bg-white/10 mt-1" />
                        )}
                      </div>
                      {/* Content */}
                      <div className="pb-3">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-gray-600 font-mono">
                            {item.time}
                          </span>
                          <span className="text-xs text-gray-400">{item.event}</span>
                        </div>
                        <p className="text-xs text-gray-500 mt-0.5">{item.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Top Leagues */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2 text-base">
                  <TrendingUp className="w-4 h-4 text-[#00d4ff]" />
                  Top Leagues
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {topLeagues.map((league) => (
                  <div
                    key={league.name}
                    className="flex items-center justify-between p-2.5 hover:bg-white/5 rounded-lg transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-white">{league.name}</div>
                      <div className="text-xs text-gray-600">{league.matches} matches</div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-xs text-[#00ff88]">{league.accuracy}%</div>
                        <div className="text-[10px] text-gray-600">accuracy</div>
                      </div>
                      <div className="text-right">
                        <div
                          className={cn(
                            'text-xs font-mono',
                            league.roi > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                          )}
                        >
                          +{league.roi}%
                        </div>
                        <div className="text-[10px] text-gray-600">ROI</div>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Quick Links */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader className="pb-3">
                <CardTitle className="text-white text-base">Quick Links</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1.5">
                {[
                  { label: 'All Matches', href: '/app/matches' },
                  { label: 'View All Predictions', href: '/app/predictions' },
                  { label: 'Handicap Analysis', href: '/app/handicaps' },
                  { label: 'Model Performance', href: '/app/models' },
                  { label: 'Full History', href: '/app/history' },
                ].map((link) => (
                  <Button
                    key={link.label}
                    variant="ghost"
                    className="w-full justify-between text-gray-400 hover:text-white hover:bg-white/5 h-8"
                    onClick={() => navigate(link.href)}
                  >
                    <span className="text-sm">{link.label}</span>
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </PageLayout>
  );
}

export default DashboardPage;
