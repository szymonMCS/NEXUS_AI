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

import { useState, useEffect, useCallback } from 'react';
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
  Loader2,
  Server,
  Target,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api from '@/lib/api';

// Types for dashboard data
interface KPIItem {
  label: string;
  value: string;
  change: string;
  positive: boolean;
  icon: typeof Target;
  sparkline: number[];
}

interface SystemStatusItem {
  name: string;
  status: string;
  accuracy: string | null;
  lastUpdate: string;
}

interface UpcomingPrediction {
  id: number;
  league: string;
  match: string;
  time: string;
  prediction: string;
  confidence: number;
  odds: number;
  value: boolean;
  edge: number;
}

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
  const [activeTab, setActiveTab] = useState('upcoming');
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  // Real data state
  const [kpiData, setKpiData] = useState<KPIItem[]>([
    { label: "Today's Predictions", value: '-', change: '', positive: true, icon: Target, sparkline: [] },
    { label: 'Total Bets', value: '-', change: '', positive: true, icon: Activity, sparkline: [] },
    { label: 'Win Rate (30d)', value: '-', change: '', positive: true, icon: TrendingUp, sparkline: [] },
    { label: 'Total ROI', value: '-', change: '', positive: true, icon: Zap, sparkline: [] },
  ]);
  const [upcomingPredictions, setUpcomingPredictions] = useState<UpcomingPrediction[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatusItem[]>([]);
  const [roiChartData, setRoiChartData] = useState<Array<{ day: string; roi: number; predictions: number }>>([]);

  // Fetch real data from API
  const fetchDashboardData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, statusRes, predictionsRes] = await Promise.allSettled([
        api.getStats(),
        api.getStatus(),
        api.getPredictions(),
      ]);

      // Process stats
      if (statsRes.status === 'fulfilled') {
        const stats = statsRes.value;
        setKpiData([
          { label: "Total Analyses", value: String(stats.total_analyses || 0), change: '', positive: true, icon: Target, sparkline: [] },
          { label: 'Winning Bets', value: String(stats.successful_bets || 0), change: '', positive: true, icon: Activity, sparkline: [] },
          { label: 'Win Rate (30d)', value: `${stats.win_rate || 0}%`, change: '', positive: (stats.win_rate || 0) > 50, icon: TrendingUp, sparkline: [] },
          { label: 'Total ROI', value: `${(stats.roi ?? stats.total_profit ?? 0) > 0 ? '+' : ''}${stats.roi ?? stats.total_profit ?? 0}%`, change: '', positive: (stats.roi ?? stats.total_profit ?? 0) > 0, icon: Zap, sparkline: [] },
        ]);
      }

      // Process system status
      if (statusRes.status === 'fulfilled') {
        const status = statusRes.value;
        const items: SystemStatusItem[] = [
          { name: 'API Server', status: status.status === 'running' ? 'active' : 'warning', accuracy: null, lastUpdate: 'now' },
          { name: `Mode: ${status.mode}`, status: 'active', accuracy: null, lastUpdate: `v${(status as any).version || '?'}` },
        ];
        setSystemStatus(items);
      }

      // Process predictions
      if (predictionsRes.status === 'fulfilled') {
        const preds = predictionsRes.value;
        if (preds.value_bets && preds.value_bets.length > 0) {
          setUpcomingPredictions(preds.value_bets.map((vb, i) => ({
            id: i + 1,
            league: vb.league || '',
            match: vb.match || '',
            time: preds.date || '',
            prediction: vb.selection || '',
            confidence: Math.round((vb.confidence || 0) * 100),
            odds: vb.odds || 0,
            value: (vb.edge || 0) > 0.02,
            edge: Math.round((vb.edge || 0) * 1000) / 10,
          })));
        }
      }
    } catch (e) {
      console.error('Dashboard data fetch error:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
    // Refresh every 60 seconds
    const interval = setInterval(fetchDashboardData, 60_000);
    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  return (
    <PageLayout
      title="Dashboard"
      description="Overview of matches, predictions, and performance"
    >
      <div className="space-y-6">
        {/* Loading indicator */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading dashboard data...
          </div>
        )}

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
                {roiChartData.length > 0 ? (
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
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    <div className="text-center">
                      <BarChart3 className="w-10 h-10 mx-auto mb-2 opacity-30" />
                      <p className="text-sm">No ROI data yet</p>
                      <p className="text-xs mt-1">Chart will populate after bets are settled</p>
                    </div>
                  </div>
                )}
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
                    <div className="p-6 text-center text-gray-500">
                      <Activity className="w-8 h-8 mx-auto mb-2 opacity-40" />
                      <p className="text-sm">No live matches tracked at the moment.</p>
                      <p className="text-xs mt-1">Run an analysis to start tracking matches.</p>
                    </div>
                  </TabsContent>

                  <TabsContent value="upcoming" className="mt-0">
                    {upcomingPredictions.length === 0 ? (
                      <div className="p-6 text-center text-gray-500">
                        <Target className="w-8 h-8 mx-auto mb-2 opacity-40" />
                        <p className="text-sm">No predictions yet.</p>
                        <Button
                          size="sm"
                          className="mt-3 bg-[#00d4ff] text-black hover:bg-[#00d4ff]/90"
                          onClick={() => navigate('/app/predictions')}
                        >
                          Run Analysis
                        </Button>
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow className="border-white/10 hover:bg-transparent">
                            <TableHead className="text-gray-500">Match</TableHead>
                            <TableHead className="text-gray-500">Date</TableHead>
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
                    )}
                  </TabsContent>

                  <TabsContent value="results" className="mt-0">
                    <div className="p-6 text-center text-gray-500">
                      <Clock className="w-8 h-8 mx-auto mb-2 opacity-40" />
                      <p className="text-sm">No settled results yet.</p>
                      <p className="text-xs mt-1">Results will appear after matches finish.</p>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Sidebar Content */}
          <div className="space-y-6">
            {/* System Status */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2 text-base">
                  <Clock className="w-4 h-4 text-[#00d4ff]" />
                  System Info
                </CardTitle>
              </CardHeader>
              <CardContent>
                {systemStatus.length > 0 ? (
                  <div className="space-y-2">
                    {systemStatus.map((item, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2.5 bg-white/5 rounded-lg">
                        <div className="flex items-center gap-2">
                          <div className={cn('w-2 h-2 rounded-full', item.status === 'active' ? 'bg-[#00ff88]' : 'bg-[#ff9500]')} />
                          <span className="text-sm text-white">{item.name}</span>
                        </div>
                        <span className="text-xs text-gray-500">{item.lastUpdate}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 text-center py-4">Connecting...</p>
                )}
              </CardContent>
            </Card>

            {/* Info */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2 text-base">
                  <TrendingUp className="w-4 h-4 text-[#00d4ff]" />
                  Performance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 text-center py-4">
                  Performance stats will appear after running analyses and settling bets.
                </p>
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
