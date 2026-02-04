/**
 * StatisticsPage - Betting & Prediction Statistics
 *
 * Connected to:
 * - GET /api/stats
 * - GET /api/bets/history
 *
 * Features:
 * - Overall prediction performance metrics
 * - Sport-by-sport ROI breakdown
 * - Win/loss distribution
 * - Profit tracking over time
 */

import { useState, useEffect, useCallback } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  BarChart3,
  Loader2,
  Percent,
  PieChart as PieChartIcon,
  Target,
  TrendingUp,
  Trophy,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api, { type SystemStats, type BetHistoryItem } from '@/lib/api';

export function StatisticsPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [bets, setBets] = useState<BetHistoryItem[]>([]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, betsRes] = await Promise.allSettled([
        api.getStats(),
        api.getBetHistory({ days: 365, limit: 500 }),
      ]);

      if (statsRes.status === 'fulfilled') {
        setStats(statsRes.value);
      }
      if (betsRes.status === 'fulfilled') {
        setBets(betsRes.value.bets || []);
      }
    } catch (e) {
      console.error('Failed to fetch statistics:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Sport performance table data
  const sportPerformance = stats?.roi_by_sport
    ? Object.entries(stats.roi_by_sport).map(([sport, data]) => ({
        sport,
        winRate: data.win_rate,
        roi: data.roi,
        totalBets: data.total_bets,
        profit: data.profit,
      }))
    : [];

  // Win/Loss pie chart
  const winLossData = stats
    ? [
        { name: 'Won', value: stats.successful_bets, fill: '#00ff88' },
        {
          name: 'Lost',
          value: Math.max(0, stats.total_analyses - stats.successful_bets),
          fill: '#ff3860',
        },
      ]
    : [];

  // Cumulative P/L from bet history
  const cumulativePL = bets
    .filter((b) => b.status !== 'pending')
    .sort(
      (a, b) =>
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    )
    .reduce<{ date: string; cumulative: number; match: string }[]>(
      (acc, bet) => {
        const prev = acc.length > 0 ? acc[acc.length - 1].cumulative : 0;
        acc.push({
          date: new Date(bet.created_at).toLocaleDateString(),
          cumulative: Math.round((prev + bet.profit_loss) * 100) / 100,
          match: bet.match,
        });
        return acc;
      },
      []
    );

  // Daily P/L aggregation
  const dailyPL = bets
    .filter((b) => b.status !== 'pending')
    .reduce<Record<string, number>>((acc, bet) => {
      const date = new Date(bet.created_at).toLocaleDateString();
      acc[date] = (acc[date] || 0) + bet.profit_loss;
      return acc;
    }, {});

  const dailyPLChart = Object.entries(dailyPL)
    .map(([date, pl]) => ({ date, pl: Math.round(pl * 100) / 100 }))
    .sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

  // Odds distribution
  const oddsDistribution = bets.reduce<
    { range: string; count: number; won: number }[]
  >((acc, bet) => {
    let range: string;
    if (bet.odds < 1.5) range = '<1.50';
    else if (bet.odds < 2.0) range = '1.50-1.99';
    else if (bet.odds < 2.5) range = '2.00-2.49';
    else if (bet.odds < 3.0) range = '2.50-2.99';
    else range = '3.00+';

    const existing = acc.find((r) => r.range === range);
    if (existing) {
      existing.count++;
      if (bet.status === 'won') existing.won++;
    } else {
      acc.push({ range, count: 1, won: bet.status === 'won' ? 1 : 0 });
    }
    return acc;
  }, []);

  // Sort odds distribution
  const oddsOrder = ['<1.50', '1.50-1.99', '2.00-2.49', '2.50-2.99', '3.00+'];
  oddsDistribution.sort(
    (a, b) => oddsOrder.indexOf(a.range) - oddsOrder.indexOf(b.range)
  );

  // Last 7 days from stats
  const last7Days = stats?.last_7_days || [];

  return (
    <PageLayout
      title="Statistics"
      description="Betting performance metrics and prediction analytics"
      breadcrumbs={[{ label: 'Statistics' }]}
    >
      <div className="space-y-6">
        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading statistics...
          </div>
        )}

        {/* Top KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Analyses</div>
              <div className="text-2xl font-bold text-white">
                {stats?.total_analyses ?? 0}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {stats?.sports_analyzed?.length ?? 0} sport(s)
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Win Rate</div>
              <div className="text-2xl font-bold text-[#00ff88]">
                {stats?.win_rate?.toFixed(1) ?? '0'}%
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {stats?.successful_bets ?? 0} wins
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Profit</div>
              <div
                className={cn(
                  'text-2xl font-bold',
                  (stats?.total_profit ?? 0) >= 0
                    ? 'text-[#00ff88]'
                    : 'text-[#ff3860]'
                )}
              >
                {(stats?.total_profit ?? 0) >= 0 ? '+' : ''}
                {(stats?.total_profit ?? 0).toFixed(2)}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {stats?.roi?.toFixed(1) ?? '0'}% ROI
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Avg Edge</div>
              <div className="text-2xl font-bold text-[#00d4ff]">
                {stats?.avg_edge
                  ? `${(stats.avg_edge * 100).toFixed(1)}%`
                  : 'N/A'}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Quality:{' '}
                {stats?.avg_quality
                  ? `${stats.avg_quality.toFixed(0)}`
                  : 'N/A'}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <Activity className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="sports"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <Trophy className="w-4 h-4 mr-2" />
              By Sport
            </TabsTrigger>
            <TabsTrigger
              value="distribution"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Distribution
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Cumulative P/L */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-[#00ff88]" />
                    Cumulative Profit/Loss
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {cumulativePL.length > 0 ? (
                    <div className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={cumulativePL}>
                          <defs>
                            <linearGradient
                              id="colorCumPL"
                              x1="0"
                              y1="0"
                              x2="0"
                              y2="1"
                            >
                              <stop
                                offset="5%"
                                stopColor="#00ff88"
                                stopOpacity={0.3}
                              />
                              <stop
                                offset="95%"
                                stopColor="#00ff88"
                                stopOpacity={0}
                              />
                            </linearGradient>
                          </defs>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="rgba(255,255,255,0.1)"
                          />
                          <XAxis
                            dataKey="date"
                            stroke="#64748b"
                            fontSize={11}
                            tick={false}
                          />
                          <YAxis stroke="#64748b" fontSize={11} />
                          <RechartsTooltip
                            contentStyle={{
                              backgroundColor: '#0f1623',
                              border: '1px solid rgba(255,255,255,0.1)',
                              borderRadius: '8px',
                            }}
                            formatter={(value: number) => [
                              `${value}`,
                              'Cumulative P/L',
                            ]}
                          />
                          <Area
                            type="monotone"
                            dataKey="cumulative"
                            stroke="#00ff88"
                            fillOpacity={1}
                            fill="url(#colorCumPL)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[280px] flex items-center justify-center text-gray-500 text-sm">
                      No settled bets yet
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Win/Loss Distribution */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <PieChartIcon className="w-5 h-5 text-[#00d4ff]" />
                    Win/Loss Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {winLossData.some((d) => d.value > 0) ? (
                    <div className="h-[280px] flex items-center justify-center">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={winLossData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={100}
                            paddingAngle={4}
                            dataKey="value"
                          >
                            {winLossData.map((entry, index) => (
                              <Cell key={index} fill={entry.fill} />
                            ))}
                          </Pie>
                          <RechartsTooltip
                            contentStyle={{
                              backgroundColor: '#0f1623',
                              border: '1px solid rgba(255,255,255,0.1)',
                              borderRadius: '8px',
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[280px] flex items-center justify-center text-gray-500 text-sm">
                      No data yet
                    </div>
                  )}
                  {winLossData.some((d) => d.value > 0) && (
                    <div className="flex justify-center gap-6 mt-2">
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-3 h-3 rounded-full bg-[#00ff88]" />
                        <span className="text-gray-400">
                          Won ({winLossData[0]?.value ?? 0})
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-3 h-3 rounded-full bg-[#ff3860]" />
                        <span className="text-gray-400">
                          Lost ({winLossData[1]?.value ?? 0})
                        </span>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Last 7 days */}
            {last7Days.length > 0 && (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-[#00d4ff]" />
                    Last 7 Days Performance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={last7Days}>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="rgba(255,255,255,0.1)"
                        />
                        <XAxis
                          dataKey="date"
                          stroke="#64748b"
                          fontSize={11}
                        />
                        <YAxis stroke="#64748b" fontSize={11} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                          }}
                        />
                        <Bar dataKey="profit" radius={[4, 4, 0, 0]}>
                          {last7Days.map((entry, index) => (
                            <Cell
                              key={index}
                              fill={
                                entry.profit >= 0 ? '#00ff88' : '#ff3860'
                              }
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* By Sport Tab */}
          <TabsContent value="sports" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-[#00d4ff]" />
                  Performance by Sport
                </CardTitle>
              </CardHeader>
              <CardContent>
                {sportPerformance.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10 hover:bg-transparent">
                        <TableHead className="text-gray-500">Sport</TableHead>
                        <TableHead className="text-gray-500 text-center">
                          Total Bets
                        </TableHead>
                        <TableHead className="text-gray-500 text-center">
                          Win Rate
                        </TableHead>
                        <TableHead className="text-gray-500 text-center">
                          ROI
                        </TableHead>
                        <TableHead className="text-gray-500 text-center">
                          Profit
                        </TableHead>
                        <TableHead className="text-gray-500 text-center">
                          Grade
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sportPerformance.map((sp) => (
                        <TableRow
                          key={sp.sport}
                          className="border-white/5 hover:bg-white/5"
                        >
                          <TableCell className="font-medium text-white capitalize">
                            {sp.sport}
                          </TableCell>
                          <TableCell className="text-center text-gray-400">
                            {sp.totalBets}
                          </TableCell>
                          <TableCell className="text-center">
                            <span
                              className={cn(
                                'font-medium',
                                sp.winRate >= 55
                                  ? 'text-[#00ff88]'
                                  : sp.winRate >= 45
                                    ? 'text-[#ff9500]'
                                    : 'text-[#ff3860]'
                              )}
                            >
                              {sp.winRate.toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <span
                              className={cn(
                                'font-bold',
                                sp.roi > 0
                                  ? 'text-[#00ff88]'
                                  : 'text-[#ff3860]'
                              )}
                            >
                              {sp.roi > 0 ? '+' : ''}
                              {sp.roi.toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <span
                              className={cn(
                                'font-medium',
                                sp.profit > 0
                                  ? 'text-[#00ff88]'
                                  : 'text-[#ff3860]'
                              )}
                            >
                              {sp.profit > 0 ? '+' : ''}
                              {sp.profit.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <Badge
                              className={cn(
                                'border-0',
                                sp.roi >= 10
                                  ? 'bg-[#00ff88]/20 text-[#00ff88]'
                                  : sp.roi >= 5
                                    ? 'bg-[#00d4ff]/20 text-[#00d4ff]'
                                    : sp.roi >= 0
                                      ? 'bg-[#ff9500]/20 text-[#ff9500]'
                                      : 'bg-[#ff3860]/20 text-[#ff3860]'
                              )}
                            >
                              {sp.roi >= 10
                                ? 'A+'
                                : sp.roi >= 5
                                  ? 'A'
                                  : sp.roi >= 0
                                    ? 'B'
                                    : 'C'}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : !loading ? (
                  <div className="p-12 text-center text-gray-500">
                    <Trophy className="w-10 h-10 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">
                      No sport performance data available.
                    </p>
                    <p className="text-xs mt-1">
                      Run analyses across different sports to see breakdown
                      here.
                    </p>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Distribution Tab */}
          <TabsContent value="distribution" className="mt-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Odds Distribution */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-[#00d4ff]" />
                    Bets by Odds Range
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {oddsDistribution.length > 0 ? (
                    <div className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={oddsDistribution}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="rgba(255,255,255,0.1)"
                          />
                          <XAxis
                            dataKey="range"
                            stroke="#64748b"
                            fontSize={11}
                          />
                          <YAxis stroke="#64748b" fontSize={11} />
                          <RechartsTooltip
                            contentStyle={{
                              backgroundColor: '#0f1623',
                              border: '1px solid rgba(255,255,255,0.1)',
                              borderRadius: '8px',
                            }}
                          />
                          <Bar
                            dataKey="count"
                            fill="rgba(0, 212, 255, 0.5)"
                            name="Total"
                            radius={[4, 4, 0, 0]}
                          />
                          <Bar
                            dataKey="won"
                            fill="rgba(0, 255, 136, 0.5)"
                            name="Won"
                            radius={[4, 4, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[280px] flex items-center justify-center text-gray-500 text-sm">
                      No bet data yet
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Daily P/L */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Percent className="w-5 h-5 text-[#00ff88]" />
                    Daily Profit/Loss
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {dailyPLChart.length > 0 ? (
                    <div className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={dailyPLChart}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="rgba(255,255,255,0.1)"
                          />
                          <XAxis
                            dataKey="date"
                            stroke="#64748b"
                            fontSize={11}
                            tick={false}
                          />
                          <YAxis stroke="#64748b" fontSize={11} />
                          <RechartsTooltip
                            contentStyle={{
                              backgroundColor: '#0f1623',
                              border: '1px solid rgba(255,255,255,0.1)',
                              borderRadius: '8px',
                            }}
                          />
                          <Bar dataKey="pl" name="P/L" radius={[4, 4, 0, 0]}>
                            {dailyPLChart.map((entry, index) => (
                              <Cell
                                key={index}
                                fill={
                                  entry.pl >= 0 ? '#00ff88' : '#ff3860'
                                }
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[280px] flex items-center justify-center text-gray-500 text-sm">
                      No daily data yet
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Summary stats from analyzed sports */}
            {stats?.sports_analyzed && stats.sports_analyzed.length > 0 && (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Target className="w-5 h-5 text-[#00d4ff]" />
                    Analyzed Sports
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {stats.sports_analyzed.map((sport) => (
                      <Badge
                        key={sport}
                        className="bg-[#00d4ff]/10 text-[#00d4ff] border-0 capitalize px-3 py-1"
                      >
                        {sport}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default StatisticsPage;
