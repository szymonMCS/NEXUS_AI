/**
 * ReportsPage - Comprehensive analytical reports and summaries
 *
 * Connected to real API:
 * - /api/stats for KPI data and sport performance
 * - /api/bets/history for recent predictions
 */

import { useState, useEffect, useCallback } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
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
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ComposedChart,
  Bar,
} from 'recharts';
import {
  AlertTriangle,
  Brain,
  Calendar,
  CheckCircle2,
  Clock,
  Download,
  Lightbulb,
  Loader2,
  PieChart as PieChartIcon,
  Target,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api, { type BetHistoryItem, type SystemStats } from '@/lib/api';

// Helper components
const KPICard = ({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  description,
}: {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: React.ComponentType<{ className?: string }>;
  description?: string;
}) => (
  <Card className="bg-[#0f1623]/80 border-white/10">
    <CardContent className="p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {change && (
            <div className="flex items-center gap-1 mt-2">
              {changeType === 'positive' && <TrendingUp className="w-4 h-4 text-[#00ff88]" />}
              {changeType === 'negative' && <TrendingUp className="w-4 h-4 text-[#ff3860] rotate-180" />}
              <span className={cn(
                'text-sm font-medium',
                changeType === 'positive' ? 'text-[#00ff88]' :
                changeType === 'negative' ? 'text-[#ff3860]' : 'text-gray-400'
              )}>
                {change}
              </span>
            </div>
          )}
          {description && <p className="text-xs text-gray-500 mt-2">{description}</p>}
        </div>
        <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
          <Icon className="w-5 h-5 text-[#00d4ff]" />
        </div>
      </div>
    </CardContent>
  </Card>
);

interface InsightItem {
  type: 'positive' | 'warning' | 'info';
  title: string;
  description: string;
}

const InsightCard = ({ insight }: { insight: InsightItem }) => {
  const colors = {
    positive: { bg: 'bg-[#00ff88]/10', border: 'border-[#00ff88]/20', icon: CheckCircle2, iconColor: 'text-[#00ff88]' },
    warning: { bg: 'bg-[#ff9500]/10', border: 'border-[#ff9500]/20', icon: AlertTriangle, iconColor: 'text-[#ff9500]' },
    info: { bg: 'bg-[#00d4ff]/10', border: 'border-[#00d4ff]/20', icon: Lightbulb, iconColor: 'text-[#00d4ff]' },
  };

  const style = colors[insight.type];
  const Icon = style.icon;

  return (
    <Card className={cn('border', style.bg, style.border)}>
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <Icon className={cn('w-5 h-5 mt-0.5', style.iconColor)} />
          <div className="flex-1">
            <h4 className="font-medium text-white">{insight.title}</h4>
            <p className="text-sm text-gray-400 mt-1">{insight.description}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Generate insights from real stats
function generateInsights(stats: SystemStats): InsightItem[] {
  const insights: InsightItem[] = [];

  if (stats.win_rate >= 60) {
    insights.push({
      type: 'positive',
      title: 'Strong Win Rate',
      description: `Overall win rate of ${stats.win_rate}% across all sports. Model performing above expectations.`,
    });
  } else if (stats.win_rate > 0) {
    insights.push({
      type: 'warning',
      title: 'Win Rate Below Target',
      description: `Current win rate is ${stats.win_rate}%. Consider adjusting minimum quality thresholds.`,
    });
  }

  if (stats.total_profit > 0) {
    insights.push({
      type: 'positive',
      title: 'Profitable Performance',
      description: `Total profit of ${stats.total_profit.toFixed(2)} units with ${stats.roi}% ROI.`,
    });
  } else if (stats.total_profit < 0) {
    insights.push({
      type: 'warning',
      title: 'Negative P/L',
      description: `Currently at ${stats.total_profit.toFixed(2)} units. Review stake sizing and selection criteria.`,
    });
  }

  if (stats.roi_by_sport) {
    const bestSport = Object.entries(stats.roi_by_sport)
      .filter(([, v]) => v.total_bets > 0)
      .sort(([, a], [, b]) => b.roi - a.roi)[0];
    if (bestSport) {
      insights.push({
        type: 'info',
        title: `Best Sport: ${bestSport[0].charAt(0).toUpperCase() + bestSport[0].slice(1)}`,
        description: `${bestSport[1].roi.toFixed(1)}% ROI with ${bestSport[1].win_rate.toFixed(0)}% win rate from ${bestSport[1].total_bets} bets.`,
      });
    }
  }

  if (stats.total_analyses > 50) {
    insights.push({
      type: 'info',
      title: 'Sufficient Sample Size',
      description: `${stats.total_analyses} total predictions provide reliable performance metrics.`,
    });
  }

  return insights.length > 0 ? insights : [{
    type: 'info',
    title: 'Getting Started',
    description: 'Run analyses to generate performance insights. More data leads to better insights.',
  }];
}

export function ReportsPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [recentBets, setRecentBets] = useState<BetHistoryItem[]>([]);
  const [sportPerformance, setSportPerformance] = useState<Array<{
    sport: string;
    predictions: number;
    winRate: number;
    roi: number;
  }>>([]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, historyRes] = await Promise.allSettled([
        api.getStats(),
        api.getBetHistory({ limit: 10, days: 30 }),
      ]);

      if (statsRes.status === 'fulfilled') {
        const s = statsRes.value;
        setStats(s);

        // Build sport performance from roi_by_sport
        if (s.roi_by_sport) {
          const perf = Object.entries(s.roi_by_sport).map(([sport, data]) => ({
            sport: sport.charAt(0).toUpperCase() + sport.slice(1),
            predictions: data.total_bets || 0,
            winRate: data.win_rate || 0,
            roi: data.roi || 0,
          }));
          setSportPerformance(perf);
        }
      }

      if (historyRes.status === 'fulfilled') {
        setRecentBets(historyRes.value.bets);
      }
    } catch (e) {
      console.error('Failed to fetch report data:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const insights = stats ? generateInsights(stats) : [];

  // Build daily performance from bet history
  const dailyPerformance = recentBets
    .filter(b => b.status !== 'pending' && b.created_at)
    .reduce<Record<string, { date: string; predictions: number; wins: number }>>((acc, bet) => {
      const day = new Date(bet.created_at).toLocaleDateString('en-US', { weekday: 'short' });
      if (!acc[day]) acc[day] = { date: day, predictions: 0, wins: 0 };
      acc[day].predictions++;
      if (bet.status === 'won') acc[day].wins++;
      return acc;
    }, {});

  const dailyData = Object.values(dailyPerformance);

  return (
    <PageLayout
      title="Reports & Analytics"
      description="Comprehensive performance analysis and predictive insights"
      breadcrumbs={[{ label: 'Reports' }]}
      actions={
        <Button variant="outline" className="border-white/10 text-gray-400" disabled>
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      }
    >
      <div className="space-y-6">
        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading reports...
          </div>
        )}

        {/* KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total Predictions"
            value={String(stats?.total_analyses || 0)}
            changeType={stats && stats.total_analyses > 0 ? 'positive' : 'neutral'}
            icon={Target}
            description="Last 30 days"
          />
          <KPICard
            title="Win Rate"
            value={`${stats?.win_rate || 0}%`}
            changeType={stats && stats.win_rate >= 55 ? 'positive' : stats && stats.win_rate > 0 ? 'negative' : 'neutral'}
            icon={Zap}
            description={`${stats?.successful_bets || 0} wins / ${(stats?.total_analyses || 0) - (stats?.successful_bets || 0)} losses`}
          />
          <KPICard
            title="Total ROI"
            value={`${stats?.roi ? (stats.roi > 0 ? '+' : '') + stats.roi.toFixed(1) : '0'}%`}
            changeType={stats && stats.roi > 0 ? 'positive' : stats && stats.roi < 0 ? 'negative' : 'neutral'}
            icon={TrendingUp}
            description="Cumulative return"
          />
          <KPICard
            title="Avg. Edge"
            value={`${stats?.avg_edge ? stats.avg_edge.toFixed(1) : '0'}%`}
            changeType="neutral"
            icon={Brain}
            description="Per prediction"
          />
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="overview" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <PieChartIcon className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="performance" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <TrendingUp className="w-4 h-4 mr-2" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="history" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Clock className="w-4 h-4 mr-2" />
              History
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-6 space-y-6">
            {/* Insights */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">Insights</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {insights.map((insight, idx) => (
                  <InsightCard key={idx} insight={insight} />
                ))}
              </div>
            </div>

            {/* Charts Row */}
            {dailyData.length > 0 && (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Calendar className="w-5 h-5 text-[#00d4ff]" />
                    Daily Activity
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={dailyData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                        <Bar dataKey="predictions" fill="rgba(0, 212, 255, 0.3)" name="Predictions" />
                        <Bar dataKey="wins" fill="rgba(0, 255, 136, 0.5)" name="Wins" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Sport Performance */}
            {sportPerformance.length > 0 && (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Performance by Sport</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10 hover:bg-transparent">
                        <TableHead className="text-gray-500">Sport</TableHead>
                        <TableHead className="text-gray-500">Predictions</TableHead>
                        <TableHead className="text-gray-500">Win Rate</TableHead>
                        <TableHead className="text-gray-500">ROI</TableHead>
                        <TableHead className="text-gray-500">Grade</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sportPerformance.map((sport) => (
                        <TableRow key={sport.sport} className="border-white/5 hover:bg-white/5">
                          <TableCell className="font-medium text-white">{sport.sport}</TableCell>
                          <TableCell className="text-gray-400">{sport.predictions}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                <div
                                  className={cn(
                                    'h-full rounded-full',
                                    sport.winRate >= 65 ? 'bg-[#00ff88]' :
                                    sport.winRate >= 55 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
                                  )}
                                  style={{ width: `${Math.min(sport.winRate, 100)}%` }}
                                />
                              </div>
                              <span className="text-sm text-gray-400">{sport.winRate.toFixed(1)}%</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <span className={cn('font-bold', sport.roi > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]')}>
                              {sport.roi > 0 ? '+' : ''}{sport.roi.toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell>
                            <Badge className={cn(
                              'border-0',
                              sport.roi >= 15 ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                              sport.roi >= 5 ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
                              sport.roi >= 0 ? 'bg-[#ff9500]/20 text-[#ff9500]' :
                              'bg-[#ff3860]/20 text-[#ff3860]'
                            )}>
                              {sport.roi >= 15 ? 'A+' : sport.roi >= 5 ? 'A' : sport.roi >= 0 ? 'B' : 'C'}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {!loading && sportPerformance.length === 0 && dailyData.length === 0 && (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center text-gray-500">
                  <PieChartIcon className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No performance data yet.</p>
                  <p className="text-xs mt-1">Run analyses to generate reports and insights.</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="mt-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Summary */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Performance Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-[#00ff88]/10 border border-[#00ff88]/20 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-[#00ff88]/20 flex items-center justify-center">
                          <TrendingUp className="w-5 h-5 text-[#00ff88]" />
                        </div>
                        <div>
                          <div className="font-medium text-white">Win Rate</div>
                          <div className="text-sm text-gray-400">Overall performance</div>
                        </div>
                      </div>
                      <div className="text-3xl font-bold text-[#00ff88]">{stats?.win_rate || 0}%</div>
                    </div>
                  </div>

                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                          <Target className="w-5 h-5 text-[#00d4ff]" />
                        </div>
                        <div>
                          <div className="font-medium text-white">Total Profit</div>
                          <div className="text-sm text-gray-400">Cumulative P/L</div>
                        </div>
                      </div>
                      <div className={cn(
                        'text-3xl font-bold',
                        (stats?.total_profit || 0) >= 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                      )}>
                        {(stats?.total_profit || 0) >= 0 ? '+' : ''}{(stats?.total_profit || 0).toFixed(1)}
                      </div>
                    </div>
                  </div>

                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                          <Zap className="w-5 h-5 text-[#ff9500]" />
                        </div>
                        <div>
                          <div className="font-medium text-white">ROI</div>
                          <div className="text-sm text-gray-400">Return on investment</div>
                        </div>
                      </div>
                      <div className={cn(
                        'text-3xl font-bold',
                        (stats?.roi || 0) >= 0 ? 'text-[#00d4ff]' : 'text-[#ff3860]'
                      )}>
                        {(stats?.roi || 0).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Sports breakdown */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Sports Breakdown</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {sportPerformance.length > 0 ? sportPerformance.map((sport) => (
                    <div key={sport.sport} className="flex items-center gap-3 p-3 bg-white/5 rounded-lg">
                      <div className="flex-1">
                        <div className="text-white font-medium text-sm">{sport.sport}</div>
                        <div className="text-xs text-gray-500">{sport.predictions} predictions</div>
                      </div>
                      <div className="text-right">
                        <div className={cn('font-bold', sport.roi > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]')}>
                          {sport.roi > 0 ? '+' : ''}{sport.roi.toFixed(1)}% ROI
                        </div>
                        <div className="text-xs text-gray-500">{sport.winRate.toFixed(0)}% win rate</div>
                      </div>
                    </div>
                  )) : (
                    <div className="text-center text-gray-500 p-6">
                      <p className="text-sm">No sport data yet.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white">Recent Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                {recentBets.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10 hover:bg-transparent">
                        <TableHead className="text-gray-500">Match</TableHead>
                        <TableHead className="text-gray-500">Selection</TableHead>
                        <TableHead className="text-gray-500">Odds</TableHead>
                        <TableHead className="text-gray-500">Result</TableHead>
                        <TableHead className="text-gray-500">P/L</TableHead>
                        <TableHead className="text-gray-500">Date</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {recentBets.map((bet) => (
                        <TableRow key={bet.id} className="border-white/5 hover:bg-white/5">
                          <TableCell className="font-medium text-white">{bet.match}</TableCell>
                          <TableCell className="text-gray-300">{bet.selection}</TableCell>
                          <TableCell className="font-mono text-gray-300">{bet.odds.toFixed(2)}</TableCell>
                          <TableCell>
                            <Badge className={cn(
                              'border-0',
                              bet.status === 'won' ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                              bet.status === 'lost' ? 'bg-[#ff3860]/20 text-[#ff3860]' :
                              'bg-[#ff9500]/20 text-[#ff9500]'
                            )}>
                              {bet.status === 'pending' && <Clock className="w-3 h-3 mr-1" />}
                              {bet.status.toUpperCase()}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <span className={cn(
                              'font-medium',
                              bet.profit_loss > 0 ? 'text-[#00ff88]' :
                              bet.profit_loss < 0 ? 'text-[#ff3860]' : 'text-gray-400'
                            )}>
                              {bet.profit_loss > 0 ? '+' : ''}{bet.profit_loss.toFixed(2)}
                            </span>
                          </TableCell>
                          <TableCell className="text-gray-500 text-sm">
                            {bet.created_at ? new Date(bet.created_at).toLocaleDateString() : '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="p-12 text-center text-gray-500">
                    <Clock className="w-10 h-10 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">No prediction history yet.</p>
                    <p className="text-xs mt-1">Results will appear here after bets are placed.</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default ReportsPage;
