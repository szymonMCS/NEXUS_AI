// components/PerformanceCharts.tsx
/**
 * Performance Charts - ROI, Win Rate, Historical Performance
 * Inspired by nerdytips.com/progress
 */

import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Award,
  Activity,
  Download,
  BarChart3,
  PieChartIcon
} from 'lucide-react';

// Types
interface DailyPerformance {
  date: string;
  bets: number;
  wins: number;
  losses: number;
  profit: number;
  roi: number;
  avgEdge: number;
  avgQuality: number;
}

interface SportPerformance {
  sport: string;
  bets: number;
  wins: number;
  winRate: number;
  profit: number;
  roi: number;
  color: string;
}

interface PerformanceStats {
  totalBets: number;
  totalWins: number;
  winRate: number;
  totalProfit: number;
  roi: number;
  avgEdge: number;
  avgQuality: number;
  bestStreak: number;
  currentStreak: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

interface PerformanceChartsProps {
  dailyData: DailyPerformance[];
  sportData: SportPerformance[];
  stats: PerformanceStats;
  period?: '7d' | '30d' | '90d' | 'all';
  onPeriodChange?: (period: '7d' | '30d' | '90d' | 'all') => void;
  onExport?: () => void;
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload) return null;

  return (
    <div className="bg-gray-900 border border-white/10 rounded-lg p-3 shadow-xl">
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      {payload.map((item: any, index: number) => (
        <div key={index} className="flex items-center gap-2 text-sm">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
          <span className="text-gray-300">{item.name}:</span>
          <span className="text-white font-medium">
            {item.name === 'ROI' || item.name === 'Win Rate' ? `${item.value.toFixed(1)}%` :
             item.name === 'Profit' ? `$${item.value.toFixed(2)}` :
             item.value}
          </span>
        </div>
      ))}
    </div>
  );
};

// Stat Card Component
const StatCard = ({ title, value, change, icon: Icon, positive }: {
  title: string;
  value: string;
  change?: string;
  icon: any;
  positive?: boolean;
}) => (
  <Card className="bg-glass-card border-white/5">
    <CardContent className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {change && (
            <p className={`text-xs mt-1 flex items-center gap-1 ${positive ? 'text-green-400' : 'text-red-400'}`}>
              {positive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
              {change}
            </p>
          )}
        </div>
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${positive === false ? 'bg-red-500/20' : 'bg-violet-500/20'}`}>
          <Icon className={`w-6 h-6 ${positive === false ? 'text-red-400' : 'text-violet-400'}`} />
        </div>
      </div>
    </CardContent>
  </Card>
);

export function PerformanceCharts({
  dailyData,
  sportData,
  stats,
  period = '30d',
  onPeriodChange,
  onExport
}: PerformanceChartsProps) {
  const [chartType, setChartType] = useState<'line' | 'area'>('area');

  // Calculate cumulative profit
  const cumulativeData = useMemo(() => {
    let cumProfit = 0;
    return dailyData.map(day => {
      cumProfit += day.profit;
      return {
        ...day,
        cumProfit,
        winRate: day.bets > 0 ? (day.wins / day.bets) * 100 : 0
      };
    });
  }, [dailyData]);

  // Pie chart colors
  const COLORS = ['#8b5cf6', '#3b82f6', '#22c55e', '#f59e0b', '#ef4444'];

  return (
    <div className="space-y-6">
      {/* Header with Period Selector */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Activity className="w-6 h-6 text-violet-400" />
            Performance Analytics
          </h2>
          <p className="text-gray-400 text-sm">Track your betting performance over time</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex bg-white/5 rounded-lg p-1">
            {(['7d', '30d', '90d', 'all'] as const).map((p) => (
              <Button
                key={p}
                size="sm"
                variant={period === p ? 'default' : 'ghost'}
                className={period === p ? 'bg-violet-500' : 'text-gray-400'}
                onClick={() => onPeriodChange?.(p)}
              >
                {p === 'all' ? 'All' : p}
              </Button>
            ))}
          </div>
          {onExport && (
            <Button size="sm" variant="outline" className="bg-white/5 border-white/10" onClick={onExport}>
              <Download className="w-4 h-4 mr-1" />
              Export
            </Button>
          )}
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Total Profit"
          value={`$${stats.totalProfit.toFixed(2)}`}
          change={`${stats.roi > 0 ? '+' : ''}${stats.roi.toFixed(1)}% ROI`}
          icon={DollarSign}
          positive={stats.totalProfit >= 0}
        />
        <StatCard
          title="Win Rate"
          value={`${stats.winRate.toFixed(1)}%`}
          change={`${stats.totalWins}/${stats.totalBets} bets`}
          icon={Target}
          positive={stats.winRate >= 50}
        />
        <StatCard
          title="Best Streak"
          value={`${stats.bestStreak} wins`}
          change={`Current: ${stats.currentStreak}`}
          icon={Award}
          positive={stats.currentStreak > 0}
        />
        <StatCard
          title="Avg Edge"
          value={`${(stats.avgEdge * 100).toFixed(1)}%`}
          change={`Quality: ${(stats.avgQuality * 100).toFixed(0)}%`}
          icon={TrendingUp}
          positive={stats.avgEdge > 0.03}
        />
      </div>

      {/* Main Charts */}
      <Tabs defaultValue="profit" className="space-y-4">
        <TabsList className="bg-white/5 border-white/10">
          <TabsTrigger value="profit" className="data-[state=active]:bg-violet-500/20">
            <DollarSign className="w-4 h-4 mr-1" /> Profit
          </TabsTrigger>
          <TabsTrigger value="winrate" className="data-[state=active]:bg-violet-500/20">
            <Target className="w-4 h-4 mr-1" /> Win Rate
          </TabsTrigger>
          <TabsTrigger value="sports" className="data-[state=active]:bg-violet-500/20">
            <PieChartIcon className="w-4 h-4 mr-1" /> By Sport
          </TabsTrigger>
          <TabsTrigger value="volume" className="data-[state=active]:bg-violet-500/20">
            <BarChart3 className="w-4 h-4 mr-1" /> Volume
          </TabsTrigger>
        </TabsList>

        {/* Profit Chart */}
        <TabsContent value="profit">
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg text-white">Cumulative Profit</CardTitle>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant={chartType === 'area' ? 'default' : 'ghost'}
                    className={chartType === 'area' ? 'bg-violet-500/50' : ''}
                    onClick={() => setChartType('area')}
                  >
                    Area
                  </Button>
                  <Button
                    size="sm"
                    variant={chartType === 'line' ? 'default' : 'ghost'}
                    className={chartType === 'line' ? 'bg-violet-500/50' : ''}
                    onClick={() => setChartType('line')}
                  >
                    Line
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  {chartType === 'area' ? (
                    <AreaChart data={cumulativeData}>
                      <defs>
                        <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="date" stroke="#666" tick={{ fill: '#999', fontSize: 12 }} />
                      <YAxis stroke="#666" tick={{ fill: '#999', fontSize: 12 }} tickFormatter={(v) => `$${v}`} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area
                        type="monotone"
                        dataKey="cumProfit"
                        name="Profit"
                        stroke="#8b5cf6"
                        fill="url(#profitGradient)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  ) : (
                    <LineChart data={cumulativeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="date" stroke="#666" tick={{ fill: '#999', fontSize: 12 }} />
                      <YAxis stroke="#666" tick={{ fill: '#999', fontSize: 12 }} tickFormatter={(v) => `$${v}`} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line
                        type="monotone"
                        dataKey="cumProfit"
                        name="Profit"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  )}
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Win Rate Chart */}
        <TabsContent value="winrate">
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white">Daily Win Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={cumulativeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="date" stroke="#666" tick={{ fill: '#999', fontSize: 12 }} />
                    <YAxis stroke="#666" tick={{ fill: '#999', fontSize: 12 }} domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="winRate"
                      name="Win Rate"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={{ fill: '#22c55e', r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="avgQuality"
                      name="Avg Quality"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      strokeDasharray="5 5"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Sports Breakdown */}
        <TabsContent value="sports">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Pie Chart */}
            <Card className="bg-glass-card border-white/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-white">Bets by Sport</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={sportData}
                        dataKey="bets"
                        nameKey="sport"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        labelLine={false}
                      >
                        {sportData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color || COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Sport Stats Table */}
            <Card className="bg-glass-card border-white/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-white">Performance by Sport</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {sportData.map((sport, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: sport.color || COLORS[i % COLORS.length] }} />
                        <div>
                          <div className="font-medium text-white">{sport.sport}</div>
                          <div className="text-xs text-gray-400">{sport.bets} bets</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-medium ${sport.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {sport.profit >= 0 ? '+' : ''}${sport.profit.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">{sport.winRate.toFixed(1)}% WR</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Volume Chart */}
        <TabsContent value="volume">
          <Card className="bg-glass-card border-white/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white">Daily Betting Volume</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={cumulativeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="date" stroke="#666" tick={{ fill: '#999', fontSize: 12 }} />
                    <YAxis stroke="#666" tick={{ fill: '#999', fontSize: 12 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Bar dataKey="wins" name="Wins" stackId="a" fill="#22c55e" />
                    <Bar dataKey="losses" name="Losses" stackId="a" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Risk Metrics */}
      <Card className="bg-glass-card border-white/5">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg text-white">Risk Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-white/5 rounded-lg">
              <div className="text-2xl font-bold text-white">{stats.sharpeRatio.toFixed(2)}</div>
              <div className="text-xs text-gray-400">Sharpe Ratio</div>
              <Badge className={stats.sharpeRatio >= 1 ? 'bg-green-500/20 text-green-400' : 'bg-orange-500/20 text-orange-400'}>
                {stats.sharpeRatio >= 2 ? 'Excellent' : stats.sharpeRatio >= 1 ? 'Good' : 'Moderate'}
              </Badge>
            </div>
            <div className="text-center p-4 bg-white/5 rounded-lg">
              <div className="text-2xl font-bold text-red-400">-{(stats.maxDrawdown * 100).toFixed(1)}%</div>
              <div className="text-xs text-gray-400">Max Drawdown</div>
              <Badge className={stats.maxDrawdown <= 0.1 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}>
                {stats.maxDrawdown <= 0.05 ? 'Low' : stats.maxDrawdown <= 0.1 ? 'Moderate' : 'High'}
              </Badge>
            </div>
            <div className="text-center p-4 bg-white/5 rounded-lg">
              <div className="text-2xl font-bold text-white">{stats.totalBets}</div>
              <div className="text-xs text-gray-400">Total Bets</div>
              <Badge className="bg-violet-500/20 text-violet-400">
                {(stats.totalBets / 30).toFixed(1)}/day avg
              </Badge>
            </div>
            <div className="text-center p-4 bg-white/5 rounded-lg">
              <div className="text-2xl font-bold text-white">{(stats.avgQuality * 100).toFixed(0)}%</div>
              <div className="text-xs text-gray-400">Avg Data Quality</div>
              <Badge className={stats.avgQuality >= 0.7 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}>
                {stats.avgQuality >= 0.8 ? 'Excellent' : stats.avgQuality >= 0.6 ? 'Good' : 'Fair'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default PerformanceCharts;
