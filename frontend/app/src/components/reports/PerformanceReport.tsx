/**
 * PerformanceReport - Detailed performance analytics section
 * 
 * Shows:
 * - ROI over time
 * - Win rate by category
 * - Profit/loss breakdown
 * - Risk metrics
 */

import { cn } from '@/lib/utils';
import { KPICard, DataTable, RiskIndicator, StreakIndicator } from '@/components/analytics';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';

interface DailyData {
  date: string;
  profit: number;
  roi: number;
  bets: number;
  wins: number;
}

interface SportPerformance {
  sport: string;
  bets: number;
  winRate: number;
  profit: number;
  roi: number;
  color: string;
}

interface PerformanceReportProps {
  summary: {
    totalBets: number;
    winRate: number;
    totalProfit: number;
    roi: number;
    avgOdds: number;
    avgEdge: number;
  };
  dailyData: DailyData[];
  sportData: SportPerformance[];
  riskMetrics: {
    variance: number;
    maxDrawdown: number;
    kellyFraction: number;
    bankrollRisk: number;
  };
  streaks: {
    current: number;
    best: number;
    worst: number;
  };
  className?: string;
}

export function PerformanceReport({
  summary,
  dailyData,
  sportData,
  riskMetrics,
  streaks,
  className,
}: PerformanceReportProps) {
  const profitTrend = dailyData.length > 1 
    ? ((dailyData[dailyData.length - 1].profit - dailyData[0].profit) / Math.abs(dailyData[0].profit || 1)) * 100
    : 0;

  return (
    <div className={cn("space-y-4", className)}>
      {/* Summary KPIs */}
      <div className="grid grid-cols-6 gap-3">
        <KPICard
          label="Total Profit"
          value={`${summary.totalProfit >= 0 ? '+' : ''}${summary.totalProfit.toFixed(0)} PLN`}
          change={profitTrend}
          changeType={summary.totalProfit >= 0 ? 'positive' : 'negative'}
          sparklineData={dailyData.map(d => d.profit)}
        />
        <KPICard
          label="Win Rate"
          value={`${summary.winRate.toFixed(1)}%`}
          change={2.5}
          changeType="positive"
        />
        <KPICard
          label="ROI"
          value={`${summary.roi >= 0 ? '+' : ''}${summary.roi.toFixed(1)}%`}
          change={1.2}
          changeType={summary.roi >= 0 ? 'positive' : 'negative'}
        />
        <KPICard
          label="Total Bets"
          value={summary.totalBets}
          subValue={`Avg: ${(summary.totalBets / Math.max(dailyData.length, 1)).toFixed(1)}/day`}
        />
        <KPICard
          label="Avg Odds"
          value={summary.avgOdds.toFixed(2)}
          subValue="Target: 2.0"
        />
        <KPICard
          label="Avg Edge"
          value={`+${(summary.avgEdge * 100).toFixed(1)}%`}
          change={0.5}
          changeType="positive"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Profit Chart */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">
            Profit Over Time
          </h4>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={dailyData}>
                <defs>
                  <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#059669" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#059669" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ fontSize: 12, borderRadius: 6 }}
                  formatter={(value: number) => [`${value >= 0 ? '+' : ''}${value.toFixed(0)} PLN`, 'Profit']}
                />
                <Area 
                  type="monotone" 
                  dataKey="profit" 
                  stroke="#059669" 
                  fillOpacity={1} 
                  fill="url(#profitGradient)" 
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Sport Breakdown */}
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">
            By Sport
          </h4>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sportData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis 
                  dataKey="sport" 
                  type="category" 
                  tick={{ fontSize: 11 }}
                  width={70}
                />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6 }} />
                <Bar dataKey="roi" radius={[0, 4, 4, 0]}>
                  {sportData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.roi >= 0 ? '#059669' : '#dc2626'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Risk & Streaks */}
      <div className="grid grid-cols-3 gap-4">
        <RiskIndicator
          riskLevel={riskMetrics.variance > 0.3 ? 'high' : riskMetrics.variance > 0.15 ? 'medium' : 'low'}
          variance={riskMetrics.variance}
          maxDrawdown={riskMetrics.maxDrawdown}
          kellyFraction={riskMetrics.kellyFraction}
          bankrollRisk={riskMetrics.bankrollRisk}
          className="col-span-2"
        />
        <StreakIndicator
          currentStreak={streaks.current}
          bestStreak={streaks.best}
          worstStreak={streaks.worst}
        />
      </div>

      {/* Detailed Stats Table */}
      <div className="bg-white border border-slate-200 rounded-lg">
        <div className="px-4 py-3 border-b border-slate-200">
          <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wide">
            Daily Breakdown
          </h4>
        </div>
        <DataTable
          columns={[
            { key: 'date', header: 'Date', width: '100px', formatter: (v) => new Date(v as string).toLocaleDateString() },
            { key: 'bets', header: 'Bets', align: 'right', width: '80px' },
            { key: 'wins', header: 'Wins', align: 'right', width: '80px' },
            { 
              key: 'winRate', 
              header: 'Win %', 
              align: 'right', 
              width: '80px',
              formatter: (_, row) => `${((row.wins / Math.max(row.bets, 1)) * 100).toFixed(1)}%`
            },
            { 
              key: 'profit', 
              header: 'Profit', 
              align: 'right',
              formatter: (v) => (
                <span className={v as number >= 0 ? 'text-emerald-600' : 'text-red-600'}>
                  {v as number >= 0 ? '+' : ''}{(v as number).toFixed(0)} PLN
                </span>
              )
            },
            { 
              key: 'roi', 
              header: 'ROI', 
              align: 'right',
              formatter: (v) => (
                <span className={v as number >= 0 ? 'text-emerald-600' : 'text-red-600'}>
                  {v as number >= 0 ? '+' : ''}{(v as number).toFixed(1)}%
                </span>
              )
            },
          ]}
          data={dailyData.slice(-10).reverse()}
          keyExtractor={(row) => row.date}
        />
      </div>
    </div>
  );
}

export default PerformanceReport;
