/**
 * ReportsPage - Comprehensive analytical reports and summaries
 * 
 * Features:
 * - Prediction summaries with confidence scores
 * - Key influencing factors
 * - Similar historical matches
 * - Win rate and ROI analysis
 * - Model accuracy metrics
 * - Daily/weekly/monthly performance
 * - Automatic insight generation
 */

import { useState } from 'react';
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
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
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
  PieChart as PieChartIcon,
  Target,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data for reports
const dailyPerformance = [
  { date: 'Mon', predictions: 12, wins: 8, roi: 15.2 },
  { date: 'Tue', predictions: 15, wins: 9, roi: 8.5 },
  { date: 'Wed', predictions: 10, wins: 7, roi: 22.1 },
  { date: 'Thu', predictions: 18, wins: 10, roi: -5.3 },
  { date: 'Fri', predictions: 14, wins: 9, roi: 12.8 },
  { date: 'Sat', predictions: 22, wins: 15, roi: 28.4 },
  { date: 'Sun', predictions: 20, wins: 13, roi: 18.9 },
];

const modelAccuracy = [
  { metric: 'Precision', value: 72, fullMark: 100 },
  { metric: 'Recall', value: 68, fullMark: 100 },
  { metric: 'F1 Score', value: 70, fullMark: 100 },
  { metric: 'Calibration', value: 85, fullMark: 100 },
  { metric: 'ROI', value: 78, fullMark: 100 },
  { metric: 'Consistency', value: 82, fullMark: 100 },
];

const predictionBreakdown = [
  { name: 'High Confidence', value: 45, color: '#00ff88' },
  { name: 'Medium Confidence', value: 35, color: '#00d4ff' },
  { name: 'Low Confidence', value: 20, color: '#ff9500' },
];

const sportPerformance = [
  { sport: 'Football', predictions: 156, winRate: 68.5, roi: 15.2, avgOdds: 1.85 },
  { sport: 'Basketball', predictions: 89, winRate: 62.1, roi: 8.7, avgOdds: 1.72 },
  { sport: 'Tennis', predictions: 67, winRate: 71.2, roi: 22.4, avgOdds: 1.68 },
  { sport: 'Hockey', predictions: 45, winRate: 58.9, roi: 3.2, avgOdds: 1.90 },
  { sport: 'Baseball', predictions: 34, winRate: 61.8, roi: 6.5, avgOdds: 1.78 },
];

const recentPredictions = [
  { id: 1, match: 'Man City vs Arsenal', prediction: 'Home Win', confidence: 82, result: 'WIN', odds: 1.65, profit: '+6.5%', date: '2h ago' },
  { id: 2, match: 'Lakers vs Warriors', prediction: 'Away Win', confidence: 65, result: 'LOSS', odds: 1.75, profit: '-10.0%', date: '4h ago' },
  { id: 3, match: 'Real vs Barcelona', prediction: 'Draw', confidence: 58, result: 'WIN', odds: 3.40, profit: '+24.0%', date: '6h ago' },
  { id: 4, match: 'Alcaraz vs Djokovic', prediction: 'Home Win', confidence: 71, result: 'WIN', odds: 1.85, profit: '+8.5%', date: '8h ago' },
  { id: 5, match: 'Celtics vs Heat', prediction: 'Home Win', confidence: 78, result: 'PENDING', odds: 1.45, profit: '-', date: 'Upcoming' },
];

const insights = [
  {
    type: 'positive' as const,
    title: 'Strong Tennis Performance',
    description: 'Tennis predictions showing 71.2% win rate with +22.4% ROI. Model excelling on ATP matches.',
    action: 'Review Model',
  },
  {
    type: 'warning' as const,
    title: 'Hockey Underperformance',
    description: 'Hockey predictions below expectations at 58.9% win rate. Consider reducing stake size.',
    action: 'Adjust Strategy',
  },
  {
    type: 'info' as const,
    title: 'High Confidence Streak',
    description: 'High confidence predictions (80%+) on a 7-win streak. Edge detection working well.',
    action: 'View Details',
  },
  {
    type: 'positive' as const,
    title: 'Weekend Performance',
    description: 'Saturday showing exceptional results with 28.4% ROI. Market inefficiencies detected.',
    action: 'Analyze',
  },
];

const similarMatches = [
  { match: 'Liverpool vs Man Utd', date: '2024-01-15', prediction: 'Home Win', actual: 'Home Win', odds: 1.72, similarity: 92 },
  { match: 'Chelsea vs Arsenal', date: '2024-01-08', prediction: 'Draw', actual: 'Draw', odds: 3.20, similarity: 88 },
  { match: 'Spurs vs City', date: '2023-12-20', prediction: 'Away Win', actual: 'Away Win', odds: 1.55, similarity: 85 },
  { match: 'Newcastle vs Brighton', date: '2023-12-15', prediction: 'Home Win', actual: 'Home Win', odds: 1.90, similarity: 81 },
];

// Helper components
const KPICard = ({ 
  title, 
  value, 
  change,
  changeType = 'neutral',
  icon: Icon,
  description 
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
  action: string;
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
            <Button variant="link" size="sm" className="text-[#00d4ff] p-0 h-auto mt-2">
              {insight.action} →
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export function ReportsPage() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <PageLayout
      title="Reports & Analytics"
      description="Comprehensive performance analysis and predictive insights"
      breadcrumbs={[{ label: 'Reports' }]}
      actions={
        <Button variant="outline" className="border-white/10 text-gray-400">
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      }
    >
      <div className="space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total Predictions"
            value="391"
            change="+12% vs last week"
            changeType="positive"
            icon={Target}
            description="Last 30 days"
          />
          <KPICard
            title="Win Rate"
            value="66.2%"
            change="+3.5% vs last week"
            changeType="positive"
            icon={Zap}
            description="259 wins / 132 losses"
          />
          <KPICard
            title="Total ROI"
            value="+14.8%"
            change="+2.1% vs last week"
            changeType="positive"
            icon={TrendingUp}
            description="Cumulative return"
          />
          <KPICard
            title="Avg. Confidence"
            value="72.4%"
            change="+1.2% vs last week"
            changeType="positive"
            icon={Brain}
            description="Model certainty"
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
            <TabsTrigger value="model" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Brain className="w-4 h-4 mr-2" />
              Model Metrics
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
              <h3 className="text-sm font-medium text-gray-400 mb-4">AI-Generated Insights</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {insights.map((insight, idx) => (
                  <InsightCard key={idx} insight={insight} />
                ))}
              </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Daily Performance */}
              <Card className="lg:col-span-2 bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Calendar className="w-5 h-5 text-[#00d4ff]" />
                    Daily Performance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={dailyPerformance}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" stroke="#64748b" fontSize={12} />
                        <YAxis yAxisId="left" stroke="#64748b" fontSize={12} />
                        <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={12} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                        <Bar yAxisId="left" dataKey="predictions" fill="rgba(0, 212, 255, 0.3)" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Confidence Distribution */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white text-base">Confidence Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={predictionBreakdown}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          dataKey="value"
                        >
                          {predictionBreakdown.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="space-y-2 mt-4">
                    {predictionBreakdown.map((item) => (
                      <div key={item.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                          <span className="text-sm text-gray-400">{item.name}</span>
                        </div>
                        <span className="text-sm font-medium text-white">{item.value}%</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sport Performance */}
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
                      <TableHead className="text-gray-500">Avg Odds</TableHead>
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
                                  sport.winRate >= 60 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
                                )}
                                style={{ width: `${sport.winRate}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-400">{sport.winRate}%</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className={cn(
                            'font-bold',
                            sport.roi > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                          )}>
                            {sport.roi > 0 ? '+' : ''}{sport.roi}%
                          </span>
                        </TableCell>
                        <TableCell className="text-gray-400">{sport.avgOdds}</TableCell>
                        <TableCell>
                          <Badge className={cn(
                            'border-0',
                            sport.roi >= 15 ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                            sport.roi >= 10 ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
                            sport.roi >= 5 ? 'bg-[#ff9500]/20 text-[#ff9500]' :
                            'bg-[#ff3860]/20 text-[#ff3860]'
                          )}>
                            {sport.roi >= 15 ? 'A+' : sport.roi >= 10 ? 'A' : sport.roi >= 5 ? 'B' : 'C'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="mt-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Similar Historical Matches */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Clock className="w-5 h-5 text-[#00d4ff]" />
                    Similar Historical Matches
                  </CardTitle>
                  <CardDescription className="text-gray-400">
                    Past matches with similar statistical profiles
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {similarMatches.map((match, idx) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                        <div>
                          <div className="text-sm font-medium text-white">{match.match}</div>
                          <div className="text-xs text-gray-500">{match.date}</div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <div className="text-xs text-gray-500">Prediction</div>
                            <div className="text-sm text-[#00d4ff]">{match.prediction}</div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-gray-500">Result</div>
                            <Badge className={cn(
                              'border-0',
                              match.prediction === match.actual 
                                ? 'bg-[#00ff88]/20 text-[#00ff88]' 
                                : 'bg-[#ff3860]/20 text-[#ff3860]'
                            )}>
                              {match.prediction === match.actual ? 'WIN' : 'LOSS'}
                            </Badge>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-gray-500">Match</div>
                            <div className="text-sm font-mono text-white">{match.similarity}%</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Streaks */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Current Streaks</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-[#00ff88]/10 border border-[#00ff88]/20 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-[#00ff88]/20 flex items-center justify-center">
                          <TrendingUp className="w-5 h-5 text-[#00ff88]" />
                        </div>
                        <div>
                          <div className="font-medium text-white">Win Streak</div>
                          <div className="text-sm text-gray-400">High confidence predictions</div>
                        </div>
                      </div>
                      <div className="text-3xl font-bold text-[#00ff88]">7</div>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                          <Target className="w-5 h-5 text-[#00d4ff]" />
                        </div>
                        <div>
                          <div className="font-medium text-white">Profit Streak</div>
                          <div className="text-sm text-gray-400">Consecutive profitable days</div>
                        </div>
                      </div>
                      <div className="text-3xl font-bold text-[#00d4ff]">5</div>
                    </div>
                  </div>

                  <div className="p-4 bg-[#ff9500]/10 border border-[#ff9500]/20 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-[#ff9500]/20 flex items-center justify-center">
                          <AlertTriangle className="w-5 h-5 text-[#ff9500]" />
                        </div>
                        <div>
                          <div className="font-medium text-white">ROI Variance</div>
                          <div className="text-sm text-gray-400">Standard deviation: 18.4%</div>
                        </div>
                      </div>
                      <div className="text-sm text-gray-400">Moderate</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Model Metrics Tab */}
          <TabsContent value="model" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Model Accuracy Radar</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={modelAccuracy}>
                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
                        <Radar
                          name="Current Model"
                          dataKey="value"
                          stroke="#00d4ff"
                          strokeWidth={2}
                          fill="#00d4ff"
                          fillOpacity={0.3}
                        />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Calibration Analysis</CardTitle>
                  <CardDescription className="text-gray-400">
                    Predicted probability vs actual win rate
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      { range: '90-100%', predicted: 95, actual: 94, accuracy: 99 },
                      { range: '80-89%', predicted: 85, actual: 82, accuracy: 96 },
                      { range: '70-79%', predicted: 75, actual: 72, accuracy: 96 },
                      { range: '60-69%', predicted: 65, actual: 63, accuracy: 97 },
                      { range: '50-59%', predicted: 55, actual: 52, accuracy: 95 },
                    ].map((item) => (
                      <div key={item.range} className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Confidence {item.range}</span>
                          <span className="text-white">{item.accuracy}% calibration</span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#00d4ff] rounded-full"
                            style={{ width: `${item.accuracy}%` }}
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>Predicted: {item.predicted}%</span>
                          <span>Actual: {item.actual}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
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
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Match</TableHead>
                      <TableHead className="text-gray-500">Prediction</TableHead>
                      <TableHead className="text-gray-500">Conf.</TableHead>
                      <TableHead className="text-gray-500">Odds</TableHead>
                      <TableHead className="text-gray-500">Result</TableHead>
                      <TableHead className="text-gray-500">P/L</TableHead>
                      <TableHead className="text-gray-500">Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentPredictions.map((pred) => (
                      <TableRow key={pred.id} className="border-white/5 hover:bg-white/5">
                        <TableCell className="font-medium text-white">{pred.match}</TableCell>
                        <TableCell className="text-gray-300">{pred.prediction}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  'h-full rounded-full',
                                  pred.confidence >= 75 ? 'bg-[#00ff88]' :
                                  pred.confidence >= 60 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
                                )}
                                style={{ width: `${pred.confidence}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-400">{pred.confidence}%</span>
                          </div>
                        </TableCell>
                        <TableCell className="font-mono text-gray-300">{pred.odds}</TableCell>
                        <TableCell>
                          <Badge className={cn(
                            'border-0',
                            pred.result === 'WIN' ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                            pred.result === 'LOSS' ? 'bg-[#ff3860]/20 text-[#ff3860]' :
                            'bg-[#ff9500]/20 text-[#ff9500]'
                          )}>
                            {pred.result === 'PENDING' && <Clock className="w-3 h-3 mr-1" />}
                            {pred.result}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className={cn(
                            'font-medium',
                            pred.profit.startsWith('+') ? 'text-[#00ff88]' :
                            pred.profit.startsWith('-') ? 'text-[#ff3860]' :
                            'text-gray-400'
                          )}>
                            {pred.profit}
                          </span>
                        </TableCell>
                        <TableCell className="text-gray-500 text-sm">{pred.date}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default ReportsPage;
