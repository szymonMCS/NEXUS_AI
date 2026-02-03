/**
 * HistoryPage - Prediction History and Results
 * 
 * Features:
 * - Complete prediction history with filters
 * - Results tracking and analysis
 * - Profit/loss statements
 * - Export capabilities
 * - Timeline view
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
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
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  Calendar,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  Filter,
  Search,
  TrendingUp,
  XCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data
const monthlyData = [
  { month: 'Aug', predictions: 145, wins: 92, profit: 1250 },
  { month: 'Sep', predictions: 180, wins: 115, profit: 1890 },
  { month: 'Oct', predictions: 165, wins: 108, profit: 1560 },
  { month: 'Nov', predictions: 190, wins: 124, profit: 2240 },
  { month: 'Dec', predictions: 175, wins: 112, profit: 1680 },
  { month: 'Jan', predictions: 156, wins: 104, profit: 1420 },
];

const predictionHistory = [
  { id: 1, date: '2026-01-28', time: '18:30', league: 'Premier League', match: 'Man City vs Arsenal', selection: 'Man City', odds: 1.65, stake: 100, result: 'WIN', profit: 65, confidence: 82, model: 'Ensemble' },
  { id: 2, date: '2026-01-28', time: '21:00', league: 'La Liga', match: 'Real Madrid vs Barcelona', selection: 'Draw', odds: 3.40, stake: 50, result: 'LOSS', profit: -50, confidence: 58, model: 'MLP' },
  { id: 3, date: '2026-01-27', time: '20:45', league: 'Serie A', match: 'Juventus vs Inter', selection: 'Inter', odds: 2.20, stake: 75, result: 'WIN', profit: 90, confidence: 71, model: 'RF+ARA' },
  { id: 4, date: '2026-01-27', time: '19:30', league: 'Bundesliga', match: 'Bayern vs Dortmund', selection: 'Bayern', odds: 1.45, stake: 100, result: 'WIN', profit: 45, confidence: 78, model: 'Ensemble' },
  { id: 5, date: '2026-01-26', time: '17:00', league: 'Ligue 1', match: 'PSG vs Marseille', selection: 'PSG', odds: 1.55, stake: 100, result: 'WIN', profit: 55, confidence: 75, model: 'MLP' },
  { id: 6, date: '2026-01-26', time: '16:00', league: 'Premier League', match: 'Liverpool vs Chelsea', selection: 'Liverpool', odds: 1.80, stake: 100, result: 'LOSS', profit: -100, confidence: 68, model: 'RF+ARA' },
  { id: 7, date: '2026-01-25', time: '15:30', league: 'La Liga', match: 'Atletico vs Sevilla', selection: 'Atletico', odds: 1.70, stake: 100, result: 'WIN', profit: 70, confidence: 73, model: 'Ensemble' },
  { id: 8, date: '2026-01-25', time: '14:00', league: 'Serie A', match: 'AC Milan vs Roma', selection: 'Roma', odds: 2.80, stake: 50, result: 'WIN', profit: 90, confidence: 65, model: 'MLP' },
];

const summaryStats = {
  totalPredictions: 391,
  totalWins: 259,
  totalLosses: 132,
  winRate: 66.2,
  totalStake: 28450,
  totalReturn: 32650,
  totalProfit: 4200,
  roi: 14.8,
  avgOdds: 1.84,
  avgConfidence: 72.4,
};

const modelBreakdown = [
  { model: 'Ensemble', predictions: 156, wins: 112, winRate: 71.8, profit: 2450, roi: 18.2 },
  { model: 'MLP + PCA', predictions: 124, wins: 79, winRate: 63.7, profit: 1180, roi: 11.4 },
  { model: 'RF + ARA', predictions: 89, wins: 56, winRate: 62.9, profit: 520, roi: 7.8 },
  { model: 'Transformer', predictions: 22, wins: 12, winRate: 54.5, profit: 50, roi: 3.2 },
];

export function HistoryPage() {
  const [activeTab, setActiveTab] = useState('history');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = 12;

  return (
    <PageLayout
      title="Prediction History"
      description="Complete history of all predictions and results"
      breadcrumbs={[{ label: 'History' }]}
      actions={
        <Button variant="outline" className="border-white/10 text-gray-400">
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      }
    >
      <div className="space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Predictions</div>
              <div className="text-2xl font-bold text-white">{summaryStats.totalPredictions}</div>
              <div className="text-xs text-[#00ff88] mt-1">{summaryStats.winRate}% win rate</div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Profit</div>
              <div className="text-2xl font-bold text-[#00ff88]">+${summaryStats.totalProfit.toLocaleString()}</div>
              <div className="text-xs text-[#00ff88] mt-1">{summaryStats.roi}% ROI</div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Win / Loss</div>
              <div className="text-2xl font-bold text-white">{summaryStats.totalWins} / {summaryStats.totalLosses}</div>
              <div className="flex gap-1 mt-1">
                <div className="h-1.5 flex-1 bg-[#00ff88] rounded-l" style={{ width: `${summaryStats.winRate}%` }} />
                <div className="h-1.5 flex-1 bg-[#ff3860] rounded-r" style={{ width: `${100 - summaryStats.winRate}%` }} />
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Avg Odds</div>
              <div className="text-2xl font-bold text-white">{summaryStats.avgOdds}</div>
              <div className="text-xs text-gray-400 mt-1">Confidence: {summaryStats.avgConfidence}%</div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="history" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Calendar className="w-4 h-4 mr-2" />
              History
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <TrendingUp className="w-4 h-4 mr-2" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Filter className="w-4 h-4 mr-2" />
              By Model
            </TabsTrigger>
          </TabsList>

          {/* History Tab */}
          <TabsContent value="history" className="mt-6 space-y-4">
            {/* Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <Input
                  placeholder="Search matches, teams, leagues..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-white/5 border-white/10 text-white"
                />
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                  <Calendar className="w-4 h-4 mr-2" />
                  Date Range
                </Button>
              </div>
            </div>

            {/* History Table */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Date</TableHead>
                      <TableHead className="text-gray-500">League</TableHead>
                      <TableHead className="text-gray-500">Match</TableHead>
                      <TableHead className="text-gray-500">Selection</TableHead>
                      <TableHead className="text-gray-500">Odds</TableHead>
                      <TableHead className="text-gray-500">Stake</TableHead>
                      <TableHead className="text-gray-500">Result</TableHead>
                      <TableHead className="text-gray-500">P/L</TableHead>
                      <TableHead className="text-gray-500">Conf.</TableHead>
                      <TableHead className="text-gray-500">Model</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {predictionHistory.map((pred) => (
                      <TableRow key={pred.id} className="border-white/5 hover:bg-white/5">
                        <TableCell className="text-gray-400 text-sm">
                          <div>{pred.date}</div>
                          <div className="text-gray-600">{pred.time}</div>
                        </TableCell>
                        <TableCell className="text-gray-400 text-sm">{pred.league}</TableCell>
                        <TableCell className="text-white">{pred.match}</TableCell>
                        <TableCell className="text-[#00d4ff]">{pred.selection}</TableCell>
                        <TableCell className="font-mono text-white">{pred.odds.toFixed(2)}</TableCell>
                        <TableCell className="text-white">${pred.stake}</TableCell>
                        <TableCell>
                          <Badge className={cn(
                            'border-0',
                            pred.result === 'WIN' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#ff3860]/20 text-[#ff3860]'
                          )}>
                            {pred.result === 'WIN' ? <CheckCircle2 className="w-3 h-3 mr-1" /> : <XCircle className="w-3 h-3 mr-1" />}
                            {pred.result}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className={cn(
                            'font-medium',
                            pred.profit > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                          )}>
                            {pred.profit > 0 ? '+' : ''}{pred.profit}
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  'h-full rounded-full',
                                  pred.confidence >= 75 ? 'bg-[#00ff88]' : pred.confidence >= 60 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
                                )}
                                style={{ width: `${pred.confidence}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-400">{pred.confidence}%</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-gray-400 text-sm">{pred.model}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Pagination */}
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-500">
                Showing {(currentPage - 1) * 10 + 1} to {currentPage * 10} of {totalPages * 10} results
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-white/10 text-gray-400"
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-gray-400">
                  Page {currentPage} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-white/10 text-gray-400"
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="mt-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Monthly Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={monthlyData}>
                        <defs>
                          <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="month" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                        <Area type="monotone" dataKey="profit" stroke="#00ff88" fillOpacity={1} fill="url(#colorProfit)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Win Rate by Month</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={monthlyData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="month" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} domain={[0, 200]} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                        <Bar dataKey="predictions" fill="rgba(0, 212, 255, 0.5)" name="Predictions" />
                        <Bar dataKey="wins" fill="rgba(0, 255, 136, 0.5)" name="Wins" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Cumulative Profit Chart */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white">Cumulative Profit/Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={[
                      { day: 1, cumulative: 0 },
                      { day: 5, cumulative: 120 },
                      { day: 10, cumulative: 280 },
                      { day: 15, cumulative: 450 },
                      { day: 20, cumulative: 380 },
                      { day: 25, cumulative: 620 },
                      { day: 30, cumulative: 850 },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="day" stroke="#64748b" fontSize={12} label={{ value: 'Day of Month', position: 'bottom', fill: '#64748b' }} />
                      <YAxis stroke="#64748b" fontSize={12} />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#0f1623',
                          border: '1px solid rgba(255,255,255,0.1)',
                        }}
                        formatter={(value: number) => [`$${value}`, 'Cumulative P/L']}
                      />
                      <Line type="monotone" dataKey="cumulative" stroke="#00ff88" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* By Model Tab */}
          <TabsContent value="models" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white">Performance by Model</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Model</TableHead>
                      <TableHead className="text-gray-500">Predictions</TableHead>
                      <TableHead className="text-gray-500">Wins</TableHead>
                      <TableHead className="text-gray-500">Win Rate</TableHead>
                      <TableHead className="text-gray-500">Profit</TableHead>
                      <TableHead className="text-gray-500">ROI</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {modelBreakdown.map((model) => (
                      <TableRow key={model.model} className="border-white/5 hover:bg-white/5">
                        <TableCell className="font-medium text-white">{model.model}</TableCell>
                        <TableCell className="text-gray-400">{model.predictions}</TableCell>
                        <TableCell className="text-[#00ff88]">{model.wins}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  'h-full rounded-full',
                                  model.winRate >= 65 ? 'bg-[#00ff88]' : 'bg-[#00d4ff]'
                                )}
                                style={{ width: `${model.winRate}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-400">{model.winRate}%</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className={cn(
                            'font-medium',
                            model.profit > 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
                          )}>
                            {model.profit > 0 ? '+' : ''}${model.profit}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Badge className={cn(
                            'border-0',
                            model.roi >= 15 ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                            model.roi >= 10 ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
                            model.roi >= 5 ? 'bg-[#ff9500]/20 text-[#ff9500]' :
                            'bg-[#ff3860]/20 text-[#ff3860]'
                          )}>
                            {model.roi > 0 ? '+' : ''}{model.roi}%
                          </Badge>
                        </TableCell>
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

export default HistoryPage;
