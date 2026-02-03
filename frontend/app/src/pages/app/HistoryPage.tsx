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

import { useState, useEffect, useCallback } from 'react';
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
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
} from 'recharts';
import {
  Calendar,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  Filter,
  Loader2,
  Search,
  TrendingUp,
  XCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api, { type BetHistoryItem } from '@/lib/api';

const PAGE_SIZE = 20;

export function HistoryPage() {
  const [activeTab, setActiveTab] = useState('history');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [bets, setBets] = useState<BetHistoryItem[]>([]);
  const [totalBets, setTotalBets] = useState(0);
  const [summaryStats, setSummaryStats] = useState({
    totalPredictions: 0,
    totalWins: 0,
    totalLosses: 0,
    winRate: 0,
    totalProfit: 0,
    roi: 0,
    avgOdds: 0,
  });

  const totalPages = Math.max(1, Math.ceil(totalBets / PAGE_SIZE));

  const fetchHistory = useCallback(async (page: number) => {
    setLoading(true);
    try {
      const [historyRes, statsRes] = await Promise.allSettled([
        api.getBetHistory({
          limit: PAGE_SIZE,
          offset: (page - 1) * PAGE_SIZE,
          days: 365,
        }),
        api.getStats(),
      ]);

      if (historyRes.status === 'fulfilled') {
        setBets(historyRes.value.bets);
        setTotalBets(historyRes.value.total);
      }

      if (statsRes.status === 'fulfilled') {
        const s = statsRes.value;
        setSummaryStats({
          totalPredictions: s.total_analyses || 0,
          totalWins: s.successful_bets || 0,
          totalLosses: (s.total_analyses || 0) - (s.successful_bets || 0),
          winRate: s.win_rate || 0,
          totalProfit: s.total_profit || 0,
          roi: s.roi || 0,
          avgOdds: 0,
        });
      }
    } catch (e) {
      console.error('Failed to fetch history:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory(currentPage);
  }, [currentPage, fetchHistory]);

  // Compute cumulative P/L from bets
  const cumulativeData = bets
    .filter(b => b.status !== 'pending')
    .reduce<{ idx: number; cumulative: number; match: string }[]>((acc, bet, i) => {
      const prev = acc.length > 0 ? acc[acc.length - 1].cumulative : 0;
      acc.push({ idx: i + 1, cumulative: Math.round((prev + bet.profit_loss) * 100) / 100, match: bet.match });
      return acc;
    }, []);

  const filteredBets = bets.filter(b => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return b.match.toLowerCase().includes(q) || b.league.toLowerCase().includes(q) || b.selection.toLowerCase().includes(q);
  });

  return (
    <PageLayout
      title="Prediction History"
      description="Complete history of all predictions and results"
      breadcrumbs={[{ label: 'History' }]}
      actions={
        <Button variant="outline" className="border-white/10 text-gray-400" disabled>
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      }
    >
      <div className="space-y-6">
        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading history...
          </div>
        )}

        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Bets</div>
              <div className="text-2xl font-bold text-white">{summaryStats.totalPredictions}</div>
              <div className="text-xs text-[#00ff88] mt-1">{summaryStats.winRate}% win rate</div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Profit</div>
              <div className={cn(
                'text-2xl font-bold',
                summaryStats.totalProfit >= 0 ? 'text-[#00ff88]' : 'text-[#ff3860]'
              )}>
                {summaryStats.totalProfit >= 0 ? '+' : ''}{summaryStats.totalProfit.toFixed(2)}
              </div>
              <div className="text-xs text-[#00ff88] mt-1">{summaryStats.roi}% ROI</div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Win / Loss</div>
              <div className="text-2xl font-bold text-white">{summaryStats.totalWins} / {summaryStats.totalLosses}</div>
              {summaryStats.totalPredictions > 0 && (
                <div className="flex gap-1 mt-1">
                  <div className="h-1.5 bg-[#00ff88] rounded-l" style={{ width: `${summaryStats.winRate}%` }} />
                  <div className="h-1.5 bg-[#ff3860] rounded-r" style={{ width: `${100 - summaryStats.winRate}%` }} />
                </div>
              )}
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="text-xs text-gray-500 mb-1">Total Records</div>
              <div className="text-2xl font-bold text-white">{totalBets}</div>
              <div className="text-xs text-gray-400 mt-1">In database</div>
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
            </div>

            {/* History Table */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardContent className="p-0">
                {filteredBets.length > 0 ? (
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
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredBets.map((bet) => (
                        <TableRow key={bet.id} className="border-white/5 hover:bg-white/5">
                          <TableCell className="text-gray-400 text-sm">
                            {bet.created_at ? new Date(bet.created_at).toLocaleDateString() : '-'}
                          </TableCell>
                          <TableCell className="text-gray-400 text-sm">{bet.league}</TableCell>
                          <TableCell className="text-white">{bet.match}</TableCell>
                          <TableCell className="text-[#00d4ff]">{bet.selection}</TableCell>
                          <TableCell className="font-mono text-white">{bet.odds.toFixed(2)}</TableCell>
                          <TableCell className="text-white">{bet.stake.toFixed(0)}</TableCell>
                          <TableCell>
                            <Badge className={cn(
                              'border-0',
                              bet.status === 'won' ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                              bet.status === 'lost' ? 'bg-[#ff3860]/20 text-[#ff3860]' :
                              'bg-[#ff9500]/20 text-[#ff9500]'
                            )}>
                              {bet.status === 'won' ? <CheckCircle2 className="w-3 h-3 mr-1" /> :
                               bet.status === 'lost' ? <XCircle className="w-3 h-3 mr-1" /> : null}
                              {bet.status.toUpperCase()}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <span className={cn(
                              'font-medium',
                              bet.profit_loss > 0 ? 'text-[#00ff88]' : bet.profit_loss < 0 ? 'text-[#ff3860]' : 'text-gray-400'
                            )}>
                              {bet.profit_loss > 0 ? '+' : ''}{bet.profit_loss.toFixed(2)}
                            </span>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="p-12 text-center text-gray-500">
                    <Calendar className="w-10 h-10 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">No bet history yet.</p>
                    <p className="text-xs mt-1">Run analyses and place bets to see history here.</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Pagination */}
            {totalBets > PAGE_SIZE && (
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-500">
                  Showing {(currentPage - 1) * PAGE_SIZE + 1} to {Math.min(currentPage * PAGE_SIZE, totalBets)} of {totalBets}
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
            )}
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="mt-6 space-y-6">
            {cumulativeData.length > 0 ? (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Cumulative Profit/Loss</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={cumulativeData}>
                        <defs>
                          <linearGradient id="colorCumulative" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="idx" stroke="#64748b" fontSize={12} />
                        <YAxis stroke="#64748b" fontSize={12} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                          }}
                          formatter={(value: number) => [`${value}`, 'Cumulative P/L']}
                        />
                        <Area type="monotone" dataKey="cumulative" stroke="#00ff88" fillOpacity={1} fill="url(#colorCumulative)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center text-gray-500">
                  <TrendingUp className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No analytics data yet.</p>
                  <p className="text-xs mt-1">Charts will populate after bets are settled.</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default HistoryPage;
