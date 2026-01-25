// pages/ReportsPage.tsx
/**
 * Reports Page - Historical analysis reports and performance tracking
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  FileText,
  Download,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Target,
  Trophy,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import { PerformanceCharts } from '@/components/PerformanceCharts';
import { BetHistory } from '@/components/BetHistory';

// Sample data for PerformanceCharts
const sampleDailyData = Array.from({ length: 30 }, (_, i) => {
  const bets = Math.floor(Math.random() * 10) + 1;
  const wins = Math.floor(Math.random() * bets);
  return {
    date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    bets,
    wins,
    losses: bets - wins,
    profit: Math.round((Math.random() * 200 - 50) * 100) / 100,
    roi: Math.round((Math.random() * 20 - 5) * 100) / 100,
    avgEdge: Math.round(Math.random() * 10 * 100) / 100 / 100,
    avgQuality: Math.round((Math.random() * 30 + 60) * 100) / 100 / 100,
  };
});

const sampleSportData = [
  { sport: 'Tennis', bets: 45, wins: 28, winRate: 62.2, profit: 234.50, roi: 12.5, color: '#8b5cf6' },
  { sport: 'Basketball', bets: 30, wins: 17, winRate: 56.7, profit: 156.20, roi: 8.3, color: '#3b82f6' },
  { sport: 'Handball', bets: 15, wins: 9, winRate: 60.0, profit: 78.30, roi: 15.2, color: '#22c55e' },
  { sport: 'Table Tennis', bets: 10, wins: 5, winRate: 50.0, profit: 45.00, roi: 4.5, color: '#f59e0b' },
];

const sampleStats = {
  totalBets: 100,
  totalWins: 59,
  winRate: 59.0,
  totalProfit: 514.00,
  roi: 10.3,
  avgEdge: 0.052,
  avgQuality: 0.72,
  bestStreak: 8,
  currentStreak: 3,
  sharpeRatio: 1.45,
  maxDrawdown: 0.085,
};

// Sample data for BetHistory
const sampleBetHistory = Array.from({ length: 100 }, (_, i) => {
  const odds = Math.round((Math.random() * 2 + 1.5) * 100) / 100;
  const stake = Math.round((Math.random() * 50 + 10) * 100) / 100;
  const status = ['pending', 'won', 'lost', 'void'][Math.floor(Math.random() * 4)] as 'pending' | 'won' | 'lost' | 'void';
  const profit = status === 'won' ? Math.round((stake * (odds - 1)) * 100) / 100 : status === 'lost' ? -stake : 0;
  return {
    id: `bet-${i + 1}`,
    date: new Date(Date.now() - i * 8 * 60 * 60 * 1000).toISOString().split('T')[0],
    sport: ['Tennis', 'Basketball', 'Handball', 'Table Tennis'][i % 4],
    match: ['Djokovic vs Nadal', 'Lakers vs Celtics', 'Barcelona vs Kiel', 'Dortmund vs Bayern', 'Federer vs Alcaraz'][i % 5],
    selection: ['ML Home', 'Over 2.5', 'Handicap -1.5', 'Under 185.5', 'ML Away'][i % 5],
    odds,
    stake,
    potentialWin: Math.round(stake * odds * 100) / 100,
    status,
    profit,
    edge: Math.round(Math.random() * 10) / 100,
    quality: Math.floor(Math.random() * 40) + 60,
    confidence: Math.round((Math.random() * 30 + 60)) / 100,
    bookmaker: ['Fortuna', 'STS', 'Betclic', 'Bet365'][i % 4],
  };
});

const reportTemplates = [
  { id: 'daily', name: 'Raport dzienny', description: 'Podsumowanie wyników z ostatnich 24h' },
  { id: 'weekly', name: 'Raport tygodniowy', description: 'Analiza wydajności z ostatniego tygodnia' },
  { id: 'monthly', name: 'Raport miesięczny', description: 'Szczegółowe statystyki z ostatniego miesiąca' },
  { id: 'sport', name: 'Raport sportowy', description: 'Wyniki w podziale na sporty' },
  { id: 'roi', name: 'Analiza ROI', description: 'Zwrot z inwestycji w czasie' },
];

export function ReportsPage() {
  const [selectedPeriod, setSelectedPeriod] = useState('30d');
  const [selectedReport, setSelectedReport] = useState<string | null>(null);

  const summaryStats = {
    totalBets: sampleBetHistory.length,
    wonBets: sampleBetHistory.filter(b => b.status === 'won').length,
    lostBets: sampleBetHistory.filter(b => b.status === 'lost').length,
    pendingBets: sampleBetHistory.filter(b => b.status === 'pending').length,
    totalProfit: sampleBetHistory.reduce((sum, b) => sum + b.profit, 0),
    winRate: (sampleBetHistory.filter(b => b.status === 'won').length /
              sampleBetHistory.filter(b => b.status !== 'pending' && b.status !== 'void').length * 100) || 0,
    avgEdge: sampleBetHistory.reduce((sum, b) => sum + b.edge, 0) / sampleBetHistory.length,
    avgQuality: sampleBetHistory.reduce((sum, b) => sum + b.quality, 0) / sampleBetHistory.length,
  };

  const generateReport = (reportId: string) => {
    setSelectedReport(reportId);
    // In production, this would generate a PDF or detailed report
    console.log('Generating report:', reportId);
  };

  return (
    <div className="min-h-screen bg-background pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              Raporty
            </h1>
            <p className="text-gray-400 mt-1">
              Szczegółowe raporty i analizy wydajności
            </p>
          </div>

          <div className="flex items-center gap-4">
            <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
              <SelectTrigger className="w-40 bg-white/5 border-white/10 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="7d">Ostatnie 7 dni</SelectItem>
                <SelectItem value="30d">Ostatnie 30 dni</SelectItem>
                <SelectItem value="90d">Ostatnie 90 dni</SelectItem>
                <SelectItem value="1y">Ostatni rok</SelectItem>
                <SelectItem value="all">Wszystko</SelectItem>
              </SelectContent>
            </Select>
            <Button className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:opacity-90">
              <Download className="w-4 h-4 mr-2" />
              Eksportuj
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Całkowity zysk</p>
                  <p className={`text-2xl font-bold ${summaryStats.totalProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {summaryStats.totalProfit >= 0 ? '+' : ''}{summaryStats.totalProfit.toFixed(2)} PLN
                  </p>
                </div>
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${summaryStats.totalProfit >= 0 ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                  {summaryStats.totalProfit >= 0 ? (
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-red-400" />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Win Rate</p>
                  <p className="text-2xl font-bold text-white">{summaryStats.winRate.toFixed(1)}%</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Target className="w-5 h-5 text-blue-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Zakłady</p>
                  <p className="text-2xl font-bold text-white">{summaryStats.totalBets}</p>
                  <p className="text-xs text-gray-500">
                    {summaryStats.wonBets}W / {summaryStats.lostBets}L / {summaryStats.pendingBets}P
                  </p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-violet-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Avg Edge</p>
                  <p className="text-2xl font-bold text-white">+{summaryStats.avgEdge.toFixed(1)}%</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <Trophy className="w-5 h-5 text-yellow-400" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="performance" className="space-y-6">
          <TabsList className="bg-glass-card border border-white/10 p-1">
            <TabsTrigger value="performance" className="data-[state=active]:bg-emerald-500/20">
              <BarChart3 className="w-4 h-4 mr-2" />
              Wyniki
            </TabsTrigger>
            <TabsTrigger value="history" className="data-[state=active]:bg-emerald-500/20">
              <Clock className="w-4 h-4 mr-2" />
              Historia
            </TabsTrigger>
            <TabsTrigger value="templates" className="data-[state=active]:bg-emerald-500/20">
              <FileText className="w-4 h-4 mr-2" />
              Szablony
            </TabsTrigger>
          </TabsList>

          {/* Performance Tab */}
          <TabsContent value="performance">
            <PerformanceCharts
              dailyData={sampleDailyData}
              sportData={sampleSportData}
              stats={sampleStats}
              period={selectedPeriod as '7d' | '30d' | '90d' | 'all'}
              onPeriodChange={(p) => setSelectedPeriod(p)}
            />
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history">
            <BetHistory bets={sampleBetHistory} />
          </TabsContent>

          {/* Templates Tab */}
          <TabsContent value="templates">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {reportTemplates.map((template) => (
                <Card
                  key={template.id}
                  className={`bg-glass-card border-white/5 cursor-pointer transition-all hover:border-emerald-500/30 ${
                    selectedReport === template.id ? 'border-emerald-500/50 bg-emerald-500/10' : ''
                  }`}
                  onClick={() => generateReport(template.id)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="w-12 h-12 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                        <FileText className="w-6 h-6 text-emerald-400" />
                      </div>
                      {selectedReport === template.id && (
                        <Badge className="bg-emerald-500/20 text-emerald-300">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Wybrany
                        </Badge>
                      )}
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">{template.name}</h3>
                    <p className="text-sm text-gray-400 mb-4">{template.description}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full bg-white/5 border-white/10 text-white hover:bg-white/10"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Generuj raport
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Report Preview */}
            {selectedReport && (
              <Card className="bg-glass-card border-white/5 mt-6">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <FileText className="w-5 h-5 text-emerald-400" />
                    Podgląd raportu: {reportTemplates.find(t => t.id === selectedReport)?.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-gray-400">Podsumowanie</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Okres:</span>
                          <span className="text-white">{selectedPeriod === '30d' ? '30 dni' : selectedPeriod}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Zakłady:</span>
                          <span className="text-white">{summaryStats.totalBets}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Win Rate:</span>
                          <span className="text-white">{summaryStats.winRate.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-gray-400">Wyniki</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Zysk:</span>
                          <span className={summaryStats.totalProfit >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {summaryStats.totalProfit >= 0 ? '+' : ''}{summaryStats.totalProfit.toFixed(2)} PLN
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">ROI:</span>
                          <span className="text-white">+12.5%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Avg Edge:</span>
                          <span className="text-white">+{summaryStats.avgEdge.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-gray-400">Ryzyko</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Max Drawdown:</span>
                          <span className="text-red-400">-8.5%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Sharpe Ratio:</span>
                          <span className="text-white">1.45</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Risk of Ruin:</span>
                          <span className="text-white">2.1%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-6 pt-6 border-t border-white/10 flex justify-end gap-4">
                    <Button
                      variant="outline"
                      className="bg-white/5 border-white/10 text-white"
                    >
                      Podgląd PDF
                    </Button>
                    <Button className="bg-gradient-to-r from-emerald-500 to-teal-600">
                      <Download className="w-4 h-4 mr-2" />
                      Pobierz raport
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default ReportsPage;
