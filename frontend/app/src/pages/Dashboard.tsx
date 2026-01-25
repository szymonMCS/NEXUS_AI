// pages/Dashboard.tsx
/**
 * Main Dashboard - Central hub for NEXUS AI
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  TrendingUp,
  Trophy,
  Activity,
  BarChart3,
  Clock,
  Zap,
  Target,
  RefreshCw,
  CheckCircle2
} from 'lucide-react';
import { Top3ValueBets } from '@/components/Top3ValueBets';
import { SportSelector } from '@/components/SportSelector';
import { MatchDetailsModal } from '@/components/MatchDetailsModal';
import api, { type ValueBet, type SportId, type SystemStats } from '@/lib/api';

export function Dashboard() {
  const [selectedSport, setSelectedSport] = useState<SportId>('tennis');
  const [valueBets, setValueBets] = useState<ValueBet[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedBet, setSelectedBet] = useState<ValueBet | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    loadData();
  }, [selectedSport]);

  const loadData = async () => {
    try {
      const [statsData, betsData] = await Promise.all([
        api.getStats().catch(() => null),
        api.getValueBets().catch(() => []),
      ]);

      if (statsData) setStats(statsData);
      setValueBets(betsData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    }
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      await api.runAnalysis({ sport: selectedSport });
      api.connectWebSocket((data) => {
        if (data.type === 'complete' && data.data) {
          setValueBets(data.data);
          setIsAnalyzing(false);
          setLastUpdate(new Date());
        }
      });
    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
    }
  };

  const handleViewDetails = (bet: ValueBet) => {
    setSelectedBet(bet);
    setModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-background pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              Dashboard
            </h1>
            <p className="text-gray-400 mt-1">
              Monitoruj wyniki i zarządzaj zakładami
            </p>
          </div>

          <div className="flex items-center gap-4">
            <SportSelector
              value={selectedSport}
              onChange={setSelectedSport}
              variant="compact"
            />
            <Button
              onClick={runAnalysis}
              disabled={isAnalyzing}
              className="bg-gradient-to-r from-violet-500 to-purple-600 hover:opacity-90"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Analizuję...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Nowa analiza
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Dzisiejszy zysk</p>
                  <p className={`text-2xl font-bold ${(stats?.total_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(stats?.total_profit || 0) >= 0 ? '+' : ''}{stats?.total_profit?.toFixed(2) || '0.00'}%
                  </p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-glass-card border-white/5">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Win Rate</p>
                  <p className="text-2xl font-bold text-white">
                    {stats?.win_rate?.toFixed(1) || '0.0'}%
                  </p>
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
                  <p className="text-sm text-gray-400">Aktywne zakłady</p>
                  <p className="text-2xl font-bold text-white">
                    {valueBets.length}
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
                  <p className="text-2xl font-bold text-white">
                    +{stats?.avg_edge?.toFixed(1) || '0.0'}%
                  </p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <Trophy className="w-5 h-5 text-yellow-400" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Status Bar */}
        <div className="flex items-center justify-between mb-6 p-3 bg-glass-card rounded-lg border border-white/5">
          <div className="flex items-center gap-4">
            {lastUpdate && (
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Clock className="w-4 h-4" />
                Ostatnia aktualizacja: {lastUpdate.toLocaleTimeString('pl-PL')}
              </div>
            )}
            {isAnalyzing && (
              <Badge className="bg-violet-500/20 text-violet-300 animate-pulse">
                <Activity className="w-3 h-3 mr-1" />
                Analiza w toku...
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Badge className="bg-green-500/20 text-green-300">
              <CheckCircle2 className="w-3 h-3 mr-1" />
              API Online
            </Badge>
          </div>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="bg-glass-card border border-white/10 p-1">
            <TabsTrigger value="overview" className="data-[state=active]:bg-violet-500/20">
              Przegląd
            </TabsTrigger>
            <TabsTrigger value="bets" className="data-[state=active]:bg-violet-500/20">
              Value Bets
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <Card className="bg-glass-card border-white/5">
              <CardContent className="p-6">
                <Top3ValueBets
                  bets={valueBets}
                  onViewDetails={handleViewDetails}
                  onSave={(bet) => console.log('Save bet:', bet)}
                  onShare={(bet) => console.log('Share bet:', bet)}
                />
              </CardContent>
            </Card>

            {/* Sport Stats */}
            <Card className="bg-glass-card border-white/5">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  Statystyki sportów
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { sport: 'Tennis', value: 45, profit: 234.50 },
                    { sport: 'Basketball', value: 30, profit: 156.20 },
                    { sport: 'Handball', value: 15, profit: 78.30 },
                    { sport: 'Table Tennis', value: 10, profit: 45.00 },
                  ].map((sport) => (
                    <div key={sport.sport} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-full bg-violet-500" />
                        <span className="text-white">{sport.sport}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-gray-400">{sport.value}%</span>
                        <span className={sport.profit >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {sport.profit >= 0 ? '+' : ''}{sport.profit.toFixed(2)} PLN
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Value Bets Tab */}
          <TabsContent value="bets" className="space-y-6">
            {valueBets.length === 0 ? (
              <Card className="bg-glass-card border-white/5">
                <CardContent className="p-12 text-center">
                  <Trophy className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">Brak value bets</h3>
                  <p className="text-gray-400 mb-6">
                    Uruchom nową analizę, aby zobaczyć dostępne zakłady.
                  </p>
                  <Button
                    onClick={runAnalysis}
                    disabled={isAnalyzing}
                    className="bg-gradient-to-r from-violet-500 to-purple-600"
                  >
                    <Zap className="w-4 h-4 mr-2" />
                    Uruchom analizę
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {valueBets.map((bet, index) => (
                  <Card
                    key={`${bet.match}-${index}`}
                    className="bg-glass-card border-white/5 hover:border-violet-500/30 transition-all cursor-pointer"
                    onClick={() => handleViewDetails(bet)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <Badge className="bg-violet-500/20 text-violet-300">
                          #{bet.rank || index + 1}
                        </Badge>
                        <Badge className={`${bet.quality_score >= 70 ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                          {bet.quality_score}% jakości
                        </Badge>
                      </div>
                      <h4 className="text-white font-semibold mb-1">{bet.match}</h4>
                      <p className="text-sm text-gray-400 mb-3">{bet.league}</p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-gray-400">Typ:</span>
                          <span className="text-white ml-1">{bet.selection}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Kurs:</span>
                          <span className="text-white ml-1">{bet.odds.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Edge:</span>
                          <span className="text-green-400 ml-1">+{(bet.edge * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Pewność:</span>
                          <span className="text-blue-400 ml-1">{(bet.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Match Details Modal */}
        <MatchDetailsModal
          bet={selectedBet}
          open={modalOpen}
          onClose={() => {
            setModalOpen(false);
            setSelectedBet(null);
          }}
        />
      </div>
    </div>
  );
}

export default Dashboard;
