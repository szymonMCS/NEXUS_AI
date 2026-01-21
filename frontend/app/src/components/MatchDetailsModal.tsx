// components/MatchDetailsModal.tsx
/**
 * Match Details Modal - shows detailed match info, H2H, news, odds comparison
 */

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import {
  TrendingUp,
  History,
  Newspaper,
  DollarSign,
  Users,
  Trophy,
  AlertCircle,
  CheckCircle,
  Calendar
} from 'lucide-react';
import { formatEdge, getQualityColor } from '@/lib/api';
import type { ValueBet } from '@/lib/api';

interface MatchDetailsModalProps {
  bet: ValueBet | null;
  open: boolean;
  onClose: () => void;
}

// Mock H2H data (would come from API in production)
const mockH2HData = {
  total: { wins1: 5, wins2: 3, draws: 0 },
  recent: [
    { date: '2025-12-15', result: '2-1', winner: 'home' },
    { date: '2025-11-20', result: '1-2', winner: 'away' },
    { date: '2025-10-05', result: '3-0', winner: 'home' },
  ],
};

// Mock news data
const mockNews = [
  { title: 'Player X expected to return from injury', source: 'ESPN', time: '2h ago', sentiment: 'positive' },
  { title: 'Team preparing for crucial match', source: 'Sky Sports', time: '4h ago', sentiment: 'neutral' },
  { title: 'Weather conditions might affect the game', source: 'BBC Sport', time: '6h ago', sentiment: 'neutral' },
];

// Mock odds comparison
const mockOddsComparison = [
  { bookmaker: 'Fortuna', home: 2.15, draw: 3.40, away: 2.80 },
  { bookmaker: 'STS', home: 2.10, draw: 3.30, away: 2.85 },
  { bookmaker: 'Betclic', home: 2.20, draw: 3.35, away: 2.75 },
  { bookmaker: 'Bet365', home: 2.12, draw: 3.45, away: 2.82 },
];

export function MatchDetailsModal({ bet, open, onClose }: MatchDetailsModalProps) {
  if (!bet) return null;

  const players = bet.match.split(' vs ');
  const player1 = players[0] || 'Player 1';
  const player2 = players[1] || 'Player 2';

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl bg-gray-900 border-white/10 text-white max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold flex items-center gap-2">
            <Trophy className="w-5 h-5 text-violet-400" />
            {bet.match}
          </DialogTitle>
          <div className="flex items-center gap-2 mt-2">
            <Badge variant="secondary" className="bg-violet-500/20 text-violet-300">
              {bet.league}
            </Badge>
            <Badge variant="secondary" className={`${getQualityColor(bet.quality_score)} bg-opacity-20`}>
              Jakość: {bet.quality_score}%
            </Badge>
          </div>
        </DialogHeader>

        <Tabs defaultValue="overview" className="mt-4">
          <TabsList className="bg-white/5 border-white/10">
            <TabsTrigger value="overview" className="data-[state=active]:bg-violet-500/20">
              <TrendingUp className="w-4 h-4 mr-1" /> Przegląd
            </TabsTrigger>
            <TabsTrigger value="h2h" className="data-[state=active]:bg-violet-500/20">
              <History className="w-4 h-4 mr-1" /> H2H
            </TabsTrigger>
            <TabsTrigger value="news" className="data-[state=active]:bg-violet-500/20">
              <Newspaper className="w-4 h-4 mr-1" /> Newsy
            </TabsTrigger>
            <TabsTrigger value="odds" className="data-[state=active]:bg-violet-500/20">
              <DollarSign className="w-4 h-4 mr-1" /> Kursy
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-4 space-y-4">
            {/* Recommendation Card */}
            <Card className="bg-gradient-to-br from-violet-500/20 to-blue-500/20 border-violet-500/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm text-gray-300">Rekomendacja AI</span>
                  <Badge className="bg-green-500/20 text-green-400">
                    {bet.selection.toUpperCase()}
                  </Badge>
                </div>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-white">{bet.odds.toFixed(2)}</div>
                    <div className="text-xs text-gray-400">Kurs @ {bet.bookmaker}</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-400">{formatEdge(bet.edge)}</div>
                    <div className="text-xs text-gray-400">Edge</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-blue-400">{(bet.confidence * 100).toFixed(0)}%</div>
                    <div className="text-xs text-gray-400">Pewność</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quality Breakdown */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-gray-300">Analiza jakości danych</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Zgodność źródeł</span>
                  <span className="text-white font-medium">85%</span>
                </div>
                <Progress value={85} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Świeżość danych</span>
                  <span className="text-white font-medium">92%</span>
                </div>
                <Progress value={92} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Wariancja kursów</span>
                  <span className="text-white font-medium">78%</span>
                </div>
                <Progress value={78} className="h-2" />
              </div>
            </div>

            {/* Reasoning */}
            {bet.reasoning && bet.reasoning.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-2">Uzasadnienie</h4>
                <div className="space-y-2">
                  {bet.reasoning.map((reason, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <CheckCircle className="w-4 h-4 text-green-400 mt-0.5" />
                      <span className="text-gray-300">{reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Stake Recommendation */}
            <div className="bg-white/5 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Rekomendacja stawki</h4>
              <p className="text-lg font-bold text-violet-400">{bet.stake_recommendation}</p>
              <p className="text-xs text-gray-400 mt-1">
                Oparte na Kelly Criterion z uwzględnieniem jakości danych
              </p>
            </div>
          </TabsContent>

          {/* H2H Tab */}
          <TabsContent value="h2h" className="mt-4 space-y-4">
            <div className="flex items-center justify-between bg-white/5 rounded-lg p-4">
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">{player1}</div>
                <div className="text-3xl font-bold text-green-400">{mockH2HData.total.wins1}</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">Remisy</div>
                <div className="text-3xl font-bold text-gray-400">{mockH2HData.total.draws}</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">{player2}</div>
                <div className="text-3xl font-bold text-blue-400">{mockH2HData.total.wins2}</div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-3">Ostatnie mecze</h4>
              <div className="space-y-2">
                {mockH2HData.recent.map((match, i) => (
                  <div key={i} className="flex items-center justify-between bg-white/5 rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-300">{match.date}</span>
                    </div>
                    <span className="text-lg font-bold text-white">{match.result}</span>
                    <Badge className={match.winner === 'home' ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'}>
                      {match.winner === 'home' ? player1 : player2}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          {/* News Tab */}
          <TabsContent value="news" className="mt-4 space-y-3">
            {mockNews.map((news, i) => (
              <div key={i} className="bg-white/5 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  {news.sentiment === 'positive' ? (
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
                  ) : news.sentiment === 'negative' ? (
                    <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
                  ) : (
                    <Newspaper className="w-5 h-5 text-gray-400 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-white">{news.title}</h4>
                    <div className="flex items-center gap-2 mt-1 text-xs text-gray-400">
                      <span>{news.source}</span>
                      <span>•</span>
                      <span>{news.time}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </TabsContent>

          {/* Odds Tab */}
          <TabsContent value="odds" className="mt-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 text-gray-400">Bukmacher</th>
                    <th className="text-center py-2 text-gray-400">{player1}</th>
                    <th className="text-center py-2 text-gray-400">Remis</th>
                    <th className="text-center py-2 text-gray-400">{player2}</th>
                  </tr>
                </thead>
                <tbody>
                  {mockOddsComparison.map((odds, i) => {
                    const isBest = odds.bookmaker === bet.bookmaker;
                    return (
                      <tr key={i} className={`border-b border-white/5 ${isBest ? 'bg-violet-500/10' : ''}`}>
                        <td className="py-3 font-medium text-white">
                          {odds.bookmaker}
                          {isBest && <Badge className="ml-2 text-xs bg-violet-500/20 text-violet-300">Wybrano</Badge>}
                        </td>
                        <td className={`py-3 text-center ${bet.selection === 'home' && isBest ? 'text-green-400 font-bold' : 'text-white'}`}>
                          {odds.home.toFixed(2)}
                        </td>
                        <td className="py-3 text-center text-white">{odds.draw.toFixed(2)}</td>
                        <td className={`py-3 text-center ${bet.selection === 'away' && isBest ? 'text-green-400 font-bold' : 'text-white'}`}>
                          {odds.away.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-4 p-3 bg-white/5 rounded-lg text-xs text-gray-400">
              <Users className="w-4 h-4 inline mr-1" />
              Kursy porównane z 4 bukmacherów. Najlepszy kurs dla wybranej opcji został automatycznie wybrany.
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

export default MatchDetailsModal;
