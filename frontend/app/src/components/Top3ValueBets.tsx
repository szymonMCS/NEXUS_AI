// components/Top3ValueBets.tsx
/**
 * Top 3 Value Bets component with Gold/Silver/Bronze styling
 */

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Trophy, Star, Bookmark, Share2, ExternalLink } from 'lucide-react';
import { formatEdge, getQualityColor } from '@/lib/api';
import type { ValueBet } from '@/lib/api';

interface Top3ValueBetsProps {
  bets: ValueBet[];
  onViewDetails?: (bet: ValueBet) => void;
  onSave?: (bet: ValueBet) => void;
  onShare?: (bet: ValueBet) => void;
}

const rankStyles = {
  1: {
    gradient: 'from-yellow-500/30 via-yellow-400/20 to-amber-500/30',
    border: 'border-yellow-500/50',
    icon: 'text-yellow-400',
    badge: 'bg-yellow-500/20 text-yellow-300',
    glow: 'shadow-yellow-500/20',
    label: 'GOLD',
  },
  2: {
    gradient: 'from-gray-400/30 via-gray-300/20 to-slate-400/30',
    border: 'border-gray-400/50',
    icon: 'text-gray-300',
    badge: 'bg-gray-400/20 text-gray-300',
    glow: 'shadow-gray-400/20',
    label: 'SILVER',
  },
  3: {
    gradient: 'from-amber-700/30 via-orange-600/20 to-amber-800/30',
    border: 'border-amber-600/50',
    icon: 'text-amber-500',
    badge: 'bg-amber-600/20 text-amber-400',
    glow: 'shadow-amber-500/20',
    label: 'BRONZE',
  },
};

export function Top3ValueBets({ bets, onViewDetails, onSave, onShare }: Top3ValueBetsProps) {
  const top3 = bets.slice(0, 3);

  if (top3.length === 0) {
    return (
      <div className="text-center py-12">
        <Trophy className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-white mb-2">Brak Top 3 Value Bets</h3>
        <p className="text-gray-400">Uruchom analizę, aby znaleźć najlepsze zakłady</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-yellow-500/20 to-amber-500/20 flex items-center justify-center">
          <Trophy className="w-6 h-6 text-yellow-400" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Top 3 Value Bets</h3>
          <p className="text-sm text-gray-400">Najlepsze zakłady według naszego AI</p>
        </div>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {top3.map((bet, index) => {
          const rank = (bet.rank || index + 1) as 1 | 2 | 3;
          const style = rankStyles[rank] || rankStyles[3];

          return (
            <Card
              key={`${bet.match}-${bet.rank}`}
              className={`relative overflow-hidden transition-all duration-500 hover:scale-[1.03] bg-gradient-to-br ${style.gradient} ${style.border} shadow-lg ${style.glow}`}
            >
              {/* Rank Badge */}
              <div className={`absolute top-3 right-3 ${style.badge} px-2 py-1 rounded-full text-xs font-bold flex items-center gap-1`}>
                <Star className={`w-3 h-3 ${style.icon}`} />
                {style.label}
              </div>

              {/* Animated Glow Effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full animate-shimmer" />

              <CardContent className="p-6 relative">
                {/* Rank Number */}
                <div className={`text-6xl font-black ${style.icon} opacity-20 absolute top-2 left-4`}>
                  #{rank}
                </div>

                {/* Match Info */}
                <div className="pt-8 mb-4">
                  <Badge variant="secondary" className="mb-2 bg-white/10 text-gray-300">
                    {bet.league}
                  </Badge>
                  <h4 className="text-lg font-bold text-white mb-1">{bet.match}</h4>
                  <div className="text-sm text-gray-300">
                    Typ: <span className="text-violet-400 font-semibold">{bet.selection.toUpperCase()}</span>
                  </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-black/20 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-400 mb-1">Kurs</div>
                    <div className="text-2xl font-bold text-white">{bet.odds.toFixed(2)}</div>
                    <div className="text-xs text-gray-500">{bet.bookmaker}</div>
                  </div>
                  <div className="bg-black/20 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-400 mb-1">Edge</div>
                    <div className="text-2xl font-bold text-green-400">{formatEdge(bet.edge)}</div>
                    <div className="text-xs text-gray-500">Value</div>
                  </div>
                </div>

                {/* Quality & Confidence */}
                <div className="flex items-center justify-between mb-4 text-sm">
                  <div>
                    <span className="text-gray-400">Jakość: </span>
                    <span className={`font-bold ${getQualityColor(bet.quality_score)}`}>
                      {bet.quality_score}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Pewność: </span>
                    <span className="font-bold text-blue-400">{(bet.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {/* Stake Recommendation */}
                <div className={`p-3 rounded-lg ${style.badge} mb-4`}>
                  <div className="text-xs opacity-80 mb-1">Rekomendacja stawki</div>
                  <div className="text-sm font-semibold">{bet.stake_recommendation}</div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  {onViewDetails && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 bg-white/5 border-white/10 text-white hover:bg-white/10"
                      onClick={() => onViewDetails(bet)}
                    >
                      <ExternalLink className="w-4 h-4 mr-1" />
                      Szczegóły
                    </Button>
                  )}
                  {onSave && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-white/70 hover:text-white hover:bg-white/10"
                      onClick={() => onSave(bet)}
                    >
                      <Bookmark className="w-4 h-4" />
                    </Button>
                  )}
                  {onShare && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-white/70 hover:text-white hover:bg-white/10"
                      onClick={() => onShare(bet)}
                    >
                      <Share2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

export default Top3ValueBets;
