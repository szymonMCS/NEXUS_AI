/**
 * ExtendedMarkets - Sport-specific alternative betting markets
 * 
 * Football:
 * - Cards (Over/Under total cards, Asian cards handicap)
 * - Fouls (Over/Under, handicap)
 * - Goal difference (win by 1, 2+, exactly 1, etc.)
 * 
 * Basketball:
 * - Fouls (team/personal fouls)
 * - Win margin (1-5, 6-10, 11+, etc.)
 * - Alternative handicaps
 * 
 * Tennis:
 * - Set betting (2-0, 2-1, etc.)
 * - Total games over/under
 * - Game handicaps
 */

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  AlertTriangle, 
  Target,
  Shield,
  Activity,
  Swords,
  CircleOff,
  Timer
} from 'lucide-react';

// ==================== FOOTBALL MARKETS ====================

interface CardMarket {
  type: 'asian_cards' | 'total_cards' | 'team_cards';
  line: number | string;
  overOdds?: number;
  underOdds?: number;
  homeOdds?: number;
  awayOdds?: number;
  overProb?: number;
  underProb?: number;
  homeProb?: number;
  awayProb?: number;
  recommendation?: string;
  expectedCards?: number;  // Calculated from match intensity
}

interface FoulMarket {
  type: 'total_fouls' | 'foul_handicap' | 'team_fouls';
  line: number;
  overOdds?: number;
  underOdds?: number;
  homeOdds?: number;
  awayOdds?: number;
  overProb?: number;
  underProb?: number;
  expectedFouls?: number;
}

interface GoalDifferenceMarket {
  type: 'exact_margin' | 'win_by' | 'any_win';
  selection: string;  // "1 goal", "2+ goals", "exactly 1", etc.
  odds: number;
  probability?: number;
  edge?: number;
}

// ==================== BASKETBALL MARKETS ====================



interface WinMarginMarket {
  range: string;  // "1-5", "6-10", "11+", "1-2", "3-6", etc.
  homeOdds?: number;
  awayOdds?: number;
  homeProb?: number;
  awayProb?: number;
  recommendation?: string;
}

// ==================== TENNIS MARKETS ====================

interface SetBettingMarket {
  score: string;  // "2-0", "2-1", "0-2", "1-2", "3-0", "3-1", "3-2" etc.
  homeOdds: number;
  awayOdds: number;
  homeProb?: number;
  awayProb?: number;
  edge?: number;
}

interface GameHandicapMarket {
  line: number;  // e.g., -3.5, +3.5
  homeOdds: number;
  awayOdds: number;
  homeProb?: number;
  awayProb?: number;
  recommendation?: string;
}

// ==================== COMPONENT PROPS ====================

interface ExtendedMarketsProps {
  sport: 'football' | 'basketball' | 'tennis';
  cardMarkets?: CardMarket[];
  foulMarkets?: FoulMarket[];
  goalDiffMarkets?: GoalDifferenceMarket[];
  winMarginMarkets?: WinMarginMarket[];
  setBettingMarkets?: SetBettingMarket[];
  gameHandicapMarkets?: GameHandicapMarket[];
  className?: string;
}

export function ExtendedMarkets({
  sport,
  cardMarkets,
  foulMarkets,
  goalDiffMarkets,
  winMarginMarkets,
  setBettingMarkets,
  gameHandicapMarkets,
  className
}: ExtendedMarketsProps) {
  


  const getValueBadge = (edge?: number) => {
    if (!edge || edge < 0.05) return null;
    return (
      <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-[10px]">
        <TrendingUp className="w-3 h-3 mr-1" />
        +{(edge * 100).toFixed(0)}%
      </Badge>
    );
  };

  // ==================== FOOTBALL SECTIONS ====================
  const renderFootballMarkets = () => (
    <div className="space-y-6">
      {/* Cards Section */}
      {cardMarkets && cardMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-[#ff9500]" />
            Cards Markets
          </h4>
          
          <div className="space-y-3">
            {cardMarkets.map((market, idx) => (
              <div key={idx} className="p-3 bg-white/5 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-medium">
                      {market.type === 'total_cards' ? 'Total Cards' : 
                       market.type === 'team_cards' ? 'Team Cards AH' : 'Asian Cards'}
                    </span>
                    <Badge variant="outline" className="border-white/20 text-gray-400 text-xs">
                      Line {market.line}
                    </Badge>
                  </div>
                  {market.expectedCards && (
                    <span className="text-xs text-gray-500">
                      Exp: {market.expectedCards}
                    </span>
                  )}
                </div>
                
                {market.type === 'total_cards' ? (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Over</span>
                      <div className="flex items-center gap-2">
                        <span className="text-[#00d4ff] font-bold font-mono">{market.overOdds?.toFixed(2)}</span>
                        {market.overProb && (
                          <span className="text-xs text-gray-500">{market.overProb.toFixed(0)}%</span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Under</span>
                      <div className="flex items-center gap-2">
                        <span className="text-[#00ff88] font-bold font-mono">{market.underOdds?.toFixed(2)}</span>
                        {market.underProb && (
                          <span className="text-xs text-gray-500">{market.underProb.toFixed(0)}%</span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Home</span>
                      <span className="text-[#00d4ff] font-bold font-mono">{market.homeOdds?.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Away</span>
                      <span className="text-[#00ff88] font-bold font-mono">{market.awayOdds?.toFixed(2)}</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fouls Section */}
      {foulMarkets && foulMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <Shield className="w-4 h-4 text-[#00d4ff]" />
            Fouls Markets
          </h4>
          
          <div className="space-y-2">
            {foulMarkets.map((market, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-300">
                    {market.type === 'total_fouls' ? 'Total Fouls' : 
                     market.type === 'foul_handicap' ? 'Fouls Handicap' : 'Team Fouls'}
                  </span>
                  <Badge variant="outline" className="border-white/20 text-gray-400 text-xs">
                    {market.line}
                  </Badge>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Over</div>
                    <div className="text-[#00d4ff] font-bold font-mono">{market.overOdds?.toFixed(2)}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Under</div>
                    <div className="text-[#00ff88] font-bold font-mono">{market.underOdds?.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Goal Difference / Win Margin */}
      {goalDiffMarkets && goalDiffMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <Target className="w-4 h-4 text-[#b829dd]" />
            Goal Difference / Win Margin
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {goalDiffMarkets.map((market, idx) => (
              <div 
                key={idx} 
                className={cn(
                  "p-3 rounded-lg border text-center",
                  market.edge && market.edge > 0.05 
                    ? "border-[#00ff88]/30 bg-[#00ff88]/10" 
                    : "border-white/10 bg-white/5"
                )}
              >
                <div className="text-sm text-gray-300 mb-1">{market.selection}</div>
                <div className="text-xl font-bold text-white font-mono">{market.odds.toFixed(2)}</div>
                {market.probability && (
                  <div className="text-xs text-gray-500 mt-1">{market.probability.toFixed(0)}%</div>
                )}
                {getValueBadge(market.edge)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  // ==================== BASKETBALL SECTIONS ====================
  const renderBasketballMarkets = () => (
    <div className="space-y-6">
      {/* Win Margin */}
      {winMarginMarkets && winMarginMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#ff9500]" />
            Win Margin
          </h4>
          
          <div className="space-y-2">
            {winMarginMarkets.map((market, idx) => (
              <div key={idx} className="p-3 bg-white/5 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="outline" className="border-white/20 text-white">
                    {market.range} points
                  </Badge>
                  {market.recommendation && (
                    <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-xs">
                      Pick: {market.recommendation}
                    </Badge>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-2 bg-white/5 rounded">
                    <div className="text-xs text-gray-500 mb-1">Home by {market.range}</div>
                    <div className="text-lg font-bold text-[#00d4ff] font-mono">
                      {market.homeOdds?.toFixed(2)}
                    </div>
                    {market.homeProb && (
                      <div className="text-xs text-gray-500">{market.homeProb.toFixed(0)}%</div>
                    )}
                  </div>
                  <div className="text-center p-2 bg-white/5 rounded">
                    <div className="text-xs text-gray-500 mb-1">Away by {market.range}</div>
                    <div className="text-lg font-bold text-[#00ff88] font-mono">
                      {market.awayOdds?.toFixed(2)}
                    </div>
                    {market.awayProb && (
                      <div className="text-xs text-gray-500">{market.awayProb.toFixed(0)}%</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fouls - Basketball */}
      {foulMarkets && foulMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <CircleOff className="w-4 h-4 text-[#ff3860]" />
            Team Fouls
          </h4>
          
          <div className="grid grid-cols-2 gap-3">
            {foulMarkets.map((market, idx) => (
              <div key={idx} className="p-3 bg-white/5 rounded-lg text-center">
                <div className="text-xs text-gray-500 mb-2">{market.type === 'team_fouls' ? 'Team Total' : 'Total'}</div>
                <div className="text-sm text-gray-300 mb-2">Line {market.line}</div>
                <div className="flex items-center justify-center gap-3">
                  <div>
                    <div className="text-xs text-gray-500">Over</div>
                    <div className="text-[#00d4ff] font-bold">{market.overOdds?.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Under</div>
                    <div className="text-[#00ff88] font-bold">{market.underOdds?.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  // ==================== TENNIS SECTIONS ====================
  const renderTennisMarkets = () => (
    <div className="space-y-6">
      {/* Set Betting */}
      {setBettingMarkets && setBettingMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <Swords className="w-4 h-4 text-[#00d4ff]" />
            Correct Score (Sets)
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {setBettingMarkets.map((market, idx) => (
              <div 
                key={idx} 
                className={cn(
                  "p-3 rounded-lg border text-center",
                  market.edge && market.edge > 0.05 
                    ? "border-[#00ff88]/30 bg-[#00ff88]/10" 
                    : "border-white/10 bg-white/5"
                )}
              >
                <div className="text-lg font-bold text-white mb-2">{market.score}</div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">Home</span>
                    <span className="text-[#00d4ff] font-bold">{market.homeOdds.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">Away</span>
                    <span className="text-[#00ff88] font-bold">{market.awayOdds.toFixed(2)}</span>
                  </div>
                </div>
                {market.homeProb && (
                  <div className="mt-2 pt-2 border-t border-white/10">
                    <div className="text-[10px] text-gray-500">
                      Prob: {market.homeProb.toFixed(0)}% / {market.awayProb?.toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Game Handicap */}
      {gameHandicapMarkets && gameHandicapMarkets.length > 0 && (
        <div className="bg-[#0f1623]/60 border border-white/10 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <Timer className="w-4 h-4 text-[#ff9500]" />
            Game Handicap
          </h4>
          
          <div className="space-y-2">
            {gameHandicapMarkets.map((market, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="border-white/20 text-white">
                    {market.line > 0 ? `+${market.line}` : market.line} games
                  </Badge>
                  {market.recommendation && (
                    <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-xs">
                      Pick: {market.recommendation}
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Home</div>
                    <div className="text-[#00d4ff] font-bold font-mono">{market.homeOdds.toFixed(2)}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-500">Away</div>
                    <div className="text-[#00ff88] font-bold font-mono">{market.awayOdds.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className={cn("space-y-6", className)}>
      {sport === 'football' && renderFootballMarkets()}
      {sport === 'basketball' && renderBasketballMarkets()}
      {sport === 'tennis' && renderTennisMarkets()}
    </div>
  );
}

// Quick stats component for sidebar
export function ExtendedMarketStats({ 
  sport, 
  stats 
}: { 
  sport: 'football' | 'basketball' | 'tennis';
  stats: {
    expectedCards?: number;
    expectedFouls?: number;
    avgMargin?: number;
    likelySetScore?: string;
  };
}) {
  return (
    <div className="p-3 bg-white/5 rounded-lg">
      <h5 className="text-xs font-medium text-gray-400 mb-2 uppercase">
        {sport === 'football' ? 'Match Intensity' : 
         sport === 'basketball' ? 'Margin Stats' : 'Set Prediction'}
      </h5>
      <div className="space-y-1">
        {stats.expectedCards !== undefined && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Exp. Cards</span>
            <span className="text-[#ff9500] font-bold">{stats.expectedCards}</span>
          </div>
        )}
        {stats.expectedFouls !== undefined && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Exp. Fouls</span>
            <span className="text-[#00d4ff] font-bold">{stats.expectedFouls}</span>
          </div>
        )}
        {stats.avgMargin !== undefined && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Avg Margin</span>
            <span className="text-[#00ff88] font-bold">{stats.avgMargin} pts</span>
          </div>
        )}
        {stats.likelySetScore && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Likely Score</span>
            <span className="text-[#b829dd] font-bold">{stats.likelySetScore}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExtendedMarkets;
