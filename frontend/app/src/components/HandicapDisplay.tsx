/**
 * HandicapDisplay - Asian and European Handicap odds display
 * 
 * Shows:
 * - Asian Handicap lines (-1.5, -1, -0.5, 0, +0.5, +1, +1.5 etc.)
 * - European Handicap (1:0, 2:0, 0:1, 0:2 etc.)
 * - Probability bars for each handicap
 * - Value indicators
 */

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, Minus, AlertCircle, Scale } from 'lucide-react';

interface HandicapLine {
  line: number | string;  // e.g., -1.5, -1, "1:0" for European
  line_display?: string;  // Formatted display string
  type: 'asian' | 'european';
  homeOdds?: number;
  awayOdds?: number;
  drawOdds?: number;  // For European handicap
  homeProb?: number;
  awayProb?: number;
  drawProb?: number;
  homeEdge?: number;
  awayEdge?: number;
  drawEdge?: number;
  recommendation?: 'home' | 'away' | 'draw' | null;
}

interface HandicapDisplayProps {
  handicaps: HandicapLine[];
  sport: 'football' | 'basketball' | 'tennis';
  className?: string;
  showProbabilities?: boolean;
  showRecommendations?: boolean;
}

export function HandicapDisplay({
  handicaps,
  sport,
  className,
  showProbabilities = true,
  showRecommendations = true,
}: HandicapDisplayProps) {
  
  const getLineDescription = (line: number | string, type: 'asian' | 'european') => {
    if (type === 'european') {
      // European handicap format: "1:0", "2:0", "0:1", "0:2"
      const [homeAdv, awayAdv] = String(line).split(':').map(Number);
      if (homeAdv > 0) return `Start +${homeAdv} goals`;
      if (awayAdv > 0) return `Start -${awayAdv} goals`;
      return 'Level';
    }
    
    // Asian handicap
    const numLine = Number(line);
    if (numLine === 0) return 'Draw No Bet';
    if (numLine > 0) return `+${numLine} (${getAsianOutcome(numLine)})`;
    return `${numLine} (${getAsianOutcome(numLine)})`;
  };
  
  const getAsianOutcome = (line: number) => {
    // Explain what needs to happen for the bet to win
    if (line === -0.5) return 'Must win';
    if (line === 0) return 'Win or void';
    if (line === 0.5) return 'Win or draw';
    if (line === -1) return 'Win by 2+';
    if (line === 1) return 'Lose by 1/Win/Draw';
    if (line === -1.5) return 'Win by 2+';
    if (line === 1.5) return 'Lose by 1/Win/Draw';
    if (line === -2) return 'Win by 3+';
    if (line === 2) return 'Lose by 1-2/Win/Draw';
    if (line === -2.5) return 'Win by 3+';
    if (line === 2.5) return 'Lose by 1-2/Win/Draw';
    return '';
  };
  
  const getValueBadge = (edge?: number) => {
    if (!edge || edge < 0.05) return null;
    return (
      <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-[10px]">
        <TrendingUp className="w-3 h-3 mr-1" />
        +{(edge * 100).toFixed(0)}%
      </Badge>
    );
  };
  
  const getProbColor = (prob?: number) => {
    if (!prob) return 'bg-gray-600';
    if (prob >= 60) return 'bg-[#00ff88]';
    if (prob >= 40) return 'bg-[#00d4ff]';
    return 'bg-[#ff9500]';
  };

  return (
    <div className={cn("bg-[#0f1623]/80 border border-white/10 rounded-xl overflow-hidden", className)}>
      {/* Header */}
      <div className="px-4 py-3 bg-white/5 border-b border-white/10">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Scale className="w-4 h-4 text-[#00d4ff]" />
            Asian & European Handicaps
          </h3>
          <Badge variant="outline" className="border-white/20 text-gray-400 text-xs capitalize">
            {sport}
          </Badge>
        </div>
      </div>

      {/* Asian Handicap Section */}
      <div className="p-4 border-b border-white/10">
        <h4 className="text-xs font-medium uppercase tracking-wider text-gray-500 mb-3">
          Asian Handicap
        </h4>
        <div className="space-y-3">
          {handicaps
            .filter(h => h.type === 'asian')
            .map((handicap, idx) => (
              <div 
                key={`asian-${idx}`}
                className={cn(
                  "p-3 rounded-lg border transition-all",
                  handicap.recommendation 
                    ? "border-[#00ff88]/30 bg-[#00ff88]/5" 
                    : "border-white/5 bg-white/5 hover:border-white/10"
                )}
              >
                {/* Line Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-white font-mono">
                      {handicap.line_display || (handicap.type === 'asian' 
                        ? `AH ${Number(handicap.line) > 0 ? '+' : ''}${handicap.line}`
                        : `EH ${handicap.line}`)}
                    </span>
                    <span className="text-xs text-gray-500">
                      {getLineDescription(handicap.line, 'asian')}
                    </span>
                  </div>
                  {showRecommendations && handicap.recommendation && (
                    <Badge className="bg-[#00d4ff]/20 text-[#00d4ff] border-0">
                      Pick: {handicap.recommendation === 'home' ? 'Home' : 'Away'}
                    </Badge>
                  )}
                </div>

                {/* Odds and Probabilities */}
                <div className="grid grid-cols-2 gap-4">
                  {/* Home */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Home</span>
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-bold text-[#00d4ff] font-mono">
                          {handicap.homeOdds?.toFixed(2)}
                        </span>
                        {getValueBadge(handicap.homeEdge)}
                      </div>
                    </div>
                    {showProbabilities && handicap.homeProb && (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                            <div 
                              className={cn("h-full rounded-full transition-all", getProbColor(handicap.homeProb))}
                              style={{ width: `${handicap.homeProb}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono text-gray-400 w-10 text-right">
                            {handicap.homeProb.toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Away */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Away</span>
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-bold text-[#00ff88] font-mono">
                          {handicap.awayOdds?.toFixed(2)}
                        </span>
                        {getValueBadge(handicap.awayEdge)}
                      </div>
                    </div>
                    {showProbabilities && handicap.awayProb && (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                            <div 
                              className={cn("h-full rounded-full transition-all", getProbColor(handicap.awayProb))}
                              style={{ width: `${handicap.awayProb}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono text-gray-400 w-10 text-right">
                            {handicap.awayProb.toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* European Handicap Section */}
      <div className="p-4">
        <h4 className="text-xs font-medium uppercase tracking-wider text-gray-500 mb-3">
          European Handicap
        </h4>
        <div className="space-y-2">
          {handicaps
            .filter(h => h.type === 'european')
            .map((handicap, idx) => (
              <div 
                key={`euro-${idx}`}
                className="p-3 rounded-lg border border-white/5 bg-white/5 hover:border-white/10 transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-white">
                    {handicap.line_display || `EH ${handicap.line}`}
                  </span>
                  <span className="text-xs text-gray-500">
                    {getLineDescription(handicap.line, 'european')}
                  </span>
                </div>
                
                <div className="grid grid-cols-3 gap-2">
                  {/* 1 */}
                  <div className="text-center p-2 rounded bg-white/5">
                    <div className="text-xs text-gray-500 mb-1">1</div>
                    <div className="text-sm font-bold text-[#00d4ff] font-mono">
                      {handicap.homeOdds?.toFixed(2)}
                    </div>
                    {handicap.homeProb && (
                      <div className="text-[10px] text-gray-500 mt-1">
                        {handicap.homeProb.toFixed(0)}%
                      </div>
                    )}
                  </div>
                  
                  {/* X */}
                  <div className="text-center p-2 rounded bg-white/5">
                    <div className="text-xs text-gray-500 mb-1">X</div>
                    <div className="text-sm font-bold text-[#ff9500] font-mono">
                      {handicap.drawOdds?.toFixed(2)}
                    </div>
                    {handicap.drawProb && (
                      <div className="text-[10px] text-gray-500 mt-1">
                        {handicap.drawProb.toFixed(0)}%
                      </div>
                    )}
                  </div>
                  
                  {/* 2 */}
                  <div className="text-center p-2 rounded bg-white/5">
                    <div className="text-xs text-gray-500 mb-1">2</div>
                    <div className="text-sm font-bold text-[#00ff88] font-mono">
                      {handicap.awayOdds?.toFixed(2)}
                    </div>
                    {handicap.awayProb && (
                      <div className="text-[10px] text-gray-500 mt-1">
                        {handicap.awayProb.toFixed(0)}%
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Legend */}
      <div className="px-4 py-3 bg-white/5 border-t border-white/10">
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <Minus className="w-3 h-3" /> Asian: Stake refunded on draw
          </span>
          <span className="flex items-center gap-1">
            <AlertCircle className="w-3 h-3" /> European: Draw is possible
          </span>
        </div>
      </div>
    </div>
  );
}

// Simplified version for compact display
export function HandicapCompact({ 
  line, 
  odds, 
  probability,
  recommendation 
}: { 
  line: number;
  odds: number;
  probability?: number;
  recommendation?: boolean;
}) {
  return (
    <div className={cn(
      "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border",
      recommendation 
        ? "border-[#00ff88]/30 bg-[#00ff88]/10" 
        : "border-white/10 bg-white/5"
    )}>
      <span className="text-sm font-medium text-white">
        {line > 0 ? `+${line}` : line}
      </span>
      <span className="text-sm font-bold text-[#00d4ff] font-mono">
        {odds.toFixed(2)}
      </span>
      {probability && (
        <span className="text-xs text-gray-400">
          ({probability.toFixed(0)}%)
        </span>
      )}
    </div>
  );
}

export default HandicapDisplay;
