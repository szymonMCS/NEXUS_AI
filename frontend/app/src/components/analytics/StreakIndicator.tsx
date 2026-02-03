/**
 * StreakIndicator - Win/loss streak visualization
 * 
 * Shows current and best streaks with visual indicator
 */

import { cn } from '@/lib/utils';
import { Trophy, Flame, Snowflake } from 'lucide-react';

interface StreakIndicatorProps {
  currentStreak: number; // Positive for wins, negative for losses
  bestStreak: number;
  worstStreak?: number;
  className?: string;
}

export function StreakIndicator({
  currentStreak,
  bestStreak,
  worstStreak,
  className,
}: StreakIndicatorProps) {
  const isWinning = currentStreak > 0;
  const isLosing = currentStreak < 0;
  
  return (
    <div className={cn("bg-white border border-slate-200 rounded-lg p-4", className)}>
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">
        Streak Analysis
      </h4>
      
      {/* Current Streak */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {isWinning ? (
            <Flame className="w-5 h-5 text-orange-500" />
          ) : isLosing ? (
            <Snowflake className="w-5 h-5 text-blue-400" />
          ) : (
            <div className="w-5 h-5 rounded-full bg-slate-200" />
          )}
          <span className="text-sm text-slate-600">Current Streak</span>
        </div>
        <div className={cn(
          "text-2xl font-bold tabular-nums",
          isWinning ? 'text-emerald-600' : isLosing ? 'text-red-600' : 'text-slate-400'
        )}>
          {isWinning ? '+' : ''}{currentStreak}
        </div>
      </div>
      
      {/* Visual Bar */}
      <div className="mb-4">
        <div className="flex h-3 rounded-full overflow-hidden bg-slate-100">
          {/* Worst streak section */}
          <div 
            className="bg-red-200"
            style={{ width: `${Math.min(Math.abs(worstStreak || 0) / (bestStreak + Math.abs(worstStreak || 0)) * 50, 50)}%` }}
          />
          {/* Neutral section */}
          <div className="w-px bg-slate-300" />
          {/* Current streak indicator */}
          <div 
            className="bg-slate-300"
            style={{ width: '48%' }}
          >
            {currentStreak !== 0 && (
              <div 
                className={cn(
                  "h-full w-1 transition-all",
                  isWinning ? 'bg-emerald-500 ml-auto' : 'bg-red-500'
                )}
                style={{ 
                  marginRight: isWinning ? 'auto' : undefined,
                  marginLeft: isLosing ? 'auto' : undefined
                }}
              />
            )}
          </div>
          {/* Best streak section */}
          <div 
            className="bg-emerald-200"
            style={{ width: `${Math.min(bestStreak / (bestStreak + Math.abs(worstStreak || 0)) * 50, 50)}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-slate-400 mt-1">
          <span>{worstStreak || 0}</span>
          <span>0</span>
          <span>+{bestStreak}</span>
        </div>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center gap-2 p-2 bg-emerald-50 rounded-md">
          <Trophy className="w-4 h-4 text-emerald-600" />
          <div>
            <div className="text-xs text-slate-500">Best</div>
            <div className="font-semibold text-emerald-700">+{bestStreak}</div>
          </div>
        </div>
        {worstStreak !== undefined && (
          <div className="flex items-center gap-2 p-2 bg-red-50 rounded-md">
            <Snowflake className="w-4 h-4 text-red-600" />
            <div>
              <div className="text-xs text-slate-500">Worst</div>
              <div className="font-semibold text-red-700">{worstStreak}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StreakIndicator;
