/**
 * EdgeIndicator - Value edge visualization component
 * 
 * Displays betting edge with:
 * - Visual magnitude
 * - Color coding
 * - Optional fair odds comparison
 */

import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

interface EdgeIndicatorProps {
  edge: number; // Can be 0.08 (8%) or 8 (8%)
  odds?: number;
  fairOdds?: number;
  showOdds?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function EdgeIndicator({
  edge,
  odds,
  fairOdds,
  showOdds = true,
  size = 'md',
  className,
}: EdgeIndicatorProps) {
  // Normalize edge to decimal
  const normalizedEdge = Math.abs(edge) > 1 ? edge / 100 : edge;
  const percentage = normalizedEdge * 100;
  
  const getEdgeLevel = () => {
    const absEdge = Math.abs(normalizedEdge);
    if (absEdge >= 0.08) return 'high';
    if (absEdge >= 0.04) return 'medium';
    if (absEdge >= 0.02) return 'low';
    return 'none';
  };
  
  const getEdgeColor = () => {
    if (normalizedEdge <= 0) return 'text-slate-500 bg-slate-100';
    const level = getEdgeLevel();
    switch (level) {
      case 'high': return 'text-emerald-700 bg-emerald-100 border-emerald-200';
      case 'medium': return 'text-emerald-600 bg-emerald-50 border-emerald-100';
      case 'low': return 'text-amber-600 bg-amber-50 border-amber-100';
      default: return 'text-slate-500 bg-slate-50 border-slate-200';
    }
  };
  
  const getIcon = () => {
    if (normalizedEdge > 0) return <TrendingUp className="w-3.5 h-3.5" />;
    if (normalizedEdge < 0) return <TrendingDown className="w-3.5 h-3.5" />;
    return <AlertCircle className="w-3.5 h-3.5" />;
  };

  const sizeClasses = {
    sm: { container: 'text-xs px-1.5 py-0.5 gap-1', odds: 'text-[10px]' },
    md: { container: 'text-sm px-2 py-1 gap-1.5', odds: 'text-xs' },
    lg: { container: 'text-base px-3 py-1.5 gap-2', odds: 'text-sm' },
  };

  return (
    <div className={cn("flex flex-col", className)}>
      <div className={cn(
        "inline-flex items-center font-medium rounded-md border",
        sizeClasses[size].container,
        getEdgeColor()
      )}>
        {getIcon()}
        <span className="tabular-nums">
          {normalizedEdge > 0 ? '+' : ''}{percentage.toFixed(1)}%
        </span>
        <span className="opacity-70">edge</span>
      </div>
      
      {showOdds && (odds || fairOdds) && (
        <div className={cn("mt-1 flex items-center gap-2 text-slate-500", sizeClasses[size].odds)}>
          {odds && (
            <span className="tabular-nums">
              @ {odds.toFixed(2)}
            </span>
          )}
          {fairOdds && (
            <>
              <span className="text-slate-300">|</span>
              <span className="tabular-nums text-slate-400">
                fair: {fairOdds.toFixed(2)}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default EdgeIndicator;
