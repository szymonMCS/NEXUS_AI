/**
 * ProbabilityBar - Dynamic Sports Probability Display
 */

import { cn } from '@/lib/utils';

interface ProbabilityBarProps {
  value: number;
  label: string;
  teamColor?: string;
  confidence?: number;
  marketProb?: number;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function ProbabilityBar({
  value,
  label,
  teamColor = '#3b82f6',
  confidence,
  marketProb,
  size = 'md',
  className,
}: ProbabilityBarProps) {
  const normalizedValue = value > 1 ? value / 100 : value;
  const percentage = Math.round(normalizedValue * 100);

  const sizeClasses = {
    sm: { bar: 'h-2', text: 'text-sm' },
    md: { bar: 'h-3', text: 'text-base' },
    lg: { bar: 'h-4', text: 'text-lg' },
  };

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span className={cn("font-semibold text-white", sizeClasses[size].text)}>
          {label}
        </span>
        <div className="flex items-center gap-2">
          <span className={cn("font-bold font-mono", sizeClasses[size].text)} style={{ color: teamColor }}>
            {percentage}%
          </span>
          {confidence !== undefined && (
            <span className="text-xs text-slate-500">
              ({Math.round(confidence * 100)}% conf)
            </span>
          )}
        </div>
      </div>

      <div className="relative">
        <div className={cn("w-full bg-slate-800 rounded-full overflow-hidden", sizeClasses[size].bar)}>
          <div
            className="h-full rounded-full transition-all duration-700 ease-out relative"
            style={{ 
              width: `${percentage}%`,
              background: `linear-gradient(90deg, ${teamColor}, ${teamColor}88)`
            }}
          >
            {/* Shimmer effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
          </div>
        </div>
        
        {/* Market comparison marker */}
        {marketProb !== undefined && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/50"
            style={{ left: `${marketProb * 100}%` }}
          />
        )}
      </div>
    </div>
  );
}

export default ProbabilityBar;
