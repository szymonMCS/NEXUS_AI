/**
 * KPICard - Dynamic Sports KPI Card
 */

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface KPICardProps {
  label: string;
  value: string | number;
  subValue?: string;
  change?: number;
  changeType?: 'positive' | 'negative' | 'neutral';
  sparklineData?: number[];
  icon?: React.ReactNode;
  className?: string;
}

export function KPICard({
  label,
  value,
  subValue,
  change,
  changeType = 'neutral',
  sparklineData,
  icon,
  className,
}: KPICardProps) {
  const getChangeIcon = () => {
    if (change === undefined || change === 0) return <Minus className="w-3 h-3" />;
    if (change > 0) return <TrendingUp className="w-3 h-3" />;
    return <TrendingDown className="w-3 h-3" />;
  };

  const getChangeColor = () => {
    if (changeType === 'positive') return 'text-green-400 bg-green-400/10';
    if (changeType === 'negative') return 'text-red-400 bg-red-400/10';
    return 'text-slate-400 bg-slate-400/10';
  };

  // Simple sparkline
  const getSparkline = () => {
    if (!sparklineData || sparklineData.length < 2) return null;
    const max = Math.max(...sparklineData);
    const min = Math.min(...sparklineData);
    const range = max - min || 1;
    const points = sparklineData.map((v, i) => {
      const x = (i / (sparklineData.length - 1)) * 100;
      const y = 100 - ((v - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="w-full h-8" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={changeType === 'positive' ? '#4ade80' : changeType === 'negative' ? '#f87171' : '#60a5fa'}
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  };

  return (
    <div className={cn(
      "bg-slate-800/50 border border-slate-700 rounded-xl p-4",
      "hover:border-blue-500/30 hover:bg-slate-800 transition-all duration-300",
      "group cursor-pointer",
      className
    )}>
      <div className="flex items-start justify-between mb-2">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {label}
        </span>
        {change !== undefined && (
          <div className={cn(
            "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
            getChangeColor()
          )}>
            {getChangeIcon()}
            <span>{change > 0 ? '+' : ''}{change}%</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 mb-2">
        {icon && (
          <div className="p-2 rounded-lg bg-slate-700/50 text-blue-400 group-hover:text-blue-300 transition-colors">
            {icon}
          </div>
        )}
        <div>
          <div className="text-2xl font-bold text-white font-mono">
            {value}
          </div>
          {subValue && (
            <div className="text-xs text-slate-500">
              {subValue}
            </div>
          )}
        </div>
      </div>

      {sparklineData && (
        <div className="mt-2 opacity-50 group-hover:opacity-100 transition-opacity">
          {getSparkline()}
        </div>
      )}
    </div>
  );
}

export default KPICard;
