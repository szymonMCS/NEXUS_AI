/**
 * InsightCard - Automated insight display component
 * 
 * Shows data-driven insights with:
 * - Insight type/icon
 * - Description
 * - Supporting metric
 * - Confidence level
 */

import { cn } from '@/lib/utils';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Zap,
  BarChart3,
  type LucideIcon
} from 'lucide-react';

interface InsightCardProps {
  type: 'positive' | 'negative' | 'neutral' | 'warning' | 'opportunity';
  title: string;
  description: string;
  metric?: {
    label: string;
    value: string | number;
    change?: number;
  };
  confidence?: number;
  className?: string;
}

const insightConfig: Record<string, { 
  icon: LucideIcon; 
  colors: string;
  label: string;
}> = {
  positive: {
    icon: TrendingUp,
    colors: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    label: 'Positive Signal',
  },
  negative: {
    icon: TrendingDown,
    colors: 'bg-red-50 border-red-200 text-red-800',
    label: 'Risk Factor',
  },
  neutral: {
    icon: BarChart3,
    colors: 'bg-slate-50 border-slate-200 text-slate-800',
    label: 'Data Point',
  },
  warning: {
    icon: AlertTriangle,
    colors: 'bg-amber-50 border-amber-200 text-amber-800',
    label: 'Warning',
  },
  opportunity: {
    icon: Zap,
    colors: 'bg-indigo-50 border-indigo-200 text-indigo-800',
    label: 'Opportunity',
  },
};

export function InsightCard({
  type,
  title,
  description,
  metric,
  confidence,
  className,
}: InsightCardProps) {
  const config = insightConfig[type];
  const Icon = config.icon;
  
  return (
    <div className={cn(
      "rounded-lg border p-3",
      config.colors,
      className
    )}>
      <div className="flex items-start gap-3">
        <div className="p-1.5 rounded-md bg-white/50">
          <Icon className="w-4 h-4" />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium uppercase tracking-wide opacity-70">
              {config.label}
            </span>
            {confidence !== undefined && (
              <span className={cn(
                "text-xs px-1.5 py-0.5 rounded-full",
                confidence >= 75 ? 'bg-emerald-100 text-emerald-700' :
                confidence >= 50 ? 'bg-amber-100 text-amber-700' :
                'bg-red-100 text-red-700'
              )}>
                {confidence}% conf
              </span>
            )}
          </div>
          
          <h4 className="font-semibold text-sm mb-1">{title}</h4>
          <p className="text-sm opacity-90 leading-relaxed">{description}</p>
          
          {metric && (
            <div className="mt-2 flex items-center gap-2 text-sm">
              <span className="opacity-70">{metric.label}:</span>
              <span className="font-semibold tabular-nums">{metric.value}</span>
              {metric.change !== undefined && (
                <span className={cn(
                  "text-xs",
                  metric.change > 0 ? 'text-emerald-700' : 
                  metric.change < 0 ? 'text-red-700' : 'opacity-70'
                )}>
                  {metric.change > 0 ? '+' : ''}{metric.change}%
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default InsightCard;
