/**
 * RiskIndicator - Risk assessment visualization
 * 
 * Displays risk metrics with:
 * - Risk level indicator
 * - Variance/stdev
 * - Drawdown info
 * - Kelly criterion
 */

import { cn } from '@/lib/utils';
import { AlertTriangle, Shield, TrendingDown } from 'lucide-react';

interface RiskIndicatorProps {
  riskLevel: 'low' | 'medium' | 'high' | 'extreme';
  variance?: number;
  maxDrawdown?: number;
  kellyFraction?: number;
  bankrollRisk?: number;
  className?: string;
}

export function RiskIndicator({
  riskLevel,
  variance,
  maxDrawdown,
  kellyFraction,
  bankrollRisk,
  className,
}: RiskIndicatorProps) {
  const riskConfig = {
    low: {
      color: 'text-emerald-700 bg-emerald-50 border-emerald-200',
      icon: Shield,
      label: 'LOW RISK',
      description: 'Stable performance expected',
    },
    medium: {
      color: 'text-amber-700 bg-amber-50 border-amber-200',
      icon: AlertTriangle,
      label: 'MEDIUM RISK',
      description: 'Moderate volatility expected',
    },
    high: {
      color: 'text-orange-700 bg-orange-50 border-orange-200',
      icon: AlertTriangle,
      label: 'HIGH RISK',
      description: 'Significant volatility likely',
    },
    extreme: {
      color: 'text-red-700 bg-red-50 border-red-200',
      icon: TrendingDown,
      label: 'EXTREME RISK',
      description: 'High variance, caution advised',
    },
  };

  const config = riskConfig[riskLevel];
  const Icon = config.icon;

  return (
    <div className={cn(
      "rounded-lg border p-4",
      config.color,
      className
    )}>
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-md bg-white/50">
          <Icon className="w-5 h-5" />
        </div>
        
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-bold">{config.label}</span>
          </div>
          <p className="text-sm opacity-90 mb-3">{config.description}</p>
          
          <div className="grid grid-cols-2 gap-3">
            {variance !== undefined && (
              <div className="bg-white/50 rounded-md p-2">
                <div className="text-xs opacity-70">Variance</div>
                <div className="font-semibold tabular-nums">{variance.toFixed(3)}</div>
              </div>
            )}
            {maxDrawdown !== undefined && (
              <div className="bg-white/50 rounded-md p-2">
                <div className="text-xs opacity-70">Max Drawdown</div>
                <div className="font-semibold tabular-nums text-red-700">
                  {(maxDrawdown * 100).toFixed(1)}%
                </div>
              </div>
            )}
            {kellyFraction !== undefined && (
              <div className="bg-white/50 rounded-md p-2">
                <div className="text-xs opacity-70">Kelly %</div>
                <div className={cn(
                  "font-semibold tabular-nums",
                  kellyFraction > 0.1 ? 'text-red-700' : kellyFraction > 0.05 ? 'text-amber-700' : 'text-emerald-700'
                )}>
                  {(kellyFraction * 100).toFixed(2)}%
                </div>
              </div>
            )}
            {bankrollRisk !== undefined && (
              <div className="bg-white/50 rounded-md p-2">
                <div className="text-xs opacity-70">Ruin Risk</div>
                <div className={cn(
                  "font-semibold tabular-nums",
                  bankrollRisk > 0.05 ? 'text-red-700' : bankrollRisk > 0.02 ? 'text-amber-700' : 'text-emerald-700'
                )}>
                  {(bankrollRisk * 100).toFixed(2)}%
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default RiskIndicator;
