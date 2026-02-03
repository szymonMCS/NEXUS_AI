/**
 * ModelMetrics - Model performance and accuracy display
 * 
 * Shows key ML model metrics:
 * - Accuracy, Precision, Recall, F1
 * - Calibration metrics
 * - ROC/AUC
 */

import { cn } from '@/lib/utils';

interface ModelMetric {
  name: string;
  value: number; // 0-1
  benchmark?: number;
  description?: string;
}

interface ModelMetricsProps {
  metrics: ModelMetric[];
  calibrationScore?: number;
  aucScore?: number;
  brierScore?: number;
  logLoss?: number;
  className?: string;
}

export function ModelMetrics({
  metrics,
  calibrationScore,
  aucScore,
  brierScore,
  logLoss,
  className,
}: ModelMetricsProps) {
  const getMetricColor = (value: number) => {
    if (value >= 0.75) return 'bg-emerald-500';
    if (value >= 0.60) return 'bg-amber-500';
    return 'bg-red-500';
  };
  
  const getMetricTextColor = (value: number) => {
    if (value >= 0.75) return 'text-emerald-700';
    if (value >= 0.60) return 'text-amber-700';
    return 'text-red-700';
  };

  return (
    <div className={cn("bg-white border border-slate-200 rounded-lg p-4", className)}>
      <h3 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">
        Model Performance
      </h3>
      
      <div className="space-y-4">
        {metrics.map((metric) => (
          <div key={metric.name} className="space-y-1.5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">{metric.name}</span>
              <div className="flex items-center gap-2">
                <span className={cn("font-semibold tabular-nums", getMetricTextColor(metric.value))}>
                  {(metric.value * 100).toFixed(1)}%
                </span>
                {metric.benchmark !== undefined && (
                  <span className="text-xs text-slate-400">
                    (bench: {(metric.benchmark * 100).toFixed(0)}%)
                  </span>
                )}
              </div>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={cn("h-full rounded-full transition-all", getMetricColor(metric.value))}
                style={{ width: `${metric.value * 100}%` }}
              />
            </div>
            {metric.description && (
              <p className="text-xs text-slate-500">{metric.description}</p>
            )}
          </div>
        ))}
      </div>
      
      {/* Additional Scores */}
      <div className="mt-6 pt-4 border-t border-slate-100">
        <div className="grid grid-cols-2 gap-4">
          {aucScore !== undefined && (
            <div>
              <div className="text-xs text-slate-500 mb-1">ROC-AUC</div>
              <div className={cn(
                "text-lg font-semibold tabular-nums",
                aucScore >= 0.8 ? 'text-emerald-600' : aucScore >= 0.7 ? 'text-amber-600' : 'text-red-600'
              )}>
                {aucScore.toFixed(3)}
              </div>
            </div>
          )}
          {calibrationScore !== undefined && (
            <div>
              <div className="text-xs text-slate-500 mb-1">Calibration</div>
              <div className={cn(
                "text-lg font-semibold tabular-nums",
                calibrationScore >= 0.9 ? 'text-emerald-600' : calibrationScore >= 0.8 ? 'text-amber-600' : 'text-red-600'
              )}>
                {(calibrationScore * 100).toFixed(1)}%
              </div>
            </div>
          )}
          {brierScore !== undefined && (
            <div>
              <div className="text-xs text-slate-500 mb-1">Brier Score</div>
              <div className={cn(
                "text-lg font-semibold tabular-nums",
                brierScore <= 0.15 ? 'text-emerald-600' : brierScore <= 0.25 ? 'text-amber-600' : 'text-red-600'
              )}>
                {brierScore.toFixed(3)}
              </div>
              <div className="text-xs text-slate-400">lower is better</div>
            </div>
          )}
          {logLoss !== undefined && (
            <div>
              <div className="text-xs text-slate-500 mb-1">Log Loss</div>
              <div className={cn(
                "text-lg font-semibold tabular-nums",
                logLoss <= 0.4 ? 'text-emerald-600' : logLoss <= 0.6 ? 'text-amber-600' : 'text-red-600'
              )}>
                {logLoss.toFixed(3)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ModelMetrics;
