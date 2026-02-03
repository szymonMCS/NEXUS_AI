/**
 * ModelAnalysisReport - Detailed model accuracy and calibration report
 * 
 * Displays:
 * - Model accuracy metrics
 * - Calibration curve
 * - Prediction distribution
 * - Feature importance
 */

import { cn } from '@/lib/utils';
import { ModelMetrics, ConfidenceGauge } from '@/components/analytics';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from 'recharts';

interface CalibrationPoint {
  predicted: number;
  actual: number;
  count: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  category: 'form' | 'h2h' | 'context' | 'market';
}

interface PredictionDistribution {
  range: string;
  count: number;
  accuracy: number;
}

interface ModelAnalysisReportProps {
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    auc: number;
    brierScore: number;
    calibration: number;
    logLoss: number;
  };
  calibrationData: CalibrationPoint[];
  featureImportance: FeatureImportance[];
  predictionDistribution: PredictionDistribution[];
  recentPredictions: Array<{
    match: string;
    predicted: number;
    actual: number;
    correct: boolean;
  }>;
  className?: string;
}

const categoryColors: Record<string, string> = {
  form: '#3b82f6',
  h2h: '#8b5cf6',
  context: '#f59e0b',
  market: '#10b981',
};

export function ModelAnalysisReport({
  metrics,
  calibrationData,
  featureImportance,
  predictionDistribution,
  recentPredictions,
  className,
}: ModelAnalysisReportProps) {
  return (
    <div className={cn("space-y-4", className)}>
      {/* Metrics Row */}
      <div className="grid grid-cols-4 gap-3">
        <ModelMetrics
          metrics={[
            { name: 'Accuracy', value: metrics.accuracy, description: 'Overall prediction accuracy' },
            { name: 'Precision', value: metrics.precision, description: 'True positives / predicted positives' },
            { name: 'Recall', value: metrics.recall, description: 'True positives / actual positives' },
            { name: 'F1 Score', value: metrics.f1Score, description: 'Harmonic mean of precision and recall' },
          ]}
          calibrationScore={metrics.calibration}
          aucScore={metrics.auc}
          brierScore={metrics.brierScore}
          logLoss={metrics.logLoss}
        />
        
        {/* Overall Score */}
        <div className="bg-white border border-slate-200 rounded-lg p-4 flex flex-col items-center justify-center">
          <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
            Model Score
          </div>
          <ConfidenceGauge
            value={(metrics.accuracy + metrics.calibration + metrics.auc) / 3}
            label="Overall"
            size="lg"
          />
          <div className="mt-3 text-center">
            <div className="text-sm text-slate-600">
              AUC: <span className="font-semibold tabular-nums">{metrics.auc.toFixed(3)}</span>
            </div>
            <div className="text-xs text-slate-400">
              Brier: {metrics.brierScore.toFixed(3)}
            </div>
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Calibration Chart */}
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-900 mb-1 uppercase tracking-wide">
            Calibration Curve
          </h4>
          <p className="text-xs text-slate-500 mb-4">
            Predicted vs actual outcomes. Perfect calibration follows the diagonal.
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  type="number" 
                  dataKey="predicted" 
                  name="Predicted" 
                  domain={[0, 1]}
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: 'Predicted Probability', position: 'bottom', fontSize: 11 }}
                />
                <YAxis 
                  type="number" 
                  dataKey="actual" 
                  name="Actual" 
                  domain={[0, 1]}
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: 'Actual Rate', angle: -90, position: 'insideLeft', fontSize: 11 }}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ fontSize: 12 }}
                  formatter={(value: number, name: string) => [`${(value * 100).toFixed(1)}%`, name]}
                />
                <ReferenceLine x={0} stroke="#cbd5e1" />
                <ReferenceLine y={0} stroke="#cbd5e1" />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="5 5" />
                <Scatter 
                  name="Calibration" 
                  data={calibrationData} 
                  fill="#3b82f6"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-900 mb-1 uppercase tracking-wide">
            Feature Importance
          </h4>
          <p className="text-xs text-slate-500 mb-4">
            Relative importance of input features in model predictions.
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportance.slice(0, 8)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis 
                  dataKey="feature" 
                  type="category" 
                  tick={{ fontSize: 10 }}
                  width={100}
                />
                <Tooltip 
                  contentStyle={{ fontSize: 12 }}
                  formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {featureImportance.slice(0, 8).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={categoryColors[entry.category]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-3">
            {Object.entries(categoryColors).map(([cat, color]) => (
              <div key={cat} className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-xs text-slate-600 capitalize">{cat}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Prediction Distribution */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">
          Prediction Accuracy by Confidence Range
        </h4>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={predictionDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip contentStyle={{ fontSize: 12 }} />
              <Bar yAxisId="left" dataKey="count" name="Predictions" fill="#e2e8f0" />
              <Bar yAxisId="right" dataKey="accuracy" name="Accuracy %" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="bg-white border border-slate-200 rounded-lg">
        <div className="px-4 py-3 border-b border-slate-200">
          <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wide">
            Recent Predictions
          </h4>
        </div>
        <div className="divide-y divide-slate-100">
          {recentPredictions.slice(0, 5).map((pred, i) => (
            <div key={i} className="px-4 py-3 flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-slate-900 truncate">{pred.match}</div>
                <div className="text-xs text-slate-500">
                  Predicted: {(pred.predicted * 100).toFixed(0)}% | 
                  Actual: {(pred.actual * 100).toFixed(0)}%
                </div>
              </div>
              <div className={cn(
                "px-2 py-1 rounded text-xs font-medium",
                pred.correct 
                  ? 'bg-emerald-100 text-emerald-700' 
                  : 'bg-red-100 text-red-700'
              )}>
                {pred.correct ? 'Correct' : 'Incorrect'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ModelAnalysisReport;
