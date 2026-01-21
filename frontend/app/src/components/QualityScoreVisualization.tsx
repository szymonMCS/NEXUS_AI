// components/QualityScoreVisualization.tsx
/**
 * Quality Score Visualization with gauge and breakdown
 */

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle, AlertTriangle, Info, Shield } from 'lucide-react';

interface QualityCategory {
  name: string;
  score: number;
  weight: number;
  description: string;
}

interface QualityScoreVisualizationProps {
  overallScore: number;
  categories?: QualityCategory[];
  warnings?: string[];
  recommendations?: string[];
  compact?: boolean;
}

const defaultCategories: QualityCategory[] = [
  { name: 'Zgodność źródeł', score: 85, weight: 0.35, description: 'Spójność danych między różnymi źródłami' },
  { name: 'Świeżość danych', score: 92, weight: 0.30, description: 'Jak aktualne są dane' },
  { name: 'Cross-validation', score: 78, weight: 0.20, description: 'Wzajemna weryfikacja źródeł' },
  { name: 'Wariancja kursów', score: 65, weight: 0.15, description: 'Stabilność kursów bukmacherskich' },
];

function getScoreColor(score: number): string {
  if (score >= 80) return 'text-green-400';
  if (score >= 60) return 'text-yellow-400';
  if (score >= 40) return 'text-orange-400';
  return 'text-red-400';
}

function getScoreGradient(score: number): string {
  if (score >= 80) return 'from-green-500 to-emerald-400';
  if (score >= 60) return 'from-yellow-500 to-amber-400';
  if (score >= 40) return 'from-orange-500 to-amber-500';
  return 'from-red-500 to-rose-400';
}

function getScoreLabel(score: number): { label: string; color: string } {
  if (score >= 85) return { label: 'Doskonała', color: 'bg-green-500/20 text-green-400' };
  if (score >= 70) return { label: 'Dobra', color: 'bg-blue-500/20 text-blue-400' };
  if (score >= 50) return { label: 'Umiarkowana', color: 'bg-yellow-500/20 text-yellow-400' };
  if (score >= 40) return { label: 'Słaba', color: 'bg-orange-500/20 text-orange-400' };
  return { label: 'Niewystarczająca', color: 'bg-red-500/20 text-red-400' };
}

// SVG Gauge Component
function ScoreGauge({ score, size = 180 }: { score: number; size?: number }) {
  const circumference = 2 * Math.PI * 70; // radius = 70
  const progress = (score / 100) * 0.75; // 75% of circle
  const dashOffset = circumference * (1 - progress);
  const rotation = -225; // Start from bottom-left

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox="0 0 180 180" className="transform -rotate-90">
        {/* Background arc */}
        <circle
          cx="90"
          cy="90"
          r="70"
          fill="none"
          stroke="currentColor"
          strokeWidth="12"
          strokeDasharray={`${circumference * 0.75} ${circumference}`}
          strokeDashoffset="0"
          strokeLinecap="round"
          className="text-white/10"
          style={{ transform: `rotate(${rotation}deg)`, transformOrigin: 'center' }}
        />
        {/* Progress arc */}
        <circle
          cx="90"
          cy="90"
          r="70"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="12"
          strokeDasharray={`${circumference * 0.75} ${circumference}`}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{ transform: `rotate(${rotation}deg)`, transformOrigin: 'center' }}
        />
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={score >= 60 ? '#22c55e' : '#ef4444'} />
            <stop offset="100%" stopColor={score >= 80 ? '#10b981' : score >= 60 ? '#eab308' : '#f97316'} />
          </linearGradient>
        </defs>
      </svg>
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-4xl font-bold ${getScoreColor(score)}`}>{score}</span>
        <span className="text-sm text-gray-400">/ 100</span>
      </div>
    </div>
  );
}

export function QualityScoreVisualization({
  overallScore,
  categories = defaultCategories,
  warnings = [],
  recommendations = [],
  compact = false,
}: QualityScoreVisualizationProps) {
  const scoreInfo = useMemo(() => getScoreLabel(overallScore), [overallScore]);

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        <div className={`text-2xl font-bold ${getScoreColor(overallScore)}`}>
          {overallScore}%
        </div>
        <Badge className={scoreInfo.color}>{scoreInfo.label}</Badge>
      </div>
    );
  }

  return (
    <Card className="bg-glass-card border-white/5">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-violet-400" />
          Jakość Danych
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Gauge */}
        <div className="flex flex-col items-center">
          <ScoreGauge score={overallScore} />
          <Badge className={`mt-2 ${scoreInfo.color}`}>{scoreInfo.label}</Badge>
        </div>

        {/* Categories Breakdown */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-400">Składniki oceny</h4>
          {categories.map((cat, i) => (
            <div key={i} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-300 flex items-center gap-2">
                  {cat.name}
                  <span className="text-xs text-gray-500">({(cat.weight * 100).toFixed(0)}%)</span>
                </span>
                <span className={`font-medium ${getScoreColor(cat.score)}`}>{cat.score}%</span>
              </div>
              <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 rounded-full bg-gradient-to-r ${getScoreGradient(cat.score)} transition-all duration-500`}
                  style={{ width: `${cat.score}%` }}
                />
              </div>
              <p className="text-xs text-gray-500">{cat.description}</p>
            </div>
          ))}
        </div>

        {/* Warnings */}
        {warnings.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-orange-400 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Ostrzeżenia
            </h4>
            <div className="space-y-1">
              {warnings.map((warning, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-orange-300 bg-orange-500/10 rounded-lg p-2">
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>{warning}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-blue-400 flex items-center gap-2">
              <Info className="w-4 h-4" />
              Rekomendacje
            </h4>
            <div className="space-y-1">
              {recommendations.map((rec, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-blue-300 bg-blue-500/10 rounded-lg p-2">
                  <CheckCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>{rec}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default QualityScoreVisualization;
