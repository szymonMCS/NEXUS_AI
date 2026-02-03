/**
 * MatchPredictionSummary - Comprehensive match prediction report section
 * 
 * Displays:
 * - Match info and teams
 * - Prediction with probability
 * - Confidence level
 * - Key factors summary
 * - Value assessment
 */

import { cn } from '@/lib/utils';
import { 
  KPICard, 
  ProbabilityBar, 
  EdgeIndicator,
  InsightCard 
} from '@/components/analytics';
import { Calendar, MapPin, Clock } from 'lucide-react';

interface Team {
  name: string;
  rank?: number;
  form: string; // WWWLW
  winRate: number;
  homeAwayWinRate?: number;
}

interface Factor {
  name: string;
  impact: 'high' | 'medium' | 'low';
  value: string;
  description: string;
}

interface MatchPredictionSummaryProps {
  match: {
    homeTeam: Team;
    awayTeam: Team;
    league: string;
    venue?: string;
    datetime: string;
  };
  prediction: {
    winner: 'home' | 'away' | 'draw';
    probability: number;
    confidence: number;
    method: string;
  };
  valueBet?: {
    selection: string;
    odds: number;
    fairOdds: number;
    edge: number;
    stakeRecommendation: string;
  };
  factors: Factor[];
  insights: Array<{
    type: 'positive' | 'negative' | 'warning' | 'opportunity';
    title: string;
    description: string;
  }>;
  className?: string;
}

export function MatchPredictionSummary({
  match,
  prediction,
  valueBet,
  factors,
  insights,
  className,
}: MatchPredictionSummaryProps) {
  const predictedTeam = prediction.winner === 'home' ? match.homeTeam : 
                        prediction.winner === 'away' ? match.awayTeam : null;
  
  return (
    <div className={cn("space-y-4", className)}>
      {/* Match Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-1">
              {match.league}
            </div>
            <h2 className="text-lg font-semibold text-slate-900">
              {match.homeTeam.name} <span className="text-slate-400">vs</span> {match.awayTeam.name}
            </h2>
            <div className="flex items-center gap-4 mt-2 text-sm text-slate-500">
              <span className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                {new Date(match.datetime).toLocaleDateString()}
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                {new Date(match.datetime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
              {match.venue && (
                <span className="flex items-center gap-1">
                  <MapPin className="w-4 h-4" />
                  {match.venue}
                </span>
              )}
            </div>
          </div>
          
          {/* Main Prediction */}
          <div className="text-right">
            <div className="text-xs text-slate-500 mb-1">AI Prediction</div>
            <div className="text-2xl font-bold text-slate-900">
              {predictedTeam?.name || 'Draw'}
            </div>
            <div className="text-sm text-slate-500">
              {(prediction.probability * 100).toFixed(1)}% probability
            </div>
          </div>
        </div>
        
        {/* Probability Bars */}
        <div className="space-y-2">
          <ProbabilityBar
            label={match.homeTeam.name}
            value={prediction.winner === 'home' ? prediction.probability : (1 - prediction.probability) / 2}
            confidence={prediction.confidence}
            size="sm"
          />
          {prediction.winner === 'draw' && (
            <ProbabilityBar
              label="Draw"
              value={prediction.probability}
              confidence={prediction.confidence}
              size="sm"
            />
          )}
          <ProbabilityBar
            label={match.awayTeam.name}
            value={prediction.winner === 'away' ? prediction.probability : (1 - prediction.probability) / 2}
            confidence={prediction.confidence}
            size="sm"
          />
        </div>
      </div>

      {/* KPIs Row */}
      <div className="grid grid-cols-4 gap-3">
        <KPICard
          label="Model Confidence"
          value={`${(prediction.confidence * 100).toFixed(0)}%`}
          change={prediction.confidence > 0.7 ? 5 : -2}
          changeType={prediction.confidence > 0.7 ? 'positive' : 'negative'}
        />
        <KPICard
          label="Prediction Method"
          value={prediction.method}
        />
        {valueBet ? (
          <KPICard
            label="Value Edge"
            value={`+${(valueBet.edge * 100).toFixed(1)}%`}
            subValue={`Fair: ${valueBet.fairOdds.toFixed(2)}`}
            change={valueBet.edge * 100}
            changeType={valueBet.edge > 0.05 ? 'positive' : 'neutral'}
          />
        ) : (
          <KPICard
            label="Value Assessment"
            value="No Edge"
            changeType="neutral"
          />
        )}
        <KPICard
          label="Home Advantage"
          value={`${(match.homeTeam.homeAwayWinRate || match.homeTeam.winRate * 1.1 * 100).toFixed(0)}%`}
          subValue="Win rate at venue"
        />
      </div>

      {/* Key Factors */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-slate-900 mb-3 uppercase tracking-wide">
          Key Factors
        </h4>
        <div className="grid grid-cols-2 gap-3">
          {factors.slice(0, 4).map((factor) => (
            <div 
              key={factor.name}
              className="flex items-start gap-3 p-3 bg-slate-50 rounded-md"
            >
              <div className={cn(
                "w-2 h-2 rounded-full mt-1.5",
                factor.impact === 'high' ? 'bg-red-500' :
                factor.impact === 'medium' ? 'bg-amber-500' : 'bg-slate-400'
              )} />
              <div>
                <div className="text-sm font-medium text-slate-900">{factor.name}</div>
                <div className="text-sm text-slate-600">{factor.value}</div>
                <div className="text-xs text-slate-500 mt-0.5">{factor.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Value Bet Card */}
      {valueBet && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-semibold text-emerald-900 mb-1">
                Value Bet Identified
              </h4>
              <p className="text-sm text-emerald-700">
                Selection: <strong>{valueBet.selection}</strong> @ {valueBet.odds.toFixed(2)}
              </p>
            </div>
            <EdgeIndicator 
              edge={valueBet.edge} 
              odds={valueBet.odds}
              fairOdds={valueBet.fairOdds}
              size="md"
            />
          </div>
          <div className="mt-3 pt-3 border-t border-emerald-200">
            <span className="text-sm text-emerald-800">
              Stake Recommendation: <strong>{valueBet.stakeRecommendation}</strong>
            </span>
          </div>
        </div>
      )}

      {/* Insights */}
      {insights.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wide">
            Automated Insights
          </h4>
          <div className="grid grid-cols-2 gap-3">
            {insights.slice(0, 4).map((insight, i) => (
              <InsightCard
                key={i}
                type={insight.type}
                title={insight.title}
                description={insight.description}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default MatchPredictionSummary;
