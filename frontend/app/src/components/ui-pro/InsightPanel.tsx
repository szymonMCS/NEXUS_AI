/**
 * InsightPanel - Professional insight/explanation panel
 * 
 * Displays analytical insights with context and explanations
 * Not just numbers - explains "what this means"
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Info,
  Lightbulb,
  TrendingDown,
  TrendingUp,
  XCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Insight {
  id: string;
  type: 'positive' | 'negative' | 'warning' | 'info' | 'neutral';
  title: string;
  description: string;
  metric?: {
    label: string;
    value: string;
    change?: string;
    trend?: 'up' | 'down' | 'neutral';
  };
  action?: {
    label: string;
    onClick?: () => void;
    href?: string;
  };
  details?: string[];
}

interface InsightPanelProps {
  insights: Insight[];
  title?: string;
  description?: string;
  collapsible?: boolean;
  maxDisplay?: number;
  className?: string;
}

const insightConfig = {
  positive: {
    icon: CheckCircle2,
    iconColor: 'text-[#00ff88]',
    bgColor: 'bg-[#00ff88]/10',
    borderColor: 'border-[#00ff88]/20',
    badge: 'bg-[#00ff88]/20 text-[#00ff88]',
  },
  negative: {
    icon: XCircle,
    iconColor: 'text-[#ff3860]',
    bgColor: 'bg-[#ff3860]/10',
    borderColor: 'border-[#ff3860]/20',
    badge: 'bg-[#ff3860]/20 text-[#ff3860]',
  },
  warning: {
    icon: AlertTriangle,
    iconColor: 'text-[#ff9500]',
    bgColor: 'bg-[#ff9500]/10',
    borderColor: 'border-[#ff9500]/20',
    badge: 'bg-[#ff9500]/20 text-[#ff9500]',
  },
  info: {
    icon: Info,
    iconColor: 'text-[#00d4ff]',
    bgColor: 'bg-[#00d4ff]/10',
    borderColor: 'border-[#00d4ff]/20',
    badge: 'bg-[#00d4ff]/20 text-[#00d4ff]',
  },
  neutral: {
    icon: Lightbulb,
    iconColor: 'text-gray-400',
    bgColor: 'bg-white/5',
    borderColor: 'border-white/10',
    badge: 'bg-white/10 text-gray-400',
  },
};

export function InsightPanel({
  insights,
  title = 'AI-Generated Insights',
  description,
  collapsible = false,
  maxDisplay = 4,
  className,
}: InsightPanelProps) {
  const displayInsights = insights.slice(0, maxDisplay);
  const hasMore = insights.length > maxDisplay;

  return (
    <Card className={cn('bg-[#0f1623]/80 border-white/10', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white flex items-center gap-2 text-base">
              <Lightbulb className="w-5 h-5 text-[#00d4ff]" />
              {title}
            </CardTitle>
            {description && (
              <p className="text-sm text-gray-500 mt-1">{description}</p>
            )}
          </div>
          <Badge variant="outline" className="border-white/10 text-gray-400">
            {insights.length} insights
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {displayInsights.map((insight) => {
          const config = insightConfig[insight.type];
          const Icon = config.icon;

          return (
            <div
              key={insight.id}
              className={cn(
                'p-4 rounded-lg border transition-all duration-200',
                config.bgColor,
                config.borderColor,
                'hover:brightness-110'
              )}
            >
              <div className="flex items-start gap-3">
                <Icon className={cn('w-5 h-5 mt-0.5 shrink-0', config.iconColor)} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h4 className="font-medium text-white">{insight.title}</h4>
                      <p className="text-sm text-gray-400 mt-1 leading-relaxed">
                        {insight.description}
                      </p>
                    </div>
                    <Badge className={cn('shrink-0 border-0', config.badge)}>
                      {insight.type}
                    </Badge>
                  </div>

                  {/* Metric Display */}
                  {insight.metric && (
                    <div className="mt-3 p-3 bg-black/20 rounded-lg">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">{insight.metric.label}</span>
                        <div className="flex items-center gap-2">
                          {insight.metric.trend && (
                            <>
                              {insight.metric.trend === 'up' && (
                                <TrendingUp className="w-4 h-4 text-[#00ff88]" />
                              )}
                              {insight.metric.trend === 'down' && (
                                <TrendingDown className="w-4 h-4 text-[#ff3860]" />
                              )}
                            </>
                          )}
                          <span className="text-white font-medium">{insight.metric.value}</span>
                          {insight.metric.change && (
                            <span className={cn(
                              'text-xs',
                              insight.metric.change.startsWith('+') ? 'text-[#00ff88]' : 'text-[#ff3860]'
                            )}>
                              {insight.metric.change}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Details List */}
                  {insight.details && insight.details.length > 0 && (
                    <ul className="mt-3 space-y-1">
                      {insight.details.map((detail, idx) => (
                        <li key={idx} className="text-xs text-gray-500 flex items-center gap-2">
                          <div className={cn('w-1 h-1 rounded-full', config.iconColor)} />
                          {detail}
                        </li>
                      ))}
                    </ul>
                  )}

                  {/* Action */}
                  {insight.action && (
                    <Button
                      variant="link"
                      size="sm"
                      className="text-[#00d4ff] p-0 h-auto mt-3 hover:text-[#00d4ff]/80"
                      onClick={insight.action.onClick}
                    >
                      {insight.action.label}
                      <ArrowRight className="w-3 h-3 ml-1" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {hasMore && (
          <Button variant="ghost" className="w-full text-gray-400 hover:text-white">
            View {insights.length - maxDisplay} more insights
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

export default InsightPanel;
