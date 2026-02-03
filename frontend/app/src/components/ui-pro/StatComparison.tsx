/**
 * StatComparison - Side-by-side statistical comparison component
 * 
 * For comparing two entities (teams, players, models) with visual indicators
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

export interface StatItem {
  label: string;
  valueA: number;
  valueB: number;
  unit?: string;
  higherIsBetter?: boolean;
  format?: 'number' | 'percent' | 'decimal';
}

interface StatComparisonProps {
  title: string;
  entityA: { name: string; color?: string };
  entityB: { name: string; color?: string };
  stats: StatItem[];
  className?: string;
}

export function StatComparison({
  title,
  entityA,
  entityB,
  stats,
  className,
}: StatComparisonProps) {
  const colorA = entityA.color || '#00d4ff';
  const colorB = entityB.color || '#00ff88';

  const formatValue = (value: number, format?: string, unit?: string) => {
    let formatted = value.toString();
    if (format === 'percent') formatted = `${value}%`;
    else if (format === 'decimal') formatted = value.toFixed(1);
    return `${formatted}${unit || ''}`;
  };

  return (
    <Card className={cn('bg-[#0f1623]/80 border-white/10', className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-white text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: colorA }}
            />
            <span className="font-medium text-white">{entityA.name}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-white">{entityB.name}</span>
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: colorB }}
            />
          </div>
        </div>

        {/* Stats */}
        {stats.map((stat, idx) => {
          const total = stat.valueA + stat.valueB;
          const pctA = total > 0 ? (stat.valueA / total) * 100 : 50;
          const pctB = total > 0 ? (stat.valueB / total) * 100 : 50;
          
          const aBetter = stat.higherIsBetter 
            ? stat.valueA > stat.valueB 
            : stat.valueA < stat.valueB;
          const bBetter = stat.higherIsBetter 
            ? stat.valueB > stat.valueA 
            : stat.valueB < stat.valueA;

          return (
            <div key={idx} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className={cn(
                  'font-medium',
                  aBetter ? 'text-white' : 'text-gray-400'
                )}>
                  {formatValue(stat.valueA, stat.format, stat.unit)}
                </span>
                <span className="text-gray-500 text-xs">{stat.label}</span>
                <span className={cn(
                  'font-medium',
                  bBetter ? 'text-white' : 'text-gray-400'
                )}>
                  {formatValue(stat.valueB, stat.format, stat.unit)}
                </span>
              </div>
              <div className="flex h-2 rounded-full overflow-hidden">
                <div 
                  className="transition-all duration-500"
                  style={{ 
                    width: `${pctA}%`,
                    backgroundColor: colorA,
                    opacity: 0.8
                  }}
                />
                <div 
                  className="transition-all duration-500"
                  style={{ 
                    width: `${pctB}%`,
                    backgroundColor: colorB,
                    opacity: 0.8
                  }}
                />
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

export default StatComparison;
