/**
 * TrendIndicator - Visual trend indicator with animation
 * 
 * Shows directional trends with appropriate styling
 */

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TrendIndicatorProps {
  value: number;
  previousValue?: number;
  percentChange?: number;
  showValue?: boolean;
  size?: 'sm' | 'md' | 'lg';
  format?: 'percent' | 'number' | 'currency';
  currency?: string;
  className?: string;
}

export function TrendIndicator({
  value,
  previousValue,
  percentChange,
  showValue = true,
  size = 'md',
  format = 'number',
  currency = '$',
  className,
}: TrendIndicatorProps) {
  // Calculate change if not provided
  const change = percentChange !== undefined 
    ? percentChange 
    : previousValue !== undefined && previousValue !== 0
      ? ((value - previousValue) / Math.abs(previousValue)) * 100
      : 0;

  const isPositive = change > 0;
  const isNegative = change < 0;
  const isNeutral = change === 0;

  const sizeClasses = {
    sm: { icon: 'w-3 h-3', text: 'text-xs', container: 'gap-1' },
    md: { icon: 'w-4 h-4', text: 'text-sm', container: 'gap-1.5' },
    lg: { icon: 'w-5 h-5', text: 'text-base', container: 'gap-2' },
  };

  const colors = {
    positive: 'text-[#00ff88]',
    negative: 'text-[#ff3860]',
    neutral: 'text-gray-500',
  };

  const formatValue = (val: number): string => {
    if (format === 'percent') return `${val.toFixed(1)}%`;
    if (format === 'currency') return `${currency}${val.toLocaleString()}`;
    return val.toLocaleString();
  };

  return (
    <div className={cn('flex items-center', sizeClasses[size].container, className)}>
      {isPositive && <TrendingUp className={cn(sizeClasses[size].icon, colors.positive)} />}
      {isNegative && <TrendingDown className={cn(sizeClasses[size].icon, colors.negative)} />}
      {isNeutral && <Minus className={cn(sizeClasses[size].icon, colors.neutral)} />}
      
      {showValue && (
        <span className={cn(
          sizeClasses[size].text,
          'font-medium',
          isPositive ? colors.positive : isNegative ? colors.negative : colors.neutral
        )}>
          {isPositive && '+'}{change.toFixed(1)}%
        </span>
      )}
    </div>
  );
}

export default TrendIndicator;
