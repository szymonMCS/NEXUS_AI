/**
 * ConfidenceGauge - Semi-circular gauge for confidence visualization
 * 
 * Displays model confidence with color-coded segments
 */

import { cn } from '@/lib/utils';

interface ConfidenceGaugeProps {
  value: number; // 0-1 or 0-100
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showSegments?: boolean;
  className?: string;
}

export function ConfidenceGauge({
  value,
  label = 'Confidence',
  size = 'md',
  showSegments = true,
  className,
}: ConfidenceGaugeProps) {
  const normalizedValue = value > 1 ? value / 100 : value;
  const percentage = Math.round(normalizedValue * 100);
  
  // Calculate arc path
  const radius = 40;
  const strokeWidth = 8;
  const center = 50;
  const startAngle = 180;
  const endAngle = 0;
  
  const getArcPath = (start: number, end: number) => {
    const startRad = (start * Math.PI) / 180;
    const endRad = (end * Math.PI) / 180;
    const x1 = center + radius * Math.cos(startRad);
    const y1 = center + radius * Math.sin(startRad);
    const x2 = center + radius * Math.cos(endRad);
    const y2 = center + radius * Math.sin(endRad);
    return `M ${x1} ${y1} A ${radius} ${radius} 0 0 1 ${x2} ${y2}`;
  };
  
  const getValuePosition = (val: number) => {
    const angle = 180 - (val * 180);
    const rad = (angle * Math.PI) / 180;
    return {
      x: center + radius * Math.cos(rad),
      y: center + radius * Math.sin(rad),
    };
  };
  
  const valuePos = getValuePosition(normalizedValue);
  
  const getColor = (val: number) => {
    if (val >= 0.75) return '#059669';
    if (val >= 0.5) return '#d97706';
    return '#dc2626';
  };
  
  const sizeClasses = {
    sm: { container: 'w-24 h-16', text: 'text-lg', label: 'text-xs' },
    md: { container: 'w-32 h-20', text: 'text-2xl', label: 'text-sm' },
    lg: { container: 'w-40 h-24', text: 'text-3xl', label: 'text-base' },
  };

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <div className={cn("relative", sizeClasses[size].container)}>
        <svg viewBox="0 0 100 55" className="w-full h-full">
          {/* Background arc */}
          <path
            d={getArcPath(startAngle, endAngle)}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          
          {/* Segments */}
          {showSegments && (
            <>
              <path
                d={getArcPath(180, 135)}
                fill="none"
                stroke="#dc2626"
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                opacity={0.3}
              />
              <path
                d={getArcPath(135, 90)}
                fill="none"
                stroke="#d97706"
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                opacity={0.3}
              />
              <path
                d={getArcPath(90, 45)}
                fill="none"
                stroke="#65a30d"
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                opacity={0.3}
              />
              <path
                d={getArcPath(45, 0)}
                fill="none"
                stroke="#059669"
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                opacity={0.3}
              />
            </>
          )}
          
          {/* Value arc */}
          <path
            d={getArcPath(180, 180 - (normalizedValue * 180))}
            fill="none"
            stroke={getColor(normalizedValue)}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          
          {/* Value marker */}
          <circle
            cx={valuePos.x}
            cy={valuePos.y}
            r="4"
            fill={getColor(normalizedValue)}
          />
        </svg>
        
        {/* Percentage text */}
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center">
          <span className={cn("font-bold tabular-nums", sizeClasses[size].text)} style={{ color: getColor(normalizedValue) }}>
            {percentage}%
          </span>
        </div>
      </div>
      
      {label && (
        <span className={cn("mt-1 font-medium text-slate-600", sizeClasses[size].label)}>
          {label}
        </span>
      )}
    </div>
  );
}

export default ConfidenceGauge;
