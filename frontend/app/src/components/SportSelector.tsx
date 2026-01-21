// components/SportSelector.tsx
/**
 * Reusable sport selection component
 */

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useSports } from '@/hooks/use-sports';
import type { SportId } from '@/lib/api';

interface SportSelectorProps {
  value: SportId;
  onChange: (sport: SportId) => void;
  variant?: 'buttons' | 'cards' | 'compact';
  showBeta?: boolean;
  disabled?: boolean;
}

export function SportSelector({
  value,
  onChange,
  variant = 'buttons',
  showBeta = true,
  disabled = false,
}: SportSelectorProps) {
  const { sports, loading, activeSports, betaSports } = useSports();

  if (loading) {
    return (
      <div className="flex gap-2">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-10 w-24" />
        ))}
      </div>
    );
  }

  const allSports = showBeta ? sports : activeSports;

  if (variant === 'compact') {
    return (
      <div className="flex flex-wrap gap-2">
        {allSports.map((sport) => (
          <Button
            key={sport.id}
            variant={value === sport.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => onChange(sport.id)}
            disabled={disabled}
            className={value === sport.id ? 'bg-gradient-primary' : 'bg-white/5 border-white/10'}
          >
            <span className="mr-1">{sport.icon}</span>
            {sport.name}
            {sport.status === 'beta' && (
              <Badge variant="secondary" className="ml-1 text-[10px] px-1 py-0 bg-violet-500/20 text-violet-300">
                BETA
              </Badge>
            )}
          </Button>
        ))}
      </div>
    );
  }

  if (variant === 'cards') {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
        {allSports.map((sport) => (
          <Card
            key={sport.id}
            className={`cursor-pointer transition-all duration-200 hover:scale-105 ${
              value === sport.id
                ? 'bg-gradient-to-br from-violet-500/20 to-blue-500/20 border-violet-500/50'
                : 'bg-glass-card border-white/5 hover:border-white/20'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={() => !disabled && onChange(sport.id)}
          >
            <CardContent className="p-4 text-center">
              <div className="text-3xl mb-2">{sport.icon}</div>
              <div className="text-sm font-medium text-white">{sport.name}</div>
              {sport.status === 'beta' && (
                <Badge variant="secondary" className="mt-1 text-[10px] px-1 py-0 bg-violet-500/20 text-violet-300">
                  BETA
                </Badge>
              )}
              <div className="mt-2 text-xs text-gray-500">
                {sport.markets.length} markets
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  // Default: buttons variant
  return (
    <div className="flex flex-wrap gap-2">
      {activeSports.map((sport) => (
        <Button
          key={sport.id}
          variant={value === sport.id ? 'default' : 'outline'}
          onClick={() => onChange(sport.id)}
          disabled={disabled}
          className={value === sport.id ? 'bg-gradient-primary text-white' : 'bg-white/5 border-white/10 text-white'}
        >
          <span className="mr-2">{sport.icon}</span>
          {sport.name}
        </Button>
      ))}
      {showBeta && betaSports.length > 0 && (
        <>
          <div className="w-px bg-white/10 mx-2" />
          {betaSports.map((sport) => (
            <Button
              key={sport.id}
              variant={value === sport.id ? 'default' : 'outline'}
              onClick={() => onChange(sport.id)}
              disabled={disabled}
              className={`${
                value === sport.id ? 'bg-gradient-primary text-white' : 'bg-white/5 border-white/10 text-white'
              } opacity-80`}
            >
              <span className="mr-2">{sport.icon}</span>
              {sport.name}
              <Badge variant="secondary" className="ml-2 text-[10px] px-1 py-0 bg-violet-500/20 text-violet-300">
                BETA
              </Badge>
            </Button>
          ))}
        </>
      )}
    </div>
  );
}

export default SportSelector;
