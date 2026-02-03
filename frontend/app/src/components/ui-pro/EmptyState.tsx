/**
 * EmptyState - Professional empty state component
 * 
 * Provides helpful context when no data is available
 */

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import type { LucideIcon } from 'lucide-react';
import { Search, Filter, Calendar, FileX } from 'lucide-react';
import { cn } from '@/lib/utils';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick?: () => void;
    href?: string;
  };
  secondaryAction?: {
    label: string;
    onClick?: () => void;
  };
  variant?: 'default' | 'compact' | 'inline';
  className?: string;
}

const iconMap: Record<string, LucideIcon> = {
  search: Search,
  filter: Filter,
  calendar: Calendar,
  default: FileX,
};

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  secondaryAction,
  variant = 'default',
  className,
}: EmptyStateProps) {
  const IconComponent = Icon || FileX;

  if (variant === 'inline') {
    return (
      <div className={cn('text-center py-8', className)}>
        <IconComponent className="w-10 h-10 text-gray-600 mx-auto mb-3" />
        <p className="text-gray-400 font-medium">{title}</p>
        {description && <p className="text-sm text-gray-500 mt-1">{description}</p>}
      </div>
    );
  }

  if (variant === 'compact') {
    return (
      <Card className={cn('bg-[#0f1623]/80 border-white/10 border-dashed', className)}>
        <CardContent className="p-6 text-center">
          <IconComponent className="w-8 h-8 text-gray-600 mx-auto mb-3" />
          <p className="text-white font-medium">{title}</p>
          {description && <p className="text-sm text-gray-500 mt-1">{description}</p>}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn('bg-[#0f1623]/80 border-white/10 border-dashed', className)}>
      <CardContent className="p-12 text-center">
        <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mx-auto mb-4">
          <IconComponent className="w-8 h-8 text-gray-500" />
        </div>
        <h3 className="text-lg font-medium text-white">{title}</h3>
        {description && (
          <p className="text-sm text-gray-400 mt-2 max-w-sm mx-auto">{description}</p>
        )}
        {(action || secondaryAction) && (
          <div className="flex items-center justify-center gap-3 mt-6">
            {action && (
              <Button
                className="bg-[#00d4ff] text-black hover:bg-[#00d4ff]/90"
                onClick={action.onClick}
              >
                {action.label}
              </Button>
            )}
            {secondaryAction && (
              <Button
                variant="outline"
                className="border-white/10 text-gray-400"
                onClick={secondaryAction.onClick}
              >
                {secondaryAction.label}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default EmptyState;
