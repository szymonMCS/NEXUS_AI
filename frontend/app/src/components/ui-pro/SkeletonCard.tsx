/**
 * SkeletonCard - Loading skeleton for cards
 * 
 * Professional loading state that matches the card layout
 */

import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

interface SkeletonCardProps {
  header?: boolean;
  rows?: number;
  columns?: number;
  className?: string;
}

export function SkeletonCard({
  header = true,
  rows = 3,
  columns = 1,
  className,
}: SkeletonCardProps) {
  return (
    <Card className={cn('bg-[#0f1623]/80 border-white/10', className)}>
      {header && (
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <Skeleton className="h-5 w-32 bg-white/10" />
            <Skeleton className="h-4 w-16 bg-white/10" />
          </div>
        </CardHeader>
      )}
      <CardContent className="space-y-4">
        {Array.from({ length: rows }).map((_, rowIdx) => (
          <div 
            key={rowIdx} 
            className={cn(
              'gap-4',
              columns > 1 ? 'grid' : 'flex flex-col'
            )}
            style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
          >
            {Array.from({ length: columns }).map((_, colIdx) => (
              <div key={colIdx} className="space-y-2">
                <Skeleton className="h-4 w-full bg-white/10" />
                <Skeleton className="h-8 w-3/4 bg-white/10" />
              </div>
            ))}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

interface SkeletonTableProps {
  rows?: number;
  columns?: number;
  className?: string;
}

export function SkeletonTable({
  rows = 5,
  columns = 4,
  className,
}: SkeletonTableProps) {
  return (
    <div className={cn('space-y-3', className)}>
      {/* Header */}
      <div className="flex gap-4 pb-3 border-b border-white/10">
        {Array.from({ length: columns }).map((_, idx) => (
          <Skeleton 
            key={idx} 
            className="h-4 bg-white/10 flex-1" 
          />
        ))}
      </div>
      
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIdx) => (
        <div key={rowIdx} className="flex gap-4 py-3">
          {Array.from({ length: columns }).map((_, colIdx) => (
            <Skeleton 
              key={colIdx} 
              className="h-4 bg-white/10 flex-1" 
            />
          ))}
        </div>
      ))}
    </div>
  );
}

interface SkeletonKPIProps {
  count?: number;
  className?: string;
}

export function SkeletonKPI({ count = 4, className }: SkeletonKPIProps) {
  return (
    <div className={cn('grid grid-cols-2 lg:grid-cols-4 gap-4', className)}>
      {Array.from({ length: count }).map((_, idx) => (
        <Card key={idx} className="bg-[#0f1623]/80 border-white/10">
          <CardContent className="p-5">
            <div className="flex items-start justify-between">
              <div className="space-y-2 flex-1">
                <Skeleton className="h-3 w-20 bg-white/10" />
                <Skeleton className="h-8 w-16 bg-white/10" />
                <Skeleton className="h-3 w-12 bg-white/10" />
              </div>
              <Skeleton className="h-10 w-10 rounded-lg bg-white/10" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export default SkeletonCard;
