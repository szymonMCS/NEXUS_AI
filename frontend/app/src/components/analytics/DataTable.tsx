/**
 * DataTable - Professional data table with sorting and filtering
 * 
 * Features:
 * - Column sorting
 * - Row hover states
 * - Compact density
 * - Status indicators
 */

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface Column<T> {
  key: string;
  header: string;
  width?: string;
  align?: 'left' | 'center' | 'right';
  sortable?: boolean;
  formatter?: (value: unknown, row: T) => React.ReactNode;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (row: T) => string;
  className?: string;
  emptyMessage?: string;
  onRowClick?: (row: T) => void;
  selectedRow?: string | null;
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  className,
  emptyMessage = 'No data available',
  onRowClick,
  selectedRow,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  const sortedData = [...data].sort((a, b) => {
    if (!sortKey) return 0;
    const aVal = (a as Record<string, unknown>)[sortKey];
    const bVal = (b as Record<string, unknown>)[sortKey];
    
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    }
    
    const aStr = String(aVal).toLowerCase();
    const bStr = String(bVal).toLowerCase();
    if (sortDirection === 'asc') {
      return aStr.localeCompare(bStr);
    }
    return bStr.localeCompare(aStr);
  });

  const getAlignClass = (align?: string) => {
    switch (align) {
      case 'center': return 'text-center';
      case 'right': return 'text-right';
      default: return 'text-left';
    }
  };

  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "py-2 px-3 text-xs font-medium uppercase tracking-wider text-slate-500",
                  "bg-slate-50",
                  getAlignClass(col.align),
                  col.sortable && "cursor-pointer hover:bg-slate-100 select-none"
                )}
                style={{ width: col.width }}
                onClick={() => col.sortable && handleSort(col.key)}
              >
                <div className={cn(
                  "flex items-center gap-1",
                  col.align === 'right' && "justify-end",
                  col.align === 'center' && "justify-center"
                )}>
                  {col.header}
                  {col.sortable && sortKey === col.key && (
                    sortDirection === 'asc' ? 
                      <ChevronUp className="w-3 h-3" /> : 
                      <ChevronDown className="w-3 h-3" />
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.length === 0 ? (
            <tr>
              <td 
                colSpan={columns.length} 
                className="py-8 text-center text-slate-400"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sortedData.map((row) => (
              <tr
                key={keyExtractor(row)}
                className={cn(
                  "border-b border-slate-100 transition-colors",
                  onRowClick && "cursor-pointer hover:bg-slate-50",
                  selectedRow === keyExtractor(row) && "bg-indigo-50"
                )}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((col) => {
                  const value = (row as Record<string, unknown>)[col.key];
                  return (
                    <td
                      key={col.key}
                      className={cn(
                        "py-2 px-3",
                        getAlignClass(col.align)
                      )}
                    >
                      {col.formatter ? col.formatter(value, row) : String(value ?? '-')}
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default DataTable;
