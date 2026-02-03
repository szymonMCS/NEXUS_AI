/**
 * PageLayout - Consistent page layout wrapper
 * 
 * Provides:
 * - Page header with title, description, and actions
 * - Breadcrumb navigation
 * - Consistent spacing and padding
 * - Optional tabs for sub-navigation
 */

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ChevronRight, Home } from 'lucide-react';

export interface BreadcrumbItem {
  label: string;
  href?: string;
}

export interface PageTab {
  id: string;
  label: string;
  badge?: number;
}

interface PageLayoutProps {
  title: string;
  description?: string;
  badge?: { label: string; variant?: 'default' | 'success' | 'warning' | 'error' };
  breadcrumbs?: BreadcrumbItem[];
  tabs?: PageTab[];
  activeTab?: string;
  onTabChange?: (tab: string) => void;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  fullWidth?: boolean;
}

export function PageLayout({
  title,
  description,
  badge,
  breadcrumbs = [],
  tabs,
  activeTab,
  onTabChange,
  actions,
  children,
  className,
  fullWidth = false,
}: PageLayoutProps) {
  const getBadgeClasses = (variant?: string) => {
    switch (variant) {
      case 'success':
        return 'bg-[#00ff88]/20 text-[#00ff88] border-0';
      case 'warning':
        return 'bg-[#ff9500]/20 text-[#ff9500] border-0';
      case 'error':
        return 'bg-[#ff3860]/20 text-[#ff3860] border-0';
      default:
        return 'bg-[#00d4ff]/20 text-[#00d4ff] border-0';
    }
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Breadcrumbs */}
      {breadcrumbs.length > 0 && (
        <nav className="flex items-center gap-2 text-sm text-gray-500">
          <button className="flex items-center gap-1 hover:text-white transition-colors">
            <Home className="w-4 h-4" />
          </button>
          {breadcrumbs.map((crumb, index) => (
            <div key={index} className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4" />
              {crumb.href ? (
                <button className="hover:text-white transition-colors">
                  {crumb.label}
                </button>
              ) : (
                <span className="text-gray-400">{crumb.label}</span>
              )}
            </div>
          ))}
        </nav>
      )}

      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-2xl font-bold text-white">{title}</h1>
            {badge && (
              <Badge className={getBadgeClasses(badge.variant)}>
                {badge.label}
              </Badge>
            )}
          </div>
          {description && (
            <p className="text-gray-400 text-sm">{description}</p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-2 shrink-0">
            {actions}
          </div>
        )}
      </div>

      {/* Tabs */}
      {tabs && tabs.length > 0 && (
        <Tabs value={activeTab} onValueChange={onTabChange}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10 h-10 p-1">
            {tabs.map((tab) => (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black px-4 py-1.5 text-sm"
              >
                {tab.label}
                {tab.badge !== undefined && tab.badge > 0 && (
                  <span className="ml-2 px-1.5 py-0.5 bg-white/20 rounded text-xs">
                    {tab.badge}
                  </span>
                )}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      )}

      {/* Content */}
      <div className={cn(!fullWidth && 'max-w-7xl')}>
        {children}
      </div>
    </div>
  );
}

export default PageLayout;
