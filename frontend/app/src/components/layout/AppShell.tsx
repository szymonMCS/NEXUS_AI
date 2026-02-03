/**
 * AppShell - Main application shell with sidebar navigation
 * 
 * Provides:
 * - Professional sidebar navigation with 8+ pages
 * - Consistent header with search, notifications, user
 * - Breadcrumb navigation
 * - Responsive design (collapsible on mobile)
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Activity,
  BarChart3,
  BookOpen,
  Calendar,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  History,
  LayoutDashboard,
  LogOut,
  Menu,
  Search,
  Settings,
  Target,
  Trophy,
  User,
} from 'lucide-react';

export interface NavItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  href: string;
  badge?: string | number;
  description?: string;
}

const mainNavItems: NavItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    href: '/app',
    description: 'Overview and live matches',
  },
  {
    id: 'predictions',
    label: 'Predictions',
    icon: Target,
    href: '/app/predictions',
    badge: 47,
    description: 'Match predictions and analysis',
  },
  {
    id: 'matches',
    label: 'Matches',
    icon: Calendar,
    href: '/app/matches',
    badge: 12,
    description: 'All matches by league and date',
  },
  {
    id: 'handicaps',
    label: 'Handicaps',
    icon: Trophy,
    href: '/app/handicaps',
    description: 'Asian & European handicaps',
  },
  {
    id: 'statistics',
    label: 'Statistics',
    icon: BarChart3,
    href: '/app/statistics',
    description: 'Team and player statistics',
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: ClipboardList,
    href: '/app/reports',
    description: 'Analytical reports and summaries',
  },
  {
    id: 'models',
    label: 'Model Performance',
    icon: Activity,
    href: '/app/models',
    description: 'ML model metrics and accuracy',
  },
  {
    id: 'history',
    label: 'History',
    icon: History,
    href: '/app/history',
    description: 'Past predictions and results',
  },
];

const bottomNavItems: NavItem[] = [
  {
    id: 'docs',
    label: 'Documentation',
    icon: BookOpen,
    href: '/app/docs',
    description: 'Guides and API reference',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    href: '/app/settings',
    description: 'Application preferences',
  },
];

interface AppShellProps {
  children: React.ReactNode;
  currentPage?: string;
  onNavigate?: (href: string) => void;
}

export function AppShell({ children, currentPage = 'dashboard', onNavigate }: AppShellProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const handleNavClick = (href: string) => {
    onNavigate?.(href);
    setMobileMenuOpen(false);
  };

  const NavLink = ({ item, collapsed }: { item: NavItem; collapsed: boolean }) => {
    const isActive = currentPage === item.id;
    const Icon = item.icon;

    const content = (
      <button
        onClick={() => handleNavClick(item.href)}
        className={cn(
          'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
          'hover:bg-white/5 group relative',
          isActive 
            ? 'bg-[#00d4ff]/10 text-[#00d4ff] border border-[#00d4ff]/20' 
            : 'text-gray-400 hover:text-white border border-transparent'
        )}
      >
        <Icon className={cn(
          'w-5 h-5 shrink-0 transition-colors',
          isActive ? 'text-[#00d4ff]' : 'text-gray-500 group-hover:text-gray-300'
        )} />
        
        {!collapsed && (
          <>
            <span className="flex-1 text-sm font-medium text-left">{item.label}</span>
            {item.badge && (
              <Badge 
                variant="secondary" 
                className={cn(
                  'text-xs px-1.5 py-0 h-5 min-w-[20px] flex items-center justify-center',
                  isActive 
                    ? 'bg-[#00d4ff]/20 text-[#00d4ff] border-0' 
                    : 'bg-white/10 text-gray-400 border-0'
                )}
              >
                {item.badge}
              </Badge>
            )}
          </>
        )}

        {collapsed && item.badge && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-[#00d4ff] rounded-full text-[10px] flex items-center justify-center text-black font-bold">
            {item.badge}
          </span>
        )}
      </button>
    );

    if (collapsed) {
      return (
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>{content}</TooltipTrigger>
            <TooltipContent side="right" className="bg-[#0f1623] border-white/10">
              <div>
                <p className="font-medium">{item.label}</p>
                {item.description && (
                  <p className="text-xs text-gray-500">{item.description}</p>
                )}
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    }

    return content;
  };

  return (
    <TooltipProvider delayDuration={100}>
      <div className="min-h-screen flex">
        {/* Desktop Sidebar */}
        <aside
          className={cn(
            'hidden lg:flex flex-col border-r border-white/10 bg-[#0a0e1a]/95 backdrop-blur-xl transition-all duration-300 fixed h-full z-40',
            sidebarCollapsed ? 'w-16' : 'w-64'
          )}
        >
          {/* Logo */}
          <div className={cn(
            'h-16 flex items-center border-b border-white/10 px-4',
            sidebarCollapsed ? 'justify-center' : 'justify-between'
          )}>
            {!sidebarCollapsed && (
              <button 
                onClick={() => navigate('/')}
                className="flex items-center gap-2 hover:opacity-80 transition-opacity"
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center">
                  <Activity className="w-5 h-5 text-black" />
                </div>
                <span className="font-bold text-white">
                  NEXUS<span className="text-[#00d4ff]">AI</span>
                </span>
              </button>
            )}
            {sidebarCollapsed && (
              <button 
                onClick={() => navigate('/')}
                className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center hover:opacity-80 transition-opacity"
              >
                <Activity className="w-5 h-5 text-black" />
              </button>
            )}
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-7 w-7 text-gray-500 hover:text-white hover:bg-white/10',
                sidebarCollapsed && 'hidden'
              )}
              onClick={() => setSidebarCollapsed(true)}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
          </div>

          {/* Expand button when collapsed */}
          {sidebarCollapsed && (
            <Button
              variant="ghost"
              size="icon"
              className="mx-auto mt-2 h-7 w-7 text-gray-500 hover:text-white hover:bg-white/10"
              onClick={() => setSidebarCollapsed(false)}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          )}

          {/* Main Navigation */}
          <nav className={cn(
            'flex-1 overflow-y-auto py-4 space-y-1',
            sidebarCollapsed ? 'px-2' : 'px-3'
          )}>
            {!sidebarCollapsed && (
              <div className="px-3 mb-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                Main
              </div>
            )}
            {mainNavItems.map((item) => (
              <NavLink key={item.id} item={item} collapsed={sidebarCollapsed} />
            ))}
          </nav>

          {/* Bottom Navigation */}
          <div className={cn(
            'border-t border-white/10 py-4 space-y-1',
            sidebarCollapsed ? 'px-2' : 'px-3'
          )}>
            {!sidebarCollapsed && (
              <div className="px-3 mb-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
                System
              </div>
            )}
            {bottomNavItems.map((item) => (
              <NavLink key={item.id} item={item} collapsed={sidebarCollapsed} />
            ))}
          </div>
        </aside>

        {/* Mobile Sidebar Overlay */}
        {mobileMenuOpen && (
          <div 
            className="lg:hidden fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}

        {/* Mobile Sidebar */}
        <aside
          className={cn(
            'lg:hidden fixed inset-y-0 left-0 w-64 bg-[#0a0e1a] border-r border-white/10 z-50 transform transition-transform duration-300',
            mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
          )}
        >
          <div className="h-16 flex items-center justify-between px-4 border-b border-white/10">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center">
                <Activity className="w-5 h-5 text-black" />
              </div>
              <span className="font-bold text-white">
                NEXUS<span className="text-[#00d4ff]">AI</span>
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="text-gray-500 hover:text-white"
              onClick={() => setMobileMenuOpen(false)}
            >
              <ChevronLeft className="w-5 h-5" />
            </Button>
          </div>

          <nav className="p-3 space-y-1 overflow-y-auto h-[calc(100vh-64px)]">
            <div className="px-3 mb-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
              Main
            </div>
            {mainNavItems.map((item) => (
              <NavLink key={item.id} item={item} collapsed={false} />
            ))}
            
            <div className="px-3 mt-6 mb-2 text-xs font-semibold text-gray-600 uppercase tracking-wider">
              System
            </div>
            {bottomNavItems.map((item) => (
              <NavLink key={item.id} item={item} collapsed={false} />
            ))}
          </nav>
        </aside>

        {/* Main Content Area */}
        <main className={cn(
          'flex-1 min-h-screen transition-all duration-300',
          sidebarCollapsed ? 'lg:ml-16' : 'lg:ml-64'
        )}>
          {/* Header */}
          <header className="sticky top-0 z-30 h-16 bg-[#0a0e1a]/80 backdrop-blur-xl border-b border-white/10">
            <div className="h-full flex items-center justify-between px-4 lg:px-6">
              {/* Left: Mobile menu + Search */}
              <div className="flex items-center gap-4 flex-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="lg:hidden text-gray-400 hover:text-white"
                  onClick={() => setMobileMenuOpen(true)}
                >
                  <Menu className="w-5 h-5" />
                </Button>

                <div className="relative max-w-md w-full hidden md:block">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search matches, teams, leagues..."
                    className="w-full h-9 pl-10 pr-4 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-[#00d4ff]/50 focus:bg-white/10 transition-colors"
                  />
                </div>
              </div>

              {/* Right: Actions */}
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="hidden sm:flex items-center gap-2 text-gray-400 hover:text-white"
                >
                  <span className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
                  <span className="text-sm">Live: 12</span>
                </Button>
                
                <div className="w-px h-6 bg-white/10 hidden sm:block" />
                
                {/* User Menu */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#00d4ff] to-[#00ff88] flex items-center justify-center text-black font-bold text-sm">
                        G
                      </div>
                      <span className="hidden sm:block text-sm text-gray-300">Guest</span>
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56 bg-[#0f1623] border-white/10">
                    <DropdownMenuLabel className="text-white">
                      <div className="flex flex-col">
                        <span>Guest User</span>
                        <span className="text-xs text-gray-500 font-normal">guest@nexus.ai</span>
                      </div>
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator className="bg-white/10" />
                    <DropdownMenuItem 
                      className="text-gray-300 focus:text-white focus:bg-white/5 cursor-pointer"
                      onClick={() => navigate('/app/settings')}
                    >
                      <User className="w-4 h-4 mr-2" />
                      Profile
                    </DropdownMenuItem>
                    <DropdownMenuItem 
                      className="text-gray-300 focus:text-white focus:bg-white/5 cursor-pointer"
                      onClick={() => navigate('/app/settings')}
                    >
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </DropdownMenuItem>
                    <DropdownMenuSeparator className="bg-white/10" />
                    <DropdownMenuItem 
                      className="text-[#ff3860] focus:text-[#ff3860] focus:bg-[#ff3860]/10 cursor-pointer"
                      onClick={() => navigate('/sign-in')}
                    >
                      <LogOut className="w-4 h-4 mr-2" />
                      Log out
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <div className="p-4 lg:p-6">
            {children}
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}

export default AppShell;
