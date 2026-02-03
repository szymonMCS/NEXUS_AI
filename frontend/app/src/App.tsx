/**
 * NEXUS AI - Professional Sports Prediction Platform
 * 
 * Main App Router
 * - Landing page at root
 * - App shell with sidebar navigation for all internal pages
 * - Consistent layout and navigation across all pages
 * - Clerk authentication integration
 */

import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { AppShell } from '@/components/layout';
import { AtmosphericBackground } from '@/components/AtmosphericBackground';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// App Pages
import {
  DashboardPage,
  MatchesPage,
  HandicapsPage,
  ReportsPage,
  ModelsPage,
  StatisticsPage,
  HistoryPage,
  PredictionsPage,
  SettingsPage,
  DocumentationPage,
} from '@/pages/app';

// Legacy Pages (to be refactored)
import { LandingPage } from '@/pages/LandingPage';
import { SignInPage } from '@/pages/SignInPage';
import { SignUpPage } from '@/pages/SignUpPage';

import './App.css';

// Route to page ID mapping for sidebar highlighting
const routeToPageId: Record<string, string> = {
  '/app': 'dashboard',
  '/app/': 'dashboard',
  '/app/predictions': 'predictions',
  '/app/matches': 'matches',
  '/app/handicaps': 'handicaps',
  '/app/statistics': 'statistics',
  '/app/reports': 'reports',
  '/app/models': 'models',
  '/app/history': 'history',
  '/app/settings': 'settings',
  '/app/docs': 'docs',
};

// App Shell wrapper for all app pages
function AppLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  
  const currentPage = routeToPageId[location.pathname] || 'dashboard';

  return (
    <div className="min-h-screen relative">
      <AtmosphericBackground />
      <div className="relative z-10">
        <AppShell
          currentPage={currentPage}
          onNavigate={(href) => navigate(href)}
        >
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
        </AppShell>
      </div>
    </div>
  );
}

function App() {
  return (
    <div className="min-h-screen bg-[#0a0e1a]">
      <Routes>
        {/* Landing page - main entry (public) */}
        <Route path="/" element={<LandingPage />} />
        
        {/* Auth routes (public) */}
        <Route path="/sign-in/*" element={<SignInPage />} />
        <Route path="/sign-up/*" element={<SignUpPage />} />
        
        {/* App routes with sidebar layout - PROTECTED */}
        <Route path="/app" element={
          <ProtectedRoute>
            <AppLayout><DashboardPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/predictions" element={
          <ProtectedRoute>
            <AppLayout><PredictionsPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/matches" element={
          <ProtectedRoute>
            <AppLayout><MatchesPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/handicaps" element={
          <ProtectedRoute>
            <AppLayout><HandicapsPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/statistics" element={
          <ProtectedRoute>
            <AppLayout><StatisticsPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/reports" element={
          <ProtectedRoute>
            <AppLayout><ReportsPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/models" element={
          <ProtectedRoute>
            <AppLayout><ModelsPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/history" element={
          <ProtectedRoute>
            <AppLayout><HistoryPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/settings" element={
          <ProtectedRoute>
            <AppLayout><SettingsPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/app/docs" element={
          <ProtectedRoute>
            <AppLayout><DocumentationPage /></AppLayout>
          </ProtectedRoute>
        } />
        
        {/* Legacy routes - redirect to new structure - PROTECTED */}
        <Route path="/dashboard" element={
          <ProtectedRoute>
            <AppLayout><DashboardPage /></AppLayout>
          </ProtectedRoute>
        } />
        <Route path="/analysis" element={
          <ProtectedRoute>
            <AppLayout><ReportsPage /></AppLayout>
          </ProtectedRoute>
        } />
        
        {/* Fallback */}
        <Route path="*" element={<LandingPage />} />
      </Routes>
    </div>
  );
}

export default App;
