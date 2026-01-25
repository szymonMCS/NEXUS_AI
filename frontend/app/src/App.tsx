import { Routes, Route, useLocation } from 'react-router-dom';
import { Navigation } from '@/sections/Navigation';
import { Hero } from '@/sections/Hero';
import { Stats } from '@/sections/Stats';
import { ValueBets } from '@/sections/ValueBets';
import { LivePredictions } from '@/sections/LivePredictions';
import { BettingBot } from '@/sections/BettingBot';
import { HowItWorks } from '@/sections/HowItWorks';
import { Blog } from '@/sections/Blog';
import { FAQ } from '@/sections/FAQ';
import { Footer } from '@/sections/Footer';
import { Dashboard, AnalysisPage, ValueBetsPage, ReportsPage, SettingsPage, SignInPage, SignUpPage } from '@/pages';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import './App.css';

// Landing Page component
function LandingPage() {
  return (
    <>
      <Hero />
      <Stats />
      <ValueBets />
      <LivePredictions />
      <BettingBot />
      <HowItWorks />
      <Blog />
      <FAQ />
    </>
  );
}

function App() {
  const location = useLocation();
  const isAuthPage = location.pathname.startsWith('/sign-');

  return (
    <div className="min-h-screen bg-background">
      {/* Hide navigation on auth pages */}
      {!isAuthPage && <Navigation />}
      <main>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/sign-in/*" element={<SignInPage />} />
          <Route path="/sign-up/*" element={<SignUpPage />} />

          {/* Protected routes */}
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/analysis" element={
            <ProtectedRoute>
              <AnalysisPage />
            </ProtectedRoute>
          } />
          <Route path="/value-bets" element={
            <ProtectedRoute>
              <ValueBetsPage />
            </ProtectedRoute>
          } />
          <Route path="/reports" element={
            <ProtectedRoute>
              <ReportsPage />
            </ProtectedRoute>
          } />
          <Route path="/settings" element={
            <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          } />
          {/* Bot route redirects to landing page bot section for now */}
          <Route path="/bot" element={<LandingPage />} />
        </Routes>
      </main>
      {/* Hide footer on auth pages */}
      {!isAuthPage && <Footer />}
    </div>
  );
}

export default App;
