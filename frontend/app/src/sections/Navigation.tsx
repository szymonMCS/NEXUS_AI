import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Menu, X, Zap, Trophy, Bot, Home, LayoutDashboard, FileText, Settings, Activity } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';

const navLinks = [
  { href: '/', label: 'Strona główna', icon: Home },
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/analysis', label: 'Analiza', icon: Activity },
  { href: '/value-bets', label: 'Value Bets', icon: Trophy },
  { href: '/reports', label: 'Raporty', icon: FileText },
  { href: '/bot', label: 'AI Bot', icon: Bot },
  { href: '/settings', label: 'Ustawienia', icon: Settings },
];

export function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        isScrolled
          ? 'bg-glass py-3 shadow-lg shadow-black/20'
          : 'bg-transparent py-5'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <nav className="flex items-center justify-between">
          {/* Logo */}
          <Link
            to="/"
            className="flex items-center gap-2 group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center glow-primary-sm group-hover:scale-110 transition-transform duration-300">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white">
              Sport<span className="text-gradient">AI</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const active = isActive(link.href);
              return (
                <Link
                  key={link.href}
                  to={link.href}
                  className={`flex items-center gap-2 px-4 py-2 text-sm rounded-lg transition-all duration-200 ${
                    active
                      ? 'text-white bg-white/10'
                      : 'text-gray-300 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Desktop CTA */}
          <div className="hidden md:flex items-center gap-3">
            <SignedOut>
              <Link to="/sign-in">
                <Button
                  variant="ghost"
                  className="text-gray-300 hover:text-white hover:bg-white/5"
                >
                  Zaloguj się
                </Button>
              </Link>
              <Link to="/sign-up">
                <Button className="bg-gradient-primary hover:opacity-90 text-white font-semibold px-6">
                  Rozpocznij za darmo
                </Button>
              </Link>
            </SignedOut>
            <SignedIn>
              <UserButton
                appearance={{
                  elements: {
                    avatarBox: 'w-10 h-10',
                    userButtonPopoverCard: 'bg-gray-900 border border-white/10',
                    userButtonPopoverActionButton: 'text-gray-300 hover:text-white hover:bg-white/5',
                    userButtonPopoverActionButtonText: 'text-gray-300',
                    userButtonPopoverActionButtonIcon: 'text-gray-400',
                    userButtonPopoverFooter: 'hidden',
                  },
                }}
                afterSignOutUrl="/"
              />
            </SignedIn>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
          >
            {isMobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </button>
        </nav>

        {/* Mobile Menu */}
        <div
          className={`md:hidden overflow-hidden transition-all duration-300 ${
            isMobileMenuOpen ? 'max-h-96 mt-4' : 'max-h-0'
          }`}
        >
          <div className="bg-glass rounded-xl p-4 space-y-2">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const active = isActive(link.href);
              return (
                <Link
                  key={link.href}
                  to={link.href}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    active
                      ? 'text-white bg-white/10'
                      : 'text-gray-300 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {link.label}
                </Link>
              );
            })}
            <div className="pt-2 border-t border-white/10 space-y-2">
              <SignedOut>
                <Link to="/sign-in" onClick={() => setIsMobileMenuOpen(false)}>
                  <Button
                    variant="ghost"
                    className="w-full text-gray-300 hover:text-white hover:bg-white/5"
                  >
                    Zaloguj się
                  </Button>
                </Link>
                <Link to="/sign-up" onClick={() => setIsMobileMenuOpen(false)}>
                  <Button className="w-full bg-gradient-primary hover:opacity-90 text-white font-semibold">
                    Rozpocznij za darmo
                  </Button>
                </Link>
              </SignedOut>
              <SignedIn>
                <div className="flex items-center justify-center py-2">
                  <UserButton
                    appearance={{
                      elements: {
                        avatarBox: 'w-12 h-12',
                        userButtonPopoverCard: 'bg-gray-900 border border-white/10',
                        userButtonPopoverActionButton: 'text-gray-300 hover:text-white hover:bg-white/5',
                        userButtonPopoverActionButtonText: 'text-gray-300',
                        userButtonPopoverActionButtonIcon: 'text-gray-400',
                        userButtonPopoverFooter: 'hidden',
                      },
                    }}
                    afterSignOutUrl="/"
                  />
                </div>
              </SignedIn>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}