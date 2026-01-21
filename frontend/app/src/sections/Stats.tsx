import { useEffect, useState, useRef, useMemo } from 'react';
import { TrendingUp, Calendar, Trophy, Target, Clock, Activity } from 'lucide-react';
import { useSports } from '@/hooks/use-sports';
import { useStats } from '@/hooks/use-stats';
import type { LucideIcon } from 'lucide-react';

interface StatItem {
  icon: LucideIcon;
  value: number;
  suffix: string;
  label: string;
  description: string;
}

function AnimatedCounter({ value, suffix, isVisible }: { value: number; suffix: string; isVisible: boolean }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!isVisible) return;

    let startTime: number;
    const duration = 2000;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      
      // Easing function
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      setCount(value * easeOutQuart);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, isVisible]);

  const displayValue = value % 1 !== 0 
    ? count.toFixed(1) 
    : Math.floor(count).toLocaleString();

  return (
    <span>
      {displayValue}
      {suffix}
    </span>
  );
}

export function Stats() {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);
  const { sports } = useSports();
  const { stats: apiStats } = useStats();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.2 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Generate stats from API data and static values
  const stats: StatItem[] = useMemo(() => [
    {
      icon: Trophy,
      value: sports.length || 5,
      suffix: '',
      label: 'Dyscyplin sportowych',
      description: 'Tennis, Basketball, Greyhound, Handball, Table Tennis',
    },
    {
      icon: Calendar,
      value: 1000,
      suffix: '+',
      label: 'Codziennych przewidywań',
      description: 'Nowe mecze co minutę',
    },
    {
      icon: Clock,
      value: 24,
      suffix: '/7',
      label: 'Analiza AI',
      description: 'Nieustanna praca algorytmów',
    },
    {
      icon: TrendingUp,
      value: apiStats?.avg_edge ? apiStats.avg_edge * 100 : 4.2,
      suffix: '%',
      label: 'Średni Edge',
      description: 'Przewaga nad rynkiem',
    },
    {
      icon: Activity,
      value: apiStats?.avg_quality || 72,
      suffix: '%',
      label: 'Jakość danych',
      description: 'Średnia ocena źródeł',
    },
    {
      icon: Target,
      value: apiStats?.win_rate ? apiStats.win_rate * 100 : 58,
      suffix: '%',
      label: 'Win Rate',
      description: 'Trafność predykcji',
    },
  ], [sports.length, apiStats]);

  return (
    <section ref={sectionRef} className="py-24 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-violet-500/5 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        {/* Header */}
        <div className="text-center mb-16">
          <h2
            className={`text-3xl sm:text-4xl font-bold text-white mb-4 transition-all duration-700 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Zaufaj liczbom
          </h2>
          <p
            className={`text-gray-400 max-w-2xl mx-auto transition-all duration-700 delay-100 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Nasza AI analizuje miliony danych, aby dostarczyć Ci najdokładniejsze przewidywania na rynku
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className={`group relative bg-glass-card rounded-2xl p-8 transition-all duration-500 hover:scale-105 hover:bg-white/5 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
              style={{ transitionDelay: `${index * 100}ms` }}
            >
              {/* Glow Effect */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-primary opacity-0 group-hover:opacity-10 transition-opacity duration-500" />

              {/* Icon */}
              <div className="w-14 h-14 rounded-xl bg-gradient-primary flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <stat.icon className="w-7 h-7 text-white" />
              </div>

              {/* Value */}
              <div className="text-4xl sm:text-5xl font-bold text-white mb-2">
                <AnimatedCounter value={stat.value} suffix={stat.suffix} isVisible={isVisible} />
              </div>

              {/* Label */}
              <div className="text-lg font-semibold text-white mb-1">{stat.label}</div>

              {/* Description */}
              <div className="text-sm text-gray-400">{stat.description}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
