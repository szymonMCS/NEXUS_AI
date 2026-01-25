import { useEffect, useState, useRef } from 'react';
import { Database, Brain, TrendingUp, RefreshCw, ArrowRight } from 'lucide-react';

const steps = [
  {
    number: '01',
    icon: Database,
    title: 'Zbieranie danych',
    description:
      'Nasze systemy gromadzą kompleksowe dane z tysięcy źródeł, w tym statystyki drużyn, wydajność zawodników, warunki pogodowe, kontuzje, historia spotkań i wiele więcej.',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    number: '02',
    icon: Brain,
    title: 'Przetwarzanie AI',
    description:
      'Zaawansowane algorytmy uczenia maszynowego analizują wzorce, trendy i korelacje, które mogą umknąć ludzkim analitykom. Nasza AI przetwarza miliony danych w ułamku sekundy.',
    color: 'from-violet-500 to-purple-500',
  },
  {
    number: '03',
    icon: TrendingUp,
    title: 'Generowanie przewidywań',
    description:
      'System generuje precyzyjne przewidywania z poziomami pewności i rekomendowanymi typami zakładów. Każda prognoza jest weryfikowana pod kątem wartości oczekiwanej.',
    color: 'from-green-500 to-emerald-500',
  },
  {
    number: '04',
    icon: RefreshCw,
    title: 'Ciągłe uczenie się',
    description:
      'Nasz system uczy się z każdego wyniku, stale poprawiając dokładność i dostosowując się do nowych wzorców. To samodoskonalący się cykl, który staje się coraz lepszy.',
    color: 'from-orange-500 to-red-500',
  },
];

export function HowItWorks() {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} id="how-it-works" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <div
            className={`inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-full mb-6 transition-all duration-700 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            <Brain className="w-4 h-4 text-violet-400" />
            <span className="text-sm text-violet-400 font-medium">Jak to działa</span>
          </div>
          <h2
            className={`text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-6 transition-all duration-700 delay-100 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Nauka za przewidywaniami
          </h2>
          <p
            className={`text-gray-400 max-w-2xl mx-auto text-lg transition-all duration-700 delay-200 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Sztuczna inteligencja przekształciła zakłady sportowe z zgadywania w precyzyjną naukę
          </p>
        </div>

        {/* Steps */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {steps.map((step, index) => (
            <div
              key={step.number}
              className={`group relative transition-all duration-700 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
              style={{ transitionDelay: `${index * 100 + 200}ms` }}
            >
              <div className="relative bg-glass-card rounded-2xl p-8 h-full border border-white/5 hover:border-white/10 transition-colors duration-300">
                {/* Glow */}
                <div
                  className={`absolute inset-0 rounded-2xl bg-gradient-to-r ${step.color} opacity-0 group-hover:opacity-5 transition-opacity duration-500`}
                />

                {/* Number & Icon */}
                <div className="flex items-start justify-between mb-6">
                  <div
                    className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${step.color} flex items-center justify-center`}
                  >
                    <step.icon className="w-8 h-8 text-white" />
                  </div>
                  <span className="text-5xl font-bold text-white/10">{step.number}</span>
                </div>

                {/* Content */}
                <h3 className="text-xl font-bold text-white mb-3">{step.title}</h3>
                <p className="text-gray-400 leading-relaxed">{step.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div
          className={`mt-16 text-center transition-all duration-700 delay-500 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
          }`}
        >
          <p className="text-gray-400 mb-6">
            To, co wyróżnia SportAI, to podejście ciągłego uczenia się. Każdy wynik zakładu 
            jest z powrotem wprowadzany do systemu, pozwalając naszej AI na stałe ewoluować 
            i poprawiać dokładność.
          </p>
          <a
            href="#blog"
            className="inline-flex items-center gap-2 text-violet-400 hover:text-violet-300 font-medium group"
          >
            Dowiedz się więcej w naszym blogu
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </a>
        </div>
      </div>
    </section>
  );
}
