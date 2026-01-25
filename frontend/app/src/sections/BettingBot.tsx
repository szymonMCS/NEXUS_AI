import { useEffect, useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Bot, Zap, Bell, Percent, MessageCircle, ArrowRight, CheckCircle, Star } from 'lucide-react';

const features = [
  {
    icon: Bell,
    title: 'Alerty na Telegram',
    description: 'Otrzymuj natychmiastowe powiadomienia o wartościowych zakładach',
  },
  {
    icon: Zap,
    title: '100-200 zakładów dziennie',
    description: 'Nieustanny strumień sprawdzonych przewidywań',
  },
  {
    icon: Percent,
    title: '65% zniżki',
    description: 'Specjalna oferta startowa dla nowych użytkowników',
  },
];

const testimonials = [
  {
    name: 'Marek K.',
    role: 'Profesjonalny typer',
    content: 'Bot zmienił moje podejście do bettingu. 13.9% ROI mówi samo za siebie.',
    rating: 5,
  },
  {
    name: 'Anna S.',
    role: 'Początkująca',
    content: 'Nigdy nie byłam w stanie uzyskać takich wyników sama. Polecam każdemu!',
    rating: 5,
  },
];

export function BettingBot() {
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
    <section ref={sectionRef} id="bot" className="py-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-violet-600/10 rounded-full blur-3xl" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div>
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-full mb-6 transition-all duration-700 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              <Bot className="w-4 h-4 text-violet-400" />
              <span className="text-sm text-violet-400 font-medium">AI Betting Bot</span>
            </div>

            <h2
              className={`text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-6 transition-all duration-700 delay-100 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Twój osobisty
              <span className="text-gradient block">asystent AI</span>
            </h2>

            <p
              className={`text-gray-400 text-lg mb-8 transition-all duration-700 delay-200 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Dołącz do tysięcy użytkowników, którzy zaufały naszemu botowi. Otrzymuj codziennie 
              100-200 sprawdzonych value bets bezpośrednio na Telegram.
            </p>

            {/* Features */}
            <div className="space-y-4 mb-8">
              {features.map((feature, index) => (
                <div
                  key={feature.title}
                  className={`flex items-start gap-4 transition-all duration-700 ${
                    isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4'
                  }`}
                  style={{ transitionDelay: `${index * 100 + 300}ms` }}
                >
                  <div className="w-12 h-12 rounded-xl bg-gradient-primary flex items-center justify-center flex-shrink-0">
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <div className="text-white font-semibold mb-1">{feature.title}</div>
                    <div className="text-gray-400 text-sm">{feature.description}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* CTA */}
            <div
              className={`flex flex-col sm:flex-row gap-4 transition-all duration-700 delay-500 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              <Button
                size="lg"
                className="bg-gradient-primary hover:opacity-90 text-white font-semibold px-8 py-6 text-lg glow-primary-sm group"
              >
                <MessageCircle className="w-5 h-5 mr-2" />
                Dołącz teraz
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-white/20 text-white hover:bg-white/5 px-8 py-6 text-lg"
              >
                Dowiedz się więcej
              </Button>
            </div>
          </div>

          {/* Right Content - Testimonials & Stats */}
          <div className="space-y-6">
            {/* Main Card */}
            <Card
              className={`bg-glass-card border-white/5 overflow-hidden transition-all duration-700 delay-300 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              <CardContent className="p-8">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-primary flex items-center justify-center glow-primary-sm">
                    <Bot className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">SportAI Bot</div>
                    <div className="text-gray-400">@sportaibot</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-white/5 rounded-xl p-4">
                    <div className="text-3xl font-bold text-white mb-1">13.9%</div>
                    <div className="text-sm text-gray-400">Średni ROI</div>
                  </div>
                  <div className="bg-white/5 rounded-xl p-4">
                    <div className="text-3xl font-bold text-white mb-1">~3000</div>
                    <div className="text-sm text-gray-400">Przetestowane zakłady</div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-gray-300">Zweryfikowane wyniki przez niezależnych auditorów</span>
                </div>
              </CardContent>
            </Card>

            {/* Testimonials */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {testimonials.map((testimonial, index) => (
                <Card
                  key={testimonial.name}
                  className={`bg-glass-card border-white/5 transition-all duration-700 ${
                    isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                  }`}
                  style={{ transitionDelay: `${index * 100 + 400}ms` }}
                >
                  <CardContent className="p-5">
                    <div className="flex items-center gap-1 mb-3">
                      {[...Array(testimonial.rating)].map((_, i) => (
                        <Star key={i} className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                      ))}
                    </div>
                    <p className="text-gray-300 text-sm mb-3">"{testimonial.content}"</p>
                    <div>
                      <div className="text-white font-semibold text-sm">{testimonial.name}</div>
                      <div className="text-gray-500 text-xs">{testimonial.role}</div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
