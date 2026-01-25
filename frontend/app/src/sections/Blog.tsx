import { useEffect, useState, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowRight, Calendar, Clock, BookOpen } from 'lucide-react';

const articles = [
  {
    id: 1,
    title: 'Jak technologia zmieniła świat zakładów sportowych',
    excerpt:
      'Odkryj, jak technologie takie jak kryptowaluty, sztuczna inteligencja i aplikacje mobilne zrewolucjonizowały branżę zakładów sportowych.',
    category: 'Technologia',
    date: '15 sty 2026',
    readTime: '5 min',
    image: 'tech',
  },
  {
    id: 2,
    title: 'AI a zakłady sportowe - kompletny przewodnik',
    excerpt:
      'Naucz się, jak wykorzystywać algorytmy SportAI do znajdowania wartości w zakładach sportowych na wiele dyscyplin.',
    category: 'Poradnik',
    date: '12 sty 2026',
    readTime: '8 min',
    image: 'ai',
  },
  {
    id: 3,
    title: 'Raport rentowności - piłka nożna',
    excerpt:
      'Zobacz szczegółową analizę wydajności naszych modeli do przewidywania meczów piłkarskich z sezonu 2025/2026.',
    category: 'Analiza',
    date: '10 sty 2026',
    readTime: '6 min',
    image: 'football',
  },
];

export function Blog() {
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
    <section ref={sectionRef} id="blog" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 mb-12">
          <div>
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-full mb-4 transition-all duration-700 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              <BookOpen className="w-4 h-4 text-violet-400" />
              <span className="text-sm text-violet-400 font-medium">Blog</span>
            </div>
            <h2
              className={`text-3xl sm:text-4xl font-bold text-white mb-4 transition-all duration-700 delay-100 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Najnowsze wskazówki i strategie
            </h2>
            <p
              className={`text-gray-400 max-w-xl transition-all duration-700 delay-200 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Odkryj zaawansowane strategie zakładów, spostrzeżenia AI i analizy rynku od naszych ekspertów.
            </p>
          </div>
          <a
            href="#"
            className={`inline-flex items-center gap-2 text-violet-400 hover:text-violet-300 font-medium group transition-all duration-700 delay-300 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Zobacz wszystkie artykuły
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </a>
        </div>

        {/* Articles Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {articles.map((article, index) => (
            <Card
              key={article.id}
              className={`group bg-glass-card border-white/5 overflow-hidden transition-all duration-500 hover:scale-[1.02] hover:border-violet-500/30 cursor-pointer ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
              style={{ transitionDelay: `${index * 100 + 200}ms` }}
            >
              {/* Image Placeholder */}
              <div className="relative h-48 bg-gradient-to-br from-violet-600/20 to-blue-600/20 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-4 left-4">
                  <Badge className="bg-violet-500/80 text-white">{article.category}</Badge>
                </div>
                {/* Decorative Elements */}
                <div className="absolute top-4 right-4 w-20 h-20 bg-white/5 rounded-full blur-2xl group-hover:bg-white/10 transition-colors" />
                <div className="absolute bottom-1/2 right-1/4 w-32 h-32 bg-violet-500/10 rounded-full blur-3xl" />
              </div>

              <CardContent className="p-6">
                {/* Meta */}
                <div className="flex items-center gap-4 text-sm text-gray-400 mb-3">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    {article.date}
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    {article.readTime}
                  </div>
                </div>

                {/* Title */}
                <h3 className="text-lg font-bold text-white mb-3 group-hover:text-violet-400 transition-colors line-clamp-2">
                  {article.title}
                </h3>

                {/* Excerpt */}
                <p className="text-gray-400 text-sm line-clamp-3 mb-4">{article.excerpt}</p>

                {/* Read More */}
                <div className="flex items-center gap-2 text-violet-400 font-medium text-sm group-hover:gap-3 transition-all">
                  Czytaj więcej
                  <ArrowRight className="w-4 h-4" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
