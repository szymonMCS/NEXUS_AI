import { useEffect, useState, useRef } from 'react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { HelpCircle } from 'lucide-react';

const faqs = [
  {
    question: 'Czym jest value bet w zakładach sportowych?',
    answer:
      'Value bet to zakład, w którym kurs oferowany przez bukmachera jest wyższy niż rzeczywiste prawdopodobieństwo wystąpienia danego wyniku. Nasza AI analizuje te rozbieżności i dostarcza Ci wartościowe okazje z dodatnim oczekiwanym zyskiem.',
  },
  {
    question: 'Jak AI przewiduje wyniki sportowe?',
    answer:
      'Nasz system AI analizuje ogromne ilości danych historycznych, bieżących warunków meczowych, statystyk zawodników i innych czynników przy użyciu zaawansowanych algorytmów. To pozwala na prognozowanie wyników z wysokim stopniem dokładności.',
  },
  {
    question: 'Jak opłacalne są value bets od AI Bot?',
    answer:
      'Na podstawie rygorystycznych testów i analizy około 3000 zakładów, nasz bot konsekwentnie demonstruje skuteczność, osiągając imponujący ROI na poziomie 13.9%. To podkreśla zdolność bota do dostarczania wartościowych rekomendacji.',
  },
  {
    question: 'Zakłady z których dyscyplin sportowych obejmuje bot?',
    answer:
      'Nasz bot obejmuje popularne dyscypliny sportowe, w tym piłkę nożną, futbol amerykański, tenis, koszykówkę, hokej, baseball, rugby i krykiet. Cały czas rozszerzamy ofertę o nowe sporty.',
  },
  {
    question: 'Jak wybrano listę obsługiwanych bukmacherów?',
    answer:
      'Nasza lista obsługiwanych bukmacherów składa się z renomowanych platform znanych z niezawodności i szerokiego zakresu opcji zakładów. Współpracujemy tylko z licencjonowanymi i regulowanymi operatorami.',
  },
  {
    question: 'Jak mogę się skontaktować w przypadku pytań lub wsparcia?',
    answer:
      'Mamy dedy z zespoł wsparcia dostępny 24/7. Możesz się z nami skontaktować przez email: support@sportai.pl lub przez czat na żywo w aplikacji.',
  },
  {
    question: 'Jakie metody płatności akceptujecie?',
    answer:
      'Akceptujemy najpopularniejsze metody płatności, w tym karty Visa, Mastercard, PayPal, przelewy bankowe oraz kryptowaluty. Wszystkie transakcje są szyfrowane i bezpieczne.',
  },
  {
    question: 'Czy bot oferuje rekomendacje na zakłady na żywo?',
    answer:
      'W tej chwili bot oferuje wyłącznie rekomendacje przedmeczowe. Pracujemy jednak nad dodaniem zakładów na żywo w najbliższej przyszłości.',
  },
];

export function FAQ() {
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
    <section ref={sectionRef} className="py-24 relative">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div
            className={`inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-full mb-6 transition-all duration-700 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            <HelpCircle className="w-4 h-4 text-violet-400" />
            <span className="text-sm text-violet-400 font-medium">FAQ</span>
          </div>
          <h2
            className={`text-3xl sm:text-4xl font-bold text-white mb-4 transition-all duration-700 delay-100 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Najczęściej zadawane pytania
          </h2>
          <p
            className={`text-gray-400 transition-all duration-700 delay-200 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            Znajdź odpowiedzi na najczęstsze pytania dotyczące naszej platformy
          </p>
        </div>

        {/* Accordion */}
        <div
          className={`transition-all duration-700 delay-300 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
          }`}
        >
          <Accordion type="single" collapsible className="space-y-4">
            {faqs.map((faq, index) => (
              <AccordionItem
                key={index}
                value={`item-${index}`}
                className="bg-glass-card border border-white/5 rounded-xl px-6 data-[state=open]:border-violet-500/30 transition-colors"
              >
                <AccordionTrigger className="text-left text-white hover:text-violet-400 py-5 [&[data-state=open]>svg]:text-violet-400">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="text-gray-400 pb-5 leading-relaxed">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>
    </section>
  );
}
