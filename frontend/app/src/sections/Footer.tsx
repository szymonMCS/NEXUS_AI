import { Zap, Mail, Twitter, Github, Linkedin } from 'lucide-react';

const footerLinks = {
  product: {
    title: 'Produkt',
    links: [
      { label: 'Przewidywania', href: '#predictions' },
      { label: 'Value Bets', href: '#value-bets' },
      { label: 'AI Bot', href: '#bot' },
      { label: 'Cennik', href: '#' },
    ],
  },
  company: {
    title: 'Firma',
    links: [
      { label: 'O nas', href: '#' },
      { label: 'Blog', href: '#blog' },
      { label: 'Kariera', href: '#' },
      { label: 'Kontakt', href: '#' },
    ],
  },
  legal: {
    title: 'Prawne',
    links: [
      { label: 'Regulamin', href: '#' },
      { label: 'Polityka prywatności', href: '#' },
      { label: 'Cookies', href: '#' },
      { label: 'Nota prawna', href: '#' },
    ],
  },
  support: {
    title: 'Wsparcie',
    links: [
      { label: 'Centrum pomocy', href: '#' },
      { label: 'FAQ', href: '#' },
      { label: 'Status', href: '#' },
      { label: 'API', href: '#' },
    ],
  },
};

const socialLinks = [
  { icon: Twitter, href: '#', label: 'Twitter' },
  { icon: Github, href: '#', label: 'GitHub' },
  { icon: Linkedin, href: '#', label: 'LinkedIn' },
  { icon: Mail, href: 'mailto:support@sportai.pl', label: 'Email' },
];

export function Footer() {
  const scrollToSection = (href: string) => {
    if (href.startsWith('#') && href.length > 1) {
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <footer className="relative pt-24 pb-8 border-t border-white/5">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8 mb-16">
          {/* Brand */}
          <div className="col-span-2">
            <a href="#home" className="flex items-center gap-2 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white">
                Sport<span className="text-gradient">AI</span>
              </span>
            </a>
            <p className="text-gray-400 text-sm mb-6 max-w-xs">
              Najdokładniejsze przewidywania sportowe napędzane zaawansowaną sztuczną inteligencją.
            </p>
            {/* Social Links */}
            <div className="flex items-center gap-4">
              {socialLinks.map((social) => (
                <a
                  key={social.label}
                  href={social.href}
                  aria-label={social.label}
                  className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                >
                  <social.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Links */}
          {Object.values(footerLinks).map((section) => (
            <div key={section.title}>
              <h4 className="text-white font-semibold mb-4">{section.title}</h4>
              <ul className="space-y-3">
                {section.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      onClick={(e) => {
                        if (link.href.startsWith('#') && link.href.length > 1) {
                          e.preventDefault();
                          scrollToSection(link.href);
                        }
                      }}
                      className="text-gray-400 text-sm hover:text-white transition-colors"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-500 text-sm">
            © 2026 SportAI. Wszelkie prawa zastrzeżone.
          </p>
          <p className="text-gray-500 text-sm text-center md:text-right">
            Nasza zawartość nie jest przeznaczona dla osób poniżej 18 roku życia. 
            Zawartość na tej stronie nie jest poradą i powinna być używana wyłącznie jako informacja.
          </p>
        </div>
      </div>
    </footer>
  );
}
