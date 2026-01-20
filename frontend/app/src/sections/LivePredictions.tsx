import { useEffect, useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { BarChart3, Clock, Activity, Play, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import api from '@/lib/api';

interface ProgressState {
  step: string;
  progress: number;
  message: string;
}

export function LivePredictions() {
  const [isVisible, setIsVisible] = useState(false);
  const [sport, setSport] = useState<'tennis' | 'basketball'>('tennis');
  const [date, setDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progressState, setProgressState] = useState<ProgressState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
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

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);
    setAnalysisComplete(false);
    setProgressState({ step: 'starting', progress: 0, message: 'Uruchamianie analizy...' });

    try {
      // Connect WebSocket for progress updates
      const ws = api.connectWebSocket((data) => {
        if (data.type === 'progress') {
          setProgressState({
            step: data.step,
            progress: data.progress,
            message: data.message,
          });

          if (data.step === 'complete') {
            setIsAnalyzing(false);
            setAnalysisComplete(true);
            api.disconnectWebSocket();
          } else if (data.step === 'error') {
            setError(data.message);
            setIsAnalyzing(false);
            api.disconnectWebSocket();
          }
        }
      });

      // Start analysis
      await api.runAnalysis({ sport, date });
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err instanceof Error ? err.message : 'Analiza nie powiodła się');
      setIsAnalyzing(false);
      api.disconnectWebSocket();
    }
  };

  const getStepIcon = (step: string) => {
    switch (step) {
      case 'complete':
        return <CheckCircle className="w-6 h-6 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-6 h-6 text-red-400" />;
      default:
        return <Loader2 className="w-6 h-6 text-violet-400 animate-spin" />;
    }
  };

  return (
    <section ref={sectionRef} id="live-predictions" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 mb-12">
          <div>
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-full mb-4 transition-all duration-700 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              <Activity className="w-4 h-4 text-violet-400" />
              <span className="text-sm text-violet-400 font-medium">Analiza</span>
            </div>
            <h2
              className={`text-3xl sm:text-4xl font-bold text-white mb-4 transition-all duration-700 delay-100 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Uruchom analizę
            </h2>
            <p
              className={`text-gray-400 max-w-xl transition-all duration-700 delay-200 ${
                isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
              }`}
            >
              Wybierz sport i datę, a nasz system AI przeanalizuje dostępne mecze i znajdzie najlepsze value bets.
            </p>
          </div>
        </div>

        {/* Analysis Form */}
        <Card
          className={`bg-glass-card border-white/5 mb-12 transition-all duration-700 delay-300 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
          }`}
        >
          <CardContent className="p-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {/* Sport Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Sport</label>
                <Select value={sport} onValueChange={(v) => setSport(v as 'tennis' | 'basketball')}>
                  <SelectTrigger className="bg-white/5 border-white/10 text-white">
                    <SelectValue placeholder="Wybierz sport" />
                  </SelectTrigger>
                  <SelectContent className="bg-gray-900 border-white/10">
                    <SelectItem value="tennis">🎾 Tenis</SelectItem>
                    <SelectItem value="basketball">🏀 Koszykówka</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Date Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Data</label>
                <input
                  type="date"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  className="w-full h-10 px-3 rounded-md bg-white/5 border border-white/10 text-white focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
                />
              </div>

              {/* Run Button */}
              <div className="flex items-end">
                <Button
                  onClick={startAnalysis}
                  disabled={isAnalyzing}
                  className="w-full bg-gradient-primary hover:opacity-90 text-white h-10"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Analizuję...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Uruchom analizę
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Progress Indicator */}
            {progressState && (
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  {getStepIcon(progressState.step)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">{progressState.message}</span>
                      <span className="text-gray-400 text-sm">{progressState.progress}%</span>
                    </div>
                    <Progress value={progressState.progress} className="h-2" />
                  </div>
                </div>

                {/* Steps */}
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <span className={progressState.progress >= 10 ? 'text-green-400' : ''}>Fixtures</span>
                  <span>→</span>
                  <span className={progressState.progress >= 30 ? 'text-green-400' : ''}>Analiza</span>
                  <span>→</span>
                  <span className={progressState.progress >= 70 ? 'text-green-400' : ''}>Przetwarzanie</span>
                  <span>→</span>
                  <span className={progressState.progress >= 100 ? 'text-green-400' : ''}>Gotowe</span>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <p className="text-red-400">{error}</p>
              </div>
            )}

            {/* Success Message */}
            {analysisComplete && (
              <div className="mt-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <p className="text-green-400">
                  Analiza zakończona! Sprawdź sekcję Value Bets powyżej, aby zobaczyć wyniki.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card
            className={`bg-glass-card border-white/5 transition-all duration-500 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
            style={{ transitionDelay: '400ms' }}
          >
            <CardContent className="p-6">
              <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-violet-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Analiza danych</h3>
              <p className="text-gray-400 text-sm">
                System zbiera dane z wielu źródeł: Sofascore, Flashscore, newsy, kursy bukmacherów.
              </p>
            </CardContent>
          </Card>

          <Card
            className={`bg-glass-card border-white/5 transition-all duration-500 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
            style={{ transitionDelay: '500ms' }}
          >
            <CardContent className="p-6">
              <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center mb-4">
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Weryfikacja jakości</h3>
              <p className="text-gray-400 text-sm">
                Każdy mecz jest oceniany pod kątem jakości danych. Tylko te z wynikiem 45%+ są analizowane.
              </p>
            </CardContent>
          </Card>

          <Card
            className={`bg-glass-card border-white/5 transition-all duration-500 ${
              isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
            style={{ transitionDelay: '600ms' }}
          >
            <CardContent className="p-6">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4">
                <Activity className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Ranking Top 3-5</h3>
              <p className="text-gray-400 text-sm">
                Najlepsze zakłady są rankowane według formuły: edge × quality × confidence.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
