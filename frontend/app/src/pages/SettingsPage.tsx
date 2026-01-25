// pages/SettingsPage.tsx
/**
 * Settings Page - Application configuration and preferences
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Settings,
  Bell,
  Shield,
  Wallet,
  Palette,
  Database,
  Key,
  Save,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Zap,
  Target,
  TrendingUp,
} from 'lucide-react';

interface SettingsState {
  notifications: {
    enabled: boolean;
    newBets: boolean;
    analysisComplete: boolean;
    priceChanges: boolean;
    emailDigest: boolean;
  };
  betting: {
    defaultStakePercent: number;
    maxStakePercent: number;
    kellyMultiplier: number;
    minEdge: number;
    minQuality: number;
    autoTrack: boolean;
  };
  display: {
    theme: 'dark' | 'light' | 'system';
    language: 'pl' | 'en';
    currency: 'PLN' | 'EUR' | 'USD';
    oddsFormat: 'decimal' | 'fractional' | 'american';
  };
  api: {
    backendUrl: string;
    autoRefresh: boolean;
    refreshInterval: number;
  };
}

const defaultSettings: SettingsState = {
  notifications: {
    enabled: true,
    newBets: true,
    analysisComplete: true,
    priceChanges: false,
    emailDigest: false,
  },
  betting: {
    defaultStakePercent: 2,
    maxStakePercent: 5,
    kellyMultiplier: 0.25,
    minEdge: 3,
    minQuality: 60,
    autoTrack: true,
  },
  display: {
    theme: 'dark',
    language: 'pl',
    currency: 'PLN',
    oddsFormat: 'decimal',
  },
  api: {
    backendUrl: 'http://localhost:8000',
    autoRefresh: true,
    refreshInterval: 30,
  },
};

export function SettingsPage() {
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const updateSettings = <K extends keyof SettingsState>(
    category: K,
    key: keyof SettingsState[K],
    value: SettingsState[K][keyof SettingsState[K]]
  ) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value,
      },
    }));
    setSaveStatus('idle');
  };

  const saveSettings = async () => {
    setIsSaving(true);
    try {
      // In production, save to backend or localStorage
      localStorage.setItem('nexus-settings', JSON.stringify(settings));
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (error) {
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  };

  const resetSettings = () => {
    setSettings(defaultSettings);
    setSaveStatus('idle');
  };

  return (
    <div className="min-h-screen bg-background pt-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-gray-600 to-gray-800 flex items-center justify-center">
                <Settings className="w-6 h-6 text-white" />
              </div>
              Ustawienia
            </h1>
            <p className="text-gray-400 mt-1">
              Konfiguracja aplikacji i preferencje użytkownika
            </p>
          </div>
          <div className="flex items-center gap-2">
            {saveStatus === 'success' && (
              <Badge className="bg-green-500/20 text-green-300">
                <CheckCircle2 className="w-3 h-3 mr-1" />
                Zapisano
              </Badge>
            )}
            <Button
              variant="outline"
              onClick={resetSettings}
              className="bg-white/5 border-white/10 text-white"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Reset
            </Button>
            <Button
              onClick={saveSettings}
              disabled={isSaving}
              className="bg-gradient-to-r from-violet-500 to-purple-600"
            >
              {isSaving ? (
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Save className="w-4 h-4 mr-2" />
              )}
              Zapisz
            </Button>
          </div>
        </div>

        <Tabs defaultValue="betting" className="space-y-6">
          <TabsList className="bg-glass-card border border-white/10 p-1 w-full flex">
            <TabsTrigger value="betting" className="flex-1 data-[state=active]:bg-violet-500/20">
              <Wallet className="w-4 h-4 mr-2" />
              Zakłady
            </TabsTrigger>
            <TabsTrigger value="notifications" className="flex-1 data-[state=active]:bg-violet-500/20">
              <Bell className="w-4 h-4 mr-2" />
              Powiadomienia
            </TabsTrigger>
            <TabsTrigger value="display" className="flex-1 data-[state=active]:bg-violet-500/20">
              <Palette className="w-4 h-4 mr-2" />
              Wygląd
            </TabsTrigger>
            <TabsTrigger value="api" className="flex-1 data-[state=active]:bg-violet-500/20">
              <Database className="w-4 h-4 mr-2" />
              API
            </TabsTrigger>
          </TabsList>

          {/* Betting Settings */}
          <TabsContent value="betting" className="space-y-6">
            <Card className="bg-glass-card border-white/5">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Wallet className="w-5 h-5 text-green-400" />
                  Ustawienia zakładów
                </CardTitle>
                <CardDescription>
                  Konfiguracja zarządzania bankrollem i stawkami
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Default Stake */}
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300">Domyślna stawka</Label>
                    <span className="text-white">{settings.betting.defaultStakePercent}% bankrolla</span>
                  </div>
                  <Slider
                    value={[settings.betting.defaultStakePercent]}
                    min={0.5}
                    max={10}
                    step={0.5}
                    onValueChange={([value]) => updateSettings('betting', 'defaultStakePercent', value)}
                    className="w-full"
                  />
                </div>

                {/* Max Stake */}
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300">Maksymalna stawka</Label>
                    <span className="text-white">{settings.betting.maxStakePercent}% bankrolla</span>
                  </div>
                  <Slider
                    value={[settings.betting.maxStakePercent]}
                    min={1}
                    max={20}
                    step={1}
                    onValueChange={([value]) => updateSettings('betting', 'maxStakePercent', value)}
                    className="w-full"
                  />
                </div>

                {/* Kelly Multiplier */}
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300 flex items-center gap-1">
                      <Target className="w-4 h-4" />
                      Mnożnik Kelly Criterion
                    </Label>
                    <span className="text-white">{(settings.betting.kellyMultiplier * 100).toFixed(0)}%</span>
                  </div>
                  <Slider
                    value={[settings.betting.kellyMultiplier]}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    onValueChange={([value]) => updateSettings('betting', 'kellyMultiplier', value)}
                    className="w-full"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Zalecane: 25% (Quarter Kelly) dla optymalnego zarządzania ryzykiem
                  </p>
                </div>

                {/* Min Edge */}
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300 flex items-center gap-1">
                      <TrendingUp className="w-4 h-4" />
                      Minimalny edge
                    </Label>
                    <span className="text-white">{settings.betting.minEdge}%</span>
                  </div>
                  <Slider
                    value={[settings.betting.minEdge]}
                    min={0}
                    max={15}
                    step={0.5}
                    onValueChange={([value]) => updateSettings('betting', 'minEdge', value)}
                    className="w-full"
                  />
                </div>

                {/* Min Quality */}
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300 flex items-center gap-1">
                      <Shield className="w-4 h-4" />
                      Minimalna jakość danych
                    </Label>
                    <span className="text-white">{settings.betting.minQuality}%</span>
                  </div>
                  <Slider
                    value={[settings.betting.minQuality]}
                    min={0}
                    max={100}
                    step={5}
                    onValueChange={([value]) => updateSettings('betting', 'minQuality', value)}
                    className="w-full"
                  />
                </div>

                {/* Auto Track */}
                <div className="flex items-center justify-between pt-4 border-t border-white/10">
                  <div>
                    <Label className="text-gray-300">Automatyczne śledzenie zakładów</Label>
                    <p className="text-xs text-gray-500">Automatycznie zapisuj wszystkie obstawione zakłady</p>
                  </div>
                  <Switch
                    checked={settings.betting.autoTrack}
                    onCheckedChange={(value) => updateSettings('betting', 'autoTrack', value)}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Notifications Settings */}
          <TabsContent value="notifications" className="space-y-6">
            <Card className="bg-glass-card border-white/5">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Bell className="w-5 h-5 text-yellow-400" />
                  Powiadomienia
                </CardTitle>
                <CardDescription>
                  Zarządzaj alertami i powiadomieniami
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-gray-300">Powiadomienia włączone</Label>
                    <p className="text-xs text-gray-500">Główny przełącznik powiadomień</p>
                  </div>
                  <Switch
                    checked={settings.notifications.enabled}
                    onCheckedChange={(value) => updateSettings('notifications', 'enabled', value)}
                  />
                </div>

                <div className="space-y-3 pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <Label className="text-gray-300">Nowe value bets</Label>
                    <Switch
                      checked={settings.notifications.newBets}
                      onCheckedChange={(value) => updateSettings('notifications', 'newBets', value)}
                      disabled={!settings.notifications.enabled}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-gray-300">Zakończenie analizy</Label>
                    <Switch
                      checked={settings.notifications.analysisComplete}
                      onCheckedChange={(value) => updateSettings('notifications', 'analysisComplete', value)}
                      disabled={!settings.notifications.enabled}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-gray-300">Zmiany kursów</Label>
                    <Switch
                      checked={settings.notifications.priceChanges}
                      onCheckedChange={(value) => updateSettings('notifications', 'priceChanges', value)}
                      disabled={!settings.notifications.enabled}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-gray-300">Codzienny email z podsumowaniem</Label>
                    <Switch
                      checked={settings.notifications.emailDigest}
                      onCheckedChange={(value) => updateSettings('notifications', 'emailDigest', value)}
                      disabled={!settings.notifications.enabled}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Display Settings */}
          <TabsContent value="display" className="space-y-6">
            <Card className="bg-glass-card border-white/5">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Palette className="w-5 h-5 text-purple-400" />
                  Wygląd i lokalizacja
                </CardTitle>
                <CardDescription>
                  Dostosuj wygląd aplikacji
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-gray-300 mb-2 block">Motyw</Label>
                    <Select
                      value={settings.display.theme}
                      onValueChange={(value) => updateSettings('display', 'theme', value as 'dark' | 'light' | 'system')}
                    >
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="dark">Ciemny</SelectItem>
                        <SelectItem value="light">Jasny</SelectItem>
                        <SelectItem value="system">Systemowy</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-gray-300 mb-2 block">Język</Label>
                    <Select
                      value={settings.display.language}
                      onValueChange={(value) => updateSettings('display', 'language', value as 'pl' | 'en')}
                    >
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pl">Polski</SelectItem>
                        <SelectItem value="en">English</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-gray-300 mb-2 block">Waluta</Label>
                    <Select
                      value={settings.display.currency}
                      onValueChange={(value) => updateSettings('display', 'currency', value as 'PLN' | 'EUR' | 'USD')}
                    >
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="PLN">PLN (zł)</SelectItem>
                        <SelectItem value="EUR">EUR (€)</SelectItem>
                        <SelectItem value="USD">USD ($)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-gray-300 mb-2 block">Format kursów</Label>
                    <Select
                      value={settings.display.oddsFormat}
                      onValueChange={(value) => updateSettings('display', 'oddsFormat', value as 'decimal' | 'fractional' | 'american')}
                    >
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="decimal">Dziesiętne (1.95)</SelectItem>
                        <SelectItem value="fractional">Ułamkowe (19/20)</SelectItem>
                        <SelectItem value="american">Amerykańskie (-105)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* API Settings */}
          <TabsContent value="api" className="space-y-6">
            <Card className="bg-glass-card border-white/5">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-blue-400" />
                  Ustawienia API
                </CardTitle>
                <CardDescription>
                  Konfiguracja połączenia z backendem
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <Label className="text-gray-300 mb-2 block">URL Backend</Label>
                  <Input
                    value={settings.api.backendUrl}
                    onChange={(e) => updateSettings('api', 'backendUrl', e.target.value)}
                    className="bg-white/5 border-white/10 text-white"
                    placeholder="http://localhost:8000"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-gray-300">Automatyczne odświeżanie</Label>
                    <p className="text-xs text-gray-500">Automatycznie pobieraj nowe dane</p>
                  </div>
                  <Switch
                    checked={settings.api.autoRefresh}
                    onCheckedChange={(value) => updateSettings('api', 'autoRefresh', value)}
                  />
                </div>

                {settings.api.autoRefresh && (
                  <div>
                    <div className="flex justify-between mb-2">
                      <Label className="text-gray-300">Interwał odświeżania</Label>
                      <span className="text-white">{settings.api.refreshInterval}s</span>
                    </div>
                    <Slider
                      value={[settings.api.refreshInterval]}
                      min={10}
                      max={120}
                      step={10}
                      onValueChange={([value]) => updateSettings('api', 'refreshInterval', value)}
                      className="w-full"
                    />
                  </div>
                )}

                <div className="pt-4 border-t border-white/10">
                  <Button
                    variant="outline"
                    className="bg-white/5 border-white/10 text-white"
                  >
                    <Zap className="w-4 h-4 mr-2" />
                    Testuj połączenie
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* API Keys Info */}
            <Card className="bg-glass-card border-white/5">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Key className="w-5 h-5 text-yellow-400" />
                  Klucze API
                </CardTitle>
                <CardDescription>
                  Status kluczy API (konfigurowane w .env)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Anthropic API</span>
                    <Badge className="bg-green-500/20 text-green-300">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Skonfigurowany
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Odds API</span>
                    <Badge className="bg-green-500/20 text-green-300">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Skonfigurowany
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Sports Data API</span>
                    <Badge className="bg-yellow-500/20 text-yellow-300">
                      <AlertCircle className="w-3 h-3 mr-1" />
                      Opcjonalny
                    </Badge>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-4">
                  Klucze API są konfigurowane w pliku .env na serwerze backend.
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default SettingsPage;
