// components/SettingsPanel.tsx
/**
 * Settings Panel - API keys, thresholds, notifications
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import {
  Settings,
  Key,
  Bell,
  Sliders,
  Save,
  RefreshCw,
  Eye,
  EyeOff,
  CheckCircle,
} from 'lucide-react';
import { toast } from 'sonner';

interface SettingsPanelProps {
  onSave?: (settings: SettingsState) => void;
}

interface SettingsState {
  apiKeys: {
    anthropic: string;
    oddsApi: string;
    newsApi: string;
  };
  thresholds: {
    minQuality: number;
    minEdge: number;
    maxStake: number;
    minConfidence: number;
  };
  notifications: {
    emailEnabled: boolean;
    pushEnabled: boolean;
    valueBetAlerts: boolean;
    dailySummary: boolean;
    email: string;
  };
}

const defaultSettings: SettingsState = {
  apiKeys: {
    anthropic: '',
    oddsApi: '',
    newsApi: '',
  },
  thresholds: {
    minQuality: 45,
    minEdge: 3,
    maxStake: 5,
    minConfidence: 60,
  },
  notifications: {
    emailEnabled: false,
    pushEnabled: true,
    valueBetAlerts: true,
    dailySummary: true,
    email: '',
  },
};

export function SettingsPanel({ onSave }: SettingsPanelProps) {
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);

  const updateApiKey = (key: keyof SettingsState['apiKeys'], value: string) => {
    setSettings(prev => ({
      ...prev,
      apiKeys: { ...prev.apiKeys, [key]: value },
    }));
  };

  const updateThreshold = (key: keyof SettingsState['thresholds'], value: number) => {
    setSettings(prev => ({
      ...prev,
      thresholds: { ...prev.thresholds, [key]: value },
    }));
  };

  const updateNotification = (key: keyof SettingsState['notifications'], value: boolean | string) => {
    setSettings(prev => ({
      ...prev,
      notifications: { ...prev.notifications, [key]: value },
    }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      onSave?.(settings);
      toast.success('Ustawienia zapisane pomyślnie');
    } catch {
      toast.error('Nie udało się zapisać ustawień');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setSettings(defaultSettings);
    toast.info('Ustawienia przywrócone do domyślnych');
  };

  const toggleShowApiKey = (key: string) => {
    setShowApiKeys(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <Card className="bg-glass-card border-white/5">
      <CardHeader>
        <CardTitle className="text-xl font-bold text-white flex items-center gap-2">
          <Settings className="w-5 h-5 text-violet-400" />
          Ustawienia
        </CardTitle>
        <CardDescription className="text-gray-400">
          Konfiguracja API, progów analizy i powiadomień
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="api" className="space-y-4">
          <TabsList className="bg-white/5 border-white/10">
            <TabsTrigger value="api" className="data-[state=active]:bg-violet-500/20">
              <Key className="w-4 h-4 mr-1" /> API
            </TabsTrigger>
            <TabsTrigger value="thresholds" className="data-[state=active]:bg-violet-500/20">
              <Sliders className="w-4 h-4 mr-1" /> Progi
            </TabsTrigger>
            <TabsTrigger value="notifications" className="data-[state=active]:bg-violet-500/20">
              <Bell className="w-4 h-4 mr-1" /> Powiadomienia
            </TabsTrigger>
          </TabsList>

          {/* API Keys Tab */}
          <TabsContent value="api" className="space-y-4">
            <div className="space-y-4">
              {/* Anthropic API Key */}
              <div className="space-y-2">
                <Label className="text-gray-300 flex items-center gap-2">
                  Anthropic API Key
                  {settings.apiKeys.anthropic && (
                    <Badge className="bg-green-500/20 text-green-400 text-xs">
                      <CheckCircle className="w-3 h-3 mr-1" /> Skonfigurowany
                    </Badge>
                  )}
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      type={showApiKeys['anthropic'] ? 'text' : 'password'}
                      value={settings.apiKeys.anthropic}
                      onChange={(e) => updateApiKey('anthropic', e.target.value)}
                      placeholder="sk-ant-api03-..."
                      className="bg-white/5 border-white/10 text-white pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-gray-400 hover:text-white"
                      onClick={() => toggleShowApiKey('anthropic')}
                    >
                      {showApiKeys['anthropic'] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <p className="text-xs text-gray-500">Klucz API Claude do analizy AI</p>
              </div>

              {/* TheOddsAPI Key */}
              <div className="space-y-2">
                <Label className="text-gray-300 flex items-center gap-2">
                  TheOddsAPI Key
                  {settings.apiKeys.oddsApi && (
                    <Badge className="bg-green-500/20 text-green-400 text-xs">
                      <CheckCircle className="w-3 h-3 mr-1" /> Skonfigurowany
                    </Badge>
                  )}
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      type={showApiKeys['oddsApi'] ? 'text' : 'password'}
                      value={settings.apiKeys.oddsApi}
                      onChange={(e) => updateApiKey('oddsApi', e.target.value)}
                      placeholder="Klucz API..."
                      className="bg-white/5 border-white/10 text-white pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-gray-400 hover:text-white"
                      onClick={() => toggleShowApiKey('oddsApi')}
                    >
                      {showApiKeys['oddsApi'] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <p className="text-xs text-gray-500">Klucz do pobierania kursów bukmacherskich</p>
              </div>

              {/* NewsAPI Key */}
              <div className="space-y-2">
                <Label className="text-gray-300 flex items-center gap-2">
                  NewsAPI Key
                  {settings.apiKeys.newsApi && (
                    <Badge className="bg-green-500/20 text-green-400 text-xs">
                      <CheckCircle className="w-3 h-3 mr-1" /> Skonfigurowany
                    </Badge>
                  )}
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      type={showApiKeys['newsApi'] ? 'text' : 'password'}
                      value={settings.apiKeys.newsApi}
                      onChange={(e) => updateApiKey('newsApi', e.target.value)}
                      placeholder="Klucz API..."
                      className="bg-white/5 border-white/10 text-white pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-gray-400 hover:text-white"
                      onClick={() => toggleShowApiKey('newsApi')}
                    >
                      {showApiKeys['newsApi'] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <p className="text-xs text-gray-500">Klucz do agregacji newsów sportowych</p>
              </div>
            </div>
          </TabsContent>

          {/* Thresholds Tab */}
          <TabsContent value="thresholds" className="space-y-6">
            {/* Min Quality */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Minimalna jakość danych</Label>
                <span className="text-violet-400 font-medium">{settings.thresholds.minQuality}%</span>
              </div>
              <Slider
                value={[settings.thresholds.minQuality]}
                onValueChange={([v]) => updateThreshold('minQuality', v)}
                min={0}
                max={100}
                step={5}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Mecze z jakością poniżej tego progu nie będą analizowane
              </p>
            </div>

            {/* Min Edge */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Minimalny edge</Label>
                <span className="text-green-400 font-medium">{settings.thresholds.minEdge}%</span>
              </div>
              <Slider
                value={[settings.thresholds.minEdge]}
                onValueChange={([v]) => updateThreshold('minEdge', v)}
                min={0}
                max={20}
                step={0.5}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Zakłady z edge poniżej tego progu nie będą rekomendowane
              </p>
            </div>

            {/* Max Stake */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Maksymalna stawka (% bankroll)</Label>
                <span className="text-blue-400 font-medium">{settings.thresholds.maxStake}%</span>
              </div>
              <Slider
                value={[settings.thresholds.maxStake]}
                onValueChange={([v]) => updateThreshold('maxStake', v)}
                min={1}
                max={10}
                step={0.5}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Limit stawki na pojedynczy zakład (Kelly Criterion)
              </p>
            </div>

            {/* Min Confidence */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-gray-300">Minimalna pewność AI</Label>
                <span className="text-orange-400 font-medium">{settings.thresholds.minConfidence}%</span>
              </div>
              <Slider
                value={[settings.thresholds.minConfidence]}
                onValueChange={([v]) => updateThreshold('minConfidence', v)}
                min={40}
                max={90}
                step={5}
                className="w-full"
              />
              <p className="text-xs text-gray-500">
                Predykcje z pewnością poniżej tego progu nie będą uwzględniane
              </p>
            </div>
          </TabsContent>

          {/* Notifications Tab */}
          <TabsContent value="notifications" className="space-y-4">
            {/* Email Notifications */}
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <Label className="text-gray-300">Powiadomienia email</Label>
                <p className="text-xs text-gray-500">Otrzymuj raporty na email</p>
              </div>
              <Switch
                checked={settings.notifications.emailEnabled}
                onCheckedChange={(v) => updateNotification('emailEnabled', v)}
              />
            </div>

            {settings.notifications.emailEnabled && (
              <div className="space-y-2 ml-4">
                <Label className="text-gray-400">Adres email</Label>
                <Input
                  type="email"
                  value={settings.notifications.email}
                  onChange={(e) => updateNotification('email', e.target.value)}
                  placeholder="twoj@email.com"
                  className="bg-white/5 border-white/10 text-white"
                />
              </div>
            )}

            {/* Push Notifications */}
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <Label className="text-gray-300">Powiadomienia push</Label>
                <p className="text-xs text-gray-500">Powiadomienia w przeglądarce</p>
              </div>
              <Switch
                checked={settings.notifications.pushEnabled}
                onCheckedChange={(v) => updateNotification('pushEnabled', v)}
              />
            </div>

            {/* Value Bet Alerts */}
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <Label className="text-gray-300">Alerty Value Bets</Label>
                <p className="text-xs text-gray-500">Powiadomienia o nowych value bets</p>
              </div>
              <Switch
                checked={settings.notifications.valueBetAlerts}
                onCheckedChange={(v) => updateNotification('valueBetAlerts', v)}
              />
            </div>

            {/* Daily Summary */}
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <Label className="text-gray-300">Dzienny raport</Label>
                <p className="text-xs text-gray-500">Podsumowanie dnia o 20:00</p>
              </div>
              <Switch
                checked={settings.notifications.dailySummary}
                onCheckedChange={(v) => updateNotification('dailySummary', v)}
              />
            </div>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="flex gap-3 mt-6 pt-4 border-t border-white/10">
          <Button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 bg-gradient-primary hover:opacity-90"
          >
            {saving ? (
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-2" />
            )}
            Zapisz ustawienia
          </Button>
          <Button
            variant="outline"
            onClick={handleReset}
            className="bg-white/5 border-white/10 text-white hover:bg-white/10"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Resetuj
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default SettingsPanel;
