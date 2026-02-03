/**
 * SettingsPage - Application preferences and configuration
 * 
 * Features:
 * - Notification preferences
 * - Display settings (theme, density)
 * - Model preferences
 * - Odds format selection
 * - Export/Import data
 * - API configuration
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
// import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import {
  AlertTriangle,
  Bell,
  BookOpen,
  Database,
  Download,
  Eye,
  Globe,
  Key,
  Laptop,
  Moon,
  Palette,
  Save,
  Settings2,
  // Shield,
  Sliders,
  Sun,
  Upload,
  // User,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState('general');
  const [saved, setSaved] = useState(false);

  // General settings
  const [oddsFormat, setOddsFormat] = useState('decimal');
  const [language, setLanguage] = useState('en');
  const [timezone, setTimezone] = useState('UTC');
  const [theme, setTheme] = useState('dark');

  // Notification settings
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [valueBetAlerts, setValueBetAlerts] = useState(true);
  const [lineMovementAlerts, setLineMovementAlerts] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState([70]);

  // Model settings
  const [defaultModel, setDefaultModel] = useState('ensemble');
  const [minConfidence, setMinConfidence] = useState([60]);
  const [minEdge, setMinEdge] = useState([3]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('5');

  // Display settings
  const [compactMode, setCompactMode] = useState(false);
  const [showAnimations, setShowAnimations] = useState(true);
  const [highContrast, setHighContrast] = useState(false);
  const [fontSize, setFontSize] = useState('medium');

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <PageLayout
      title="Settings"
      description="Configure application preferences and system settings"
      breadcrumbs={[{ label: 'Settings' }]}
      actions={
        <Button 
          className="bg-[#00d4ff] text-black hover:bg-[#00d4ff]/90"
          onClick={handleSave}
        >
          <Save className="w-4 h-4 mr-2" />
          {saved ? 'Saved!' : 'Save Changes'}
        </Button>
      }
    >
      <div className="space-y-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="general" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Settings2 className="w-4 h-4 mr-2" />
              General
            </TabsTrigger>
            <TabsTrigger value="notifications" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Bell className="w-4 h-4 mr-2" />
              Notifications
            </TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Sliders className="w-4 h-4 mr-2" />
              Models
            </TabsTrigger>
            <TabsTrigger value="display" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Eye className="w-4 h-4 mr-2" />
              Display
            </TabsTrigger>
            <TabsTrigger value="data" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Database className="w-4 h-4 mr-2" />
              Data
            </TabsTrigger>
            <TabsTrigger value="api" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Key className="w-4 h-4 mr-2" />
              API
            </TabsTrigger>
          </TabsList>

          {/* General Settings */}
          <TabsContent value="general" className="space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Globe className="w-5 h-5 text-[#00d4ff]" />
                  Regional Settings
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Configure language, timezone, and number formats
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="text-gray-300">Language</Label>
                    <Select value={language} onValueChange={setLanguage}>
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#0f1623] border-white/10">
                        <SelectItem value="en" className="text-white">English</SelectItem>
                        <SelectItem value="pl" className="text-white">Polski</SelectItem>
                        <SelectItem value="de" className="text-white">Deutsch</SelectItem>
                        <SelectItem value="es" className="text-white">Español</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-gray-300">Timezone</Label>
                    <Select value={timezone} onValueChange={setTimezone}>
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#0f1623] border-white/10">
                        <SelectItem value="UTC" className="text-white">UTC</SelectItem>
                        <SelectItem value="Europe/London" className="text-white">London (GMT)</SelectItem>
                        <SelectItem value="Europe/Paris" className="text-white">Paris (CET)</SelectItem>
                        <SelectItem value="America/New_York" className="text-white">New York (ET)</SelectItem>
                        <SelectItem value="America/Los_Angeles" className="text-white">Los Angeles (PT)</SelectItem>
                        <SelectItem value="Asia/Tokyo" className="text-white">Tokyo (JST)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-gray-300">Odds Format</Label>
                  <div className="flex gap-3">
                    {[
                      { value: 'decimal', label: 'Decimal (1.85)', desc: 'European style' },
                      { value: 'fractional', label: 'Fractional (17/20)', desc: 'UK style' },
                      { value: 'american', label: 'American (-118)', desc: 'US style' },
                    ].map((format) => (
                      <button
                        key={format.value}
                        onClick={() => setOddsFormat(format.value)}
                        className={cn(
                          'flex-1 p-3 rounded-lg border text-left transition-all',
                          oddsFormat === format.value
                            ? 'border-[#00d4ff] bg-[#00d4ff]/10'
                            : 'border-white/10 bg-white/5 hover:border-white/20'
                        )}
                      >
                        <div className="text-sm font-medium text-white">{format.label}</div>
                        <div className="text-xs text-gray-500">{format.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Palette className="w-5 h-5 text-[#00d4ff]" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Label className="text-gray-300">Theme</Label>
                  <div className="flex gap-3">
                    {[
                      { value: 'dark', label: 'Dark', icon: Moon },
                      { value: 'light', label: 'Light', icon: Sun },
                      { value: 'system', label: 'System', icon: Laptop },
                    ].map((t) => (
                      <button
                        key={t.value}
                        onClick={() => setTheme(t.value)}
                        className={cn(
                          'flex items-center gap-2 px-4 py-2 rounded-lg border transition-all',
                          theme === t.value
                            ? 'border-[#00d4ff] bg-[#00d4ff]/10 text-white'
                            : 'border-white/10 bg-white/5 text-gray-400 hover:border-white/20'
                        )}
                      >
                        <t.icon className="w-4 h-4" />
                        <span className="text-sm">{t.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Notifications Settings */}
          <TabsContent value="notifications" className="space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Bell className="w-5 h-5 text-[#00d4ff]" />
                  Notification Preferences
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Choose how and when you want to be notified
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between py-3 border-b border-white/10">
                    <div>
                      <div className="text-white font-medium">Email Notifications</div>
                      <div className="text-sm text-gray-500">Receive daily summaries and alerts via email</div>
                    </div>
                    <Switch checked={emailNotifications} onCheckedChange={setEmailNotifications} />
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-white/10">
                    <div>
                      <div className="text-white font-medium">Push Notifications</div>
                      <div className="text-sm text-gray-500">Browser push notifications for important events</div>
                    </div>
                    <Switch checked={pushNotifications} onCheckedChange={setPushNotifications} />
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-white/10">
                    <div>
                      <div className="text-white font-medium">Value Bet Alerts</div>
                      <div className="text-sm text-gray-500">Notify when high-value opportunities are detected</div>
                    </div>
                    <Switch checked={valueBetAlerts} onCheckedChange={setValueBetAlerts} />
                  </div>
                  <div className="flex items-center justify-between py-3">
                    <div>
                      <div className="text-white font-medium">Line Movement Alerts</div>
                      <div className="text-sm text-gray-500">Notify on significant odds changes</div>
                    </div>
                    <Switch checked={lineMovementAlerts} onCheckedChange={setLineMovementAlerts} />
                  </div>
                </div>

                <Separator className="bg-white/10" />

                <div className="space-y-4">
                  <Label className="text-gray-300">Minimum Confidence for Alerts</Label>
                  <div className="flex items-center gap-4">
                    <Slider
                      value={confidenceThreshold}
                      onValueChange={setConfidenceThreshold}
                      min={50}
                      max={95}
                      step={5}
                      className="flex-1"
                    />
                    <span className="text-white font-mono w-12 text-right">{confidenceThreshold}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Model Settings */}
          <TabsContent value="models" className="space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-[#00d4ff]" />
                  Model Configuration
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Configure default model behavior and filtering criteria
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label className="text-gray-300">Default Model</Label>
                  <Select value={defaultModel} onValueChange={setDefaultModel}>
                    <SelectTrigger className="bg-white/5 border-white/10 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-[#0f1623] border-white/10">
                      <SelectItem value="ensemble" className="text-white">Ensemble (Recommended)</SelectItem>
                      <SelectItem value="mlp" className="text-white">MLP + PCA</SelectItem>
                      <SelectItem value="rf" className="text-white">Random Forest + ARA</SelectItem>
                      <SelectItem value="transformer" className="text-white">Sports Transformer</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Separator className="bg-white/10" />

                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-gray-300">Minimum Confidence Threshold</Label>
                      <span className="text-white font-mono">{minConfidence}%</span>
                    </div>
                    <Slider
                      value={minConfidence}
                      onValueChange={setMinConfidence}
                      min={50}
                      max={90}
                      step={5}
                    />
                    <p className="text-xs text-gray-500">Predictions below this threshold will be hidden</p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-gray-300">Minimum Edge for Value Bets</Label>
                      <span className="text-white font-mono">{minEdge}%</span>
                    </div>
                    <Slider
                      value={minEdge}
                      onValueChange={setMinEdge}
                      min={1}
                      max={10}
                      step={0.5}
                    />
                    <p className="text-xs text-gray-500">Minimum edge required to mark a bet as value</p>
                  </div>
                </div>

                <Separator className="bg-white/10" />

                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-white font-medium">Auto-refresh Data</div>
                    <div className="text-sm text-gray-500">Automatically refresh predictions and odds</div>
                  </div>
                  <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
                </div>

                {autoRefresh && (
                  <div className="space-y-2">
                    <Label className="text-gray-300">Refresh Interval</Label>
                    <Select value={refreshInterval} onValueChange={setRefreshInterval}>
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#0f1623] border-white/10">
                        <SelectItem value="1" className="text-white">Every 1 minute</SelectItem>
                        <SelectItem value="5" className="text-white">Every 5 minutes</SelectItem>
                        <SelectItem value="15" className="text-white">Every 15 minutes</SelectItem>
                        <SelectItem value="30" className="text-white">Every 30 minutes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Display Settings */}
          <TabsContent value="display" className="space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Eye className="w-5 h-5 text-[#00d4ff]" />
                  Display Options
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between py-3 border-b border-white/10">
                    <div>
                      <div className="text-white font-medium">Compact Mode</div>
                      <div className="text-sm text-gray-500">Reduce spacing for more information density</div>
                    </div>
                    <Switch checked={compactMode} onCheckedChange={setCompactMode} />
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-white/10">
                    <div>
                      <div className="text-white font-medium">Show Animations</div>
                      <div className="text-sm text-gray-500">Enable smooth transitions and effects</div>
                    </div>
                    <Switch checked={showAnimations} onCheckedChange={setShowAnimations} />
                  </div>
                  <div className="flex items-center justify-between py-3">
                    <div>
                      <div className="text-white font-medium">High Contrast</div>
                      <div className="text-sm text-gray-500">Increase contrast for better accessibility</div>
                    </div>
                    <Switch checked={highContrast} onCheckedChange={setHighContrast} />
                  </div>
                </div>

                <Separator className="bg-white/10" />

                <div className="space-y-2">
                  <Label className="text-gray-300">Font Size</Label>
                  <div className="flex gap-3">
                    {[
                      { value: 'small', label: 'Small' },
                      { value: 'medium', label: 'Medium' },
                      { value: 'large', label: 'Large' },
                    ].map((size) => (
                      <button
                        key={size.value}
                        onClick={() => setFontSize(size.value)}
                        className={cn(
                          'flex-1 py-2 px-4 rounded-lg border text-sm transition-all',
                          fontSize === size.value
                            ? 'border-[#00d4ff] bg-[#00d4ff]/10 text-white'
                            : 'border-white/10 bg-white/5 text-gray-400 hover:border-white/20'
                        )}
                      >
                        {size.label}
                      </button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Data Settings */}
          <TabsContent value="data" className="space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-[#00d4ff]" />
                  Data Management
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Export or import your data and predictions history
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="bg-white/5 border-white/10">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-4">
                        <div className="w-10 h-10 rounded-lg bg-[#00d4ff]/10 flex items-center justify-center">
                          <Download className="w-5 h-5 text-[#00d4ff]" />
                        </div>
                        <div className="flex-1">
                          <h4 className="text-white font-medium">Export Data</h4>
                          <p className="text-sm text-gray-500 mt-1">Download all predictions and history as CSV or JSON</p>
                          <div className="flex gap-2 mt-3">
                            <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                              CSV
                            </Button>
                            <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                              JSON
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white/5 border-white/10">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-4">
                        <div className="w-10 h-10 rounded-lg bg-[#00ff88]/10 flex items-center justify-center">
                          <Upload className="w-5 h-5 text-[#00ff88]" />
                        </div>
                        <div className="flex-1">
                          <h4 className="text-white font-medium">Import Data</h4>
                          <p className="text-sm text-gray-500 mt-1">Import predictions from external sources</p>
                          <Button variant="outline" size="sm" className="mt-3 border-white/10 text-gray-400">
                            Select File
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Separator className="bg-white/10" />

                <div className="p-4 bg-[#ff9500]/10 border border-[#ff9500]/20 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-[#ff9500] mt-0.5" />
                    <div>
                      <h4 className="text-white font-medium">Clear Cache</h4>
                      <p className="text-sm text-gray-400 mt-1">
                        Clear local cache and reload all data from the server. This will not delete your prediction history.
                      </p>
                      <Button variant="outline" size="sm" className="mt-3 border-[#ff9500]/30 text-[#ff9500] hover:bg-[#ff9500]/10">
                        Clear Cache
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* API Settings */}
          <TabsContent value="api" className="space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Key className="w-5 h-5 text-[#00d4ff]" />
                  API Configuration
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Manage API keys and external service integrations
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-300">API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        type="password"
                        value="sk_live_••••••••••••••••••••••••"
                        readOnly
                        className="bg-white/5 border-white/10 text-white font-mono"
                      />
                      <Button variant="outline" className="border-white/10 text-gray-400">
                        Copy
                      </Button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-3 border-b border-white/10">
                    <div>
                      <div className="text-white font-medium">Enable API Access</div>
                      <div className="text-sm text-gray-500">Allow external API calls to your account</div>
                    </div>
                    <Switch defaultChecked />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-gray-300">Rate Limit</Label>
                    <Select defaultValue="1000">
                      <SelectTrigger className="bg-white/5 border-white/10 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#0f1623] border-white/10">
                        <SelectItem value="100" className="text-white">100 requests/minute</SelectItem>
                        <SelectItem value="1000" className="text-white">1,000 requests/minute</SelectItem>
                        <SelectItem value="10000" className="text-white">10,000 requests/minute</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Separator className="bg-white/10" />

                <div className="p-4 bg-white/5 rounded-lg">
                  <h4 className="text-white font-medium mb-2">API Documentation</h4>
                  <p className="text-sm text-gray-400 mb-3">
                    Access the full API documentation to integrate with your own systems.
                  </p>
                  <Button variant="outline" className="border-white/10 text-[#00d4ff]">
                    <BookOpen className="w-4 h-4 mr-2" />
                    View Documentation
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default SettingsPage;
