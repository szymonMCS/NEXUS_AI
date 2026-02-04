/**
 * ModelsPage - ML Model Performance Dashboard
 *
 * Connected to:
 * - GET /api/ml/status
 * - GET /api/ml/ensemble/status
 * - GET /api/ml/ensemble/rankings/{sport}
 */

import { useState, useEffect, useCallback } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  ChevronRight,
  Cpu,
  Download,
  GitCompare,
  Layers,
  Loader2,
  Target,
  Trophy,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api, {
  type MlModelInfo,
  type MlStatusResponse,
  type EnsembleStatusResponse,
  type EnsembleRankingsResponse,
} from '@/lib/api';

// Local model display type
interface ModelDisplay {
  id: string;
  name: string;
  version: string;
  type: string;
  status: 'production' | 'active' | 'beta';
  accuracy: number | null;
  created: string | null;
  sport: string;
}

// Helper components
const MetricCard = ({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string | number;
  subtext?: string;
}) => (
  <div className="p-4 bg-white/5 rounded-lg">
    <div className="text-xs text-gray-500 mb-1">{label}</div>
    <div className="text-xl font-bold text-white">{value}</div>
    {subtext && <div className="text-xs mt-1 text-gray-400">{subtext}</div>}
  </div>
);

const ModelCard = ({ model }: { model: ModelDisplay }) => (
  <Card
    className={cn(
      'bg-[#0f1623]/80 border-white/10 hover:border-white/20 transition-colors',
      model.status === 'production' && 'border-[#00ff88]/30'
    )}
  >
    <CardContent className="p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-white">{model.name}</h3>
            <Badge
              variant="outline"
              className="text-xs border-white/20 text-gray-400"
            >
              {model.version}
            </Badge>
          </div>
          <p className="text-xs text-gray-500 mt-1 capitalize">{model.sport}</p>
        </div>
        <Badge
          className={cn(
            'border-0',
            model.status === 'production'
              ? 'bg-[#00ff88]/20 text-[#00ff88]'
              : model.status === 'active'
                ? 'bg-[#00d4ff]/20 text-[#00d4ff]'
                : 'bg-[#ff9500]/20 text-[#ff9500]'
          )}
        >
          {model.status}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="text-center">
          <div className="text-lg font-bold text-[#00d4ff]">
            {model.accuracy != null ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <div className="text-xs text-gray-500">Accuracy</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-white capitalize">{model.type}</div>
          <div className="text-xs text-gray-500">Type</div>
        </div>
      </div>

      {model.accuracy != null && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Accuracy</span>
            <span className="text-white">{(model.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#00d4ff] rounded-full"
              style={{ width: `${Math.min(model.accuracy * 100, 100)}%` }}
            />
          </div>
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between">
        <span className="text-xs text-gray-500">
          {model.created ? `Created: ${new Date(model.created).toLocaleDateString()}` : 'No date'}
        </span>
      </div>
    </CardContent>
  </Card>
);

export function ModelsPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelDisplay[]>([]);
  const [ensembleData, setEnsembleData] = useState<EnsembleStatusResponse | null>(null);
  const [rankings, setRankings] = useState<Record<string, EnsembleRankingsResponse>>({});
  const [totalModels, setTotalModels] = useState(0);
  const [ensembleSports, setEnsembleSports] = useState<string[]>([]);
  const [selectedRankingSport, setSelectedRankingSport] = useState<string>('');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [mlRes, ensembleRes] = await Promise.allSettled([
        api.getMlStatus(),
        api.getEnsembleStatus(),
      ]);

      // Process ML status → model cards
      if (mlRes.status === 'fulfilled') {
        const allModels: ModelDisplay[] = [];
        const mlStatus = mlRes.value.ml_status;
        for (const [sport, info] of Object.entries(mlStatus)) {
          for (const m of info.models) {
            allModels.push({
              id: `${sport}-${m.name}`,
              name: m.name,
              version: m.version,
              type: m.type,
              status: 'active',
              accuracy: m.accuracy,
              created: m.created,
              sport,
            });
          }
        }
        setModels(allModels);
        setTotalModels(allModels.length);
      }

      // Process ensemble status
      if (ensembleRes.status === 'fulfilled') {
        setEnsembleData(ensembleRes.value);
        const sports = Object.keys(ensembleRes.value.ensemble_status);
        setEnsembleSports(sports);

        // Mark primary models as production
        setModels((prev) => {
          const updated = [...prev];
          for (const [sport, info] of Object.entries(ensembleRes.value.ensemble_status)) {
            for (const m of updated) {
              if (m.sport === sport && m.name === info.primary_model) {
                m.status = 'production';
              }
            }
          }
          return updated;
        });

        // Fetch rankings for each sport
        if (sports.length > 0) {
          setSelectedRankingSport(sports[0]);
          const rankingResults = await Promise.allSettled(
            sports.map((s) => api.getEnsembleRankings(s))
          );
          const rankMap: Record<string, EnsembleRankingsResponse> = {};
          rankingResults.forEach((r, i) => {
            if (r.status === 'fulfilled') {
              rankMap[sports[i]] = r.value;
            }
          });
          setRankings(rankMap);
        }
      }
    } catch (e) {
      console.error('Failed to fetch model data:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Compute summary metrics
  const bestAccuracy = models.reduce<number | null>((best, m) => {
    if (m.accuracy == null) return best;
    return best == null ? m.accuracy : Math.max(best, m.accuracy);
  }, null);

  const avgAccuracy =
    models.filter((m) => m.accuracy != null).length > 0
      ? models.filter((m) => m.accuracy != null).reduce((sum, m) => sum + (m.accuracy ?? 0), 0) /
        models.filter((m) => m.accuracy != null).length
      : null;

  const ensembleCount = ensembleSports.length;

  // Rankings chart data
  const currentRanking = selectedRankingSport ? rankings[selectedRankingSport] : null;
  const rankingChartData = currentRanking
    ? Object.entries(currentRanking.rankings)
        .map(([name, score]) => ({ name, score: Math.round(score * 100) / 100 }))
        .sort((a, b) => b.score - a.score)
    : [];

  return (
    <PageLayout
      title="Model Performance"
      description="ML model metrics, ensemble status, and rankings"
      breadcrumbs={[{ label: 'Model Performance' }]}
      actions={
        <Button variant="outline" className="border-white/10 text-gray-400" disabled>
          <Download className="w-4 h-4 mr-2" />
          Export Metrics
        </Button>
      }
    >
      <div className="space-y-6">
        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading model data...
          </div>
        )}

        {/* Top Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Total Models"
            value={totalModels}
            subtext={`Across ${ensembleSports.length} sport(s)`}
          />
          <MetricCard
            label="Best Accuracy"
            value={bestAccuracy != null ? `${(bestAccuracy * 100).toFixed(1)}%` : 'N/A'}
          />
          <MetricCard
            label="Avg Accuracy"
            value={avgAccuracy != null ? `${(avgAccuracy * 100).toFixed(1)}%` : 'N/A'}
          />
          <MetricCard
            label="Ensemble Sports"
            value={ensembleCount}
            subtext={ensembleSports.join(', ') || 'None'}
          />
        </div>

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <Cpu className="w-4 h-4 mr-2" />
              Models
            </TabsTrigger>
            <TabsTrigger
              value="ensemble"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <Trophy className="w-4 h-4 mr-2" />
              Ensemble
            </TabsTrigger>
            <TabsTrigger
              value="rankings"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <Target className="w-4 h-4 mr-2" />
              Rankings
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-6">
            {models.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {models.map((model) => (
                  <ModelCard key={model.id} model={model} />
                ))}
              </div>
            ) : !loading ? (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center text-gray-500">
                  <Cpu className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No ML models found.</p>
                  <p className="text-xs mt-1">
                    Train models to see their performance metrics here.
                  </p>
                </CardContent>
              </Card>
            ) : null}
          </TabsContent>

          {/* Ensemble Tab */}
          <TabsContent value="ensemble" className="mt-6 space-y-6">
            {ensembleData && Object.keys(ensembleData.ensemble_status).length > 0 ? (
              Object.entries(ensembleData.ensemble_status).map(([sport, info]) => (
                <Card key={sport} className="bg-[#0f1623]/80 border-white/10">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2 capitalize">
                      <GitCompare className="w-5 h-5 text-[#00d4ff]" />
                      {sport} Ensemble
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                      {info.available_models} models available &middot;
                      Ensemble {info.ensemble_enabled ? 'enabled' : 'disabled'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow className="border-white/10 hover:bg-transparent">
                          <TableHead className="text-gray-500">Property</TableHead>
                          <TableHead className="text-gray-500">Value</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        <TableRow className="border-white/5 hover:bg-white/5">
                          <TableCell className="text-gray-400">Primary Model</TableCell>
                          <TableCell className="text-[#00ff88] font-medium">
                            {info.primary_model}
                          </TableCell>
                        </TableRow>
                        <TableRow className="border-white/5 hover:bg-white/5">
                          <TableCell className="text-gray-400">Available Models</TableCell>
                          <TableCell className="text-white">{info.available_models}</TableCell>
                        </TableRow>
                        <TableRow className="border-white/5 hover:bg-white/5">
                          <TableCell className="text-gray-400">Ensemble Enabled</TableCell>
                          <TableCell>
                            <Badge
                              className={cn(
                                'border-0',
                                info.ensemble_enabled
                                  ? 'bg-[#00ff88]/20 text-[#00ff88]'
                                  : 'bg-[#ff3860]/20 text-[#ff3860]'
                              )}
                            >
                              {info.ensemble_enabled ? 'Yes' : 'No'}
                            </Badge>
                          </TableCell>
                        </TableRow>
                        {Object.entries(info.model_rankings).map(([model, rank]) => (
                          <TableRow key={model} className="border-white/5 hover:bg-white/5">
                            <TableCell className="text-gray-400">Rank: {model}</TableCell>
                            <TableCell className="text-white font-mono">
                              {typeof rank === 'number' ? rank.toFixed(4) : rank}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              ))
            ) : !loading ? (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center text-gray-500">
                  <GitCompare className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No ensemble data available.</p>
                  <p className="text-xs mt-1">
                    Configure and enable ensemble models to see status here.
                  </p>
                </CardContent>
              </Card>
            ) : null}
          </TabsContent>

          {/* Rankings Tab */}
          <TabsContent value="rankings" className="mt-6 space-y-6">
            {ensembleSports.length > 0 && (
              <div className="flex gap-2">
                {ensembleSports.map((sport) => (
                  <Button
                    key={sport}
                    variant={selectedRankingSport === sport ? 'default' : 'outline'}
                    size="sm"
                    className={cn(
                      selectedRankingSport === sport
                        ? 'bg-[#00d4ff] text-black'
                        : 'border-white/10 text-gray-400'
                    )}
                    onClick={() => setSelectedRankingSport(sport)}
                  >
                    {sport.charAt(0).toUpperCase() + sport.slice(1)}
                  </Button>
                ))}
              </div>
            )}

            {rankingChartData.length > 0 ? (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Target className="w-5 h-5 text-[#00d4ff]" />
                    Model Rankings &mdash;{' '}
                    <span className="capitalize">{selectedRankingSport}</span>
                  </CardTitle>
                  <CardDescription className="text-gray-400">
                    Primary: {currentRanking?.primary_model} &middot;{' '}
                    {currentRanking?.total_models} models total
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={rankingChartData} layout="vertical">
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="rgba(255,255,255,0.1)"
                          horizontal={false}
                        />
                        <XAxis type="number" stroke="#64748b" fontSize={12} />
                        <YAxis
                          dataKey="name"
                          type="category"
                          stroke="#64748b"
                          fontSize={12}
                          width={160}
                        />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                          }}
                        />
                        <Bar dataKey="score" fill="#00d4ff" radius={[0, 4, 4, 0]}>
                          {rankingChartData.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={
                                entry.name === currentRanking?.primary_model
                                  ? '#00ff88'
                                  : index < 2
                                    ? '#00d4ff'
                                    : '#64748b'
                              }
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            ) : !loading ? (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center text-gray-500">
                  <Target className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No ranking data available.</p>
                  <p className="text-xs mt-1">
                    Select a sport with ensemble models to view rankings.
                  </p>
                </CardContent>
              </Card>
            ) : null}
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default ModelsPage;
