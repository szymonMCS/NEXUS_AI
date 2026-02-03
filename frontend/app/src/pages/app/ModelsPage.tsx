/**
 * ModelsPage - ML Model Performance Dashboard
 * 
 * Features:
 * - Model accuracy metrics (precision/recall/calibration)
 * - A/B testing results
 * - Feature importance
 * - Model comparison
 * - Training history
 * - Prediction distribution
 */

import { useState } from 'react';
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
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  ReferenceLine,
  Cell,
} from 'recharts';
import {
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  GitCompare,
  Layers,
  Target,
  XCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Model data
const models = [
  {
    id: 'random_forest',
    name: 'Random Forest + ARA',
    version: 'v3.2.1',
    status: 'active',
    accuracy: 81.9,
    precision: 79.5,
    recall: 76.2,
    f1Score: 77.8,
    calibration: 88.4,
    roi: 15.2,
    predictions: 12450,
    lastTrained: '2026-01-25',
    dataset: 'Football-Data.co.uk',
  },
  {
    id: 'mlp',
    name: 'MLP + PCA',
    version: 'v2.8.0',
    status: 'active',
    accuracy: 86.7,
    precision: 84.2,
    recall: 82.1,
    f1Score: 83.1,
    calibration: 91.2,
    roi: 18.5,
    predictions: 8920,
    lastTrained: '2026-01-26',
    dataset: 'Football-Data.co.uk',
  },
  {
    id: 'transformer',
    name: 'Sports Transformer',
    version: 'v1.5.2',
    status: 'beta',
    accuracy: 84.3,
    precision: 82.8,
    recall: 80.5,
    f1Score: 81.6,
    calibration: 85.7,
    roi: 12.8,
    predictions: 3450,
    lastTrained: '2026-01-24',
    dataset: 'ScoreNetworkData',
  },
  {
    id: 'gnn',
    name: 'Graph Neural Network',
    version: 'v1.2.0',
    status: 'beta',
    accuracy: 79.2,
    precision: 77.4,
    recall: 75.8,
    f1Score: 76.6,
    calibration: 82.3,
    roi: 9.4,
    predictions: 2100,
    lastTrained: '2026-01-23',
    dataset: 'ScoreNetworkData',
  },
  {
    id: 'ensemble',
    name: 'Ensemble (All)',
    version: 'v4.0.0',
    status: 'production',
    accuracy: 88.2,
    precision: 86.5,
    recall: 84.2,
    f1Score: 85.3,
    calibration: 93.1,
    roi: 21.4,
    predictions: 15680,
    lastTrained: '2026-01-27',
    dataset: 'Combined',
  },
];

const abTestResults = [
  {
    metric: 'Accuracy',
    baseline: 81.5,
    challenger: 86.7,
    improvement: 6.4,
    pValue: 0.002,
    significant: true,
  },
  {
    metric: 'Precision',
    baseline: 78.2,
    challenger: 84.2,
    improvement: 7.7,
    pValue: 0.001,
    significant: true,
  },
  {
    metric: 'Recall',
    baseline: 75.4,
    challenger: 82.1,
    improvement: 8.9,
    pValue: 0.003,
    significant: true,
  },
  {
    metric: 'ROI',
    baseline: 12.8,
    challenger: 18.5,
    improvement: 44.5,
    pValue: 0.008,
    significant: true,
  },
  {
    metric: 'Calibration',
    baseline: 85.4,
    challenger: 91.2,
    improvement: 6.8,
    pValue: 0.015,
    significant: true,
  },
];

const featureImportance = [
  { feature: 'ELO Rating Diff', importance: 18.5, category: 'Ratings' },
  { feature: 'Recent Form (5)', importance: 15.2, category: 'Form' },
  { feature: 'Home Advantage', importance: 12.8, category: 'Context' },
  { feature: 'Head-to-Head', importance: 11.4, category: 'History' },
  { feature: 'Goals Scored Avg', importance: 9.6, category: 'Offense' },
  { feature: 'Goals Conceded Avg', importance: 8.9, category: 'Defense' },
  { feature: 'Possession %', importance: 7.2, category: 'Style' },
  { feature: 'Injury Count', importance: 6.5, category: 'Squad' },
  { feature: 'Rest Days', importance: 5.1, category: 'Fitness' },
  { feature: 'Weather', importance: 2.8, category: 'Context' },
];

const trainingHistory = [
  { epoch: 1, trainLoss: 0.65, valLoss: 0.68, accuracy: 72.5 },
  { epoch: 5, trainLoss: 0.48, valLoss: 0.52, accuracy: 78.2 },
  { epoch: 10, trainLoss: 0.38, valLoss: 0.42, accuracy: 82.1 },
  { epoch: 15, trainLoss: 0.32, valLoss: 0.36, accuracy: 84.5 },
  { epoch: 20, trainLoss: 0.28, valLoss: 0.31, accuracy: 86.7 },
  { epoch: 25, trainLoss: 0.25, valLoss: 0.29, accuracy: 87.2 },
  { epoch: 30, trainLoss: 0.24, valLoss: 0.28, accuracy: 87.5 },
];

const calibrationData = [
  { bin: '50-55', predicted: 52.5, actual: 51.2, count: 145 },
  { bin: '55-60', predicted: 57.5, actual: 56.8, count: 198 },
  { bin: '60-65', predicted: 62.5, actual: 63.1, count: 234 },
  { bin: '65-70', predicted: 67.5, actual: 68.4, count: 189 },
  { bin: '70-75', predicted: 72.5, actual: 71.9, count: 156 },
  { bin: '75-80', predicted: 77.5, actual: 78.2, count: 124 },
  { bin: '80-85', predicted: 82.5, actual: 81.7, count: 98 },
  { bin: '85-90', predicted: 87.5, actual: 88.1, count: 67 },
  { bin: '90-95', predicted: 92.5, actual: 93.4, count: 45 },
  { bin: '95-100', predicted: 97.5, actual: 96.8, count: 23 },
];

const predictionDistribution = [
  { prob: 0.5, count: 245, correct: 128 },
  { prob: 0.55, count: 312, correct: 178 },
  { prob: 0.6, count: 289, correct: 182 },
  { prob: 0.65, count: 234, correct: 158 },
  { prob: 0.7, count: 198, correct: 145 },
  { prob: 0.75, count: 156, correct: 122 },
  { prob: 0.8, count: 124, correct: 102 },
  { prob: 0.85, count: 98, correct: 85 },
  { prob: 0.9, count: 67, correct: 62 },
  { prob: 0.95, count: 45, correct: 43 },
];

// Helper components
const MetricCard = ({ 
  label, 
  value, 
  trend,
  trendUp = true 
}: { 
  label: string; 
  value: string | number; 
  trend?: string;
  trendUp?: boolean;
}) => (
  <div className="p-4 bg-white/5 rounded-lg">
    <div className="text-xs text-gray-500 mb-1">{label}</div>
    <div className="text-xl font-bold text-white">{value}</div>
    {trend && (
      <div className={cn(
        'text-xs mt-1',
        trendUp ? 'text-[#00ff88]' : 'text-[#ff3860]'
      )}>
        {trendUp ? '↑' : '↓'} {trend}
      </div>
    )}
  </div>
);

const ModelCard = ({ model }: { model: typeof models[0] }) => (
  <Card className={cn(
    'bg-[#0f1623]/80 border-white/10 hover:border-white/20 transition-colors',
    model.status === 'production' && 'border-[#00ff88]/30'
  )}>
    <CardContent className="p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-white">{model.name}</h3>
            <Badge variant="outline" className="text-xs border-white/20 text-gray-400">
              {model.version}
            </Badge>
          </div>
          <p className="text-xs text-gray-500 mt-1">{model.dataset}</p>
        </div>
        <Badge className={cn(
          'border-0',
          model.status === 'production' ? 'bg-[#00ff88]/20 text-[#00ff88]' :
          model.status === 'active' ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
          'bg-[#ff9500]/20 text-[#ff9500]'
        )}>
          {model.status}
        </Badge>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="text-center">
          <div className="text-lg font-bold text-[#00d4ff]">{model.accuracy}%</div>
          <div className="text-xs text-gray-500">Accuracy</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-[#00ff88]">{model.roi}%</div>
          <div className="text-xs text-gray-500">ROI</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-white">{model.predictions.toLocaleString()}</div>
          <div className="text-xs text-gray-500">Predictions</div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Precision</span>
          <span className="text-white">{model.precision}%</span>
        </div>
        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
          <div className="h-full bg-[#00d4ff] rounded-full" style={{ width: `${model.precision}%` }} />
        </div>
        
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Recall</span>
          <span className="text-white">{model.recall}%</span>
        </div>
        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
          <div className="h-full bg-[#00ff88] rounded-full" style={{ width: `${model.recall}%` }} />
        </div>
        
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Calibration</span>
          <span className="text-white">{model.calibration}%</span>
        </div>
        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
          <div className="h-full bg-[#ff9500] rounded-full" style={{ width: `${model.calibration}%` }} />
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between">
        <span className="text-xs text-gray-500">Last trained: {model.lastTrained}</span>
        <Button variant="ghost" size="sm" className="h-8 text-[#00d4ff] hover:text-[#00d4ff] hover:bg-[#00d4ff]/10">
          Details <ChevronRight className="w-4 h-4 ml-1" />
        </Button>
      </div>
    </CardContent>
  </Card>
);

export function ModelsPage() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <PageLayout
      title="Model Performance"
      description="ML model metrics, A/B testing, and feature analysis"
      breadcrumbs={[{ label: 'Model Performance' }]}
      actions={
        <Button variant="outline" className="border-white/10 text-gray-400">
          <Download className="w-4 h-4 mr-2" />
          Export Metrics
        </Button>
      }
    >
      <div className="space-y-6">
        {/* Top Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Best Accuracy" value="88.2%" trend="+2.1%" trendUp />
          <MetricCard label="Best ROI" value="21.4%" trend="+12.5%" trendUp />
          <MetricCard label="Avg Calibration" value="88.1%" trend="1.2%" trendUp />
          <MetricCard label="Total Predictions" value="42,680" trend="+3,420" trendUp />
        </div>

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="overview" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Cpu className="w-4 h-4 mr-2" />
              Models
            </TabsTrigger>
            <TabsTrigger value="abtest" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <GitCompare className="w-4 h-4 mr-2" />
              A/B Testing
            </TabsTrigger>
            <TabsTrigger value="features" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Layers className="w-4 h-4 mr-2" />
              Features
            </TabsTrigger>
            <TabsTrigger value="calibration" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Target className="w-4 h-4 mr-2" />
              Calibration
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map((model) => (
                <ModelCard key={model.id} model={model} />
              ))}
            </div>
          </TabsContent>

          {/* A/B Testing Tab */}
          <TabsContent value="abtest" className="mt-6 space-y-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <GitCompare className="w-5 h-5 text-[#00d4ff]" />
                  A/B Test Results: Random Forest vs MLP
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Statistical significance testing with p-value calculation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500">Metric</TableHead>
                      <TableHead className="text-gray-500">Baseline (RF)</TableHead>
                      <TableHead className="text-gray-500">Challenger (MLP)</TableHead>
                      <TableHead className="text-gray-500">Improvement</TableHead>
                      <TableHead className="text-gray-500">P-Value</TableHead>
                      <TableHead className="text-gray-500">Significant</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {abTestResults.map((result) => (
                      <TableRow key={result.metric} className="border-white/5 hover:bg-white/5">
                        <TableCell className="font-medium text-white">{result.metric}</TableCell>
                        <TableCell className="text-gray-400">{result.baseline}%</TableCell>
                        <TableCell className="text-[#00d4ff]">{result.challenger}%</TableCell>
                        <TableCell className="text-[#00ff88]">+{result.improvement}%</TableCell>
                        <TableCell className="font-mono text-gray-400">{result.pValue}</TableCell>
                        <TableCell>
                          {result.significant ? (
                            <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0">
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Yes
                            </Badge>
                          ) : (
                            <Badge className="bg-[#ff3860]/20 text-[#ff3860] border-0">
                              <XCircle className="w-3 h-3 mr-1" />
                              No
                            </Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Training History</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingHistory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                        <YAxis yAxisId="left" stroke="#64748b" fontSize={12} />
                        <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={12} domain={[70, 90]} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                        <Line yAxisId="left" type="monotone" dataKey="trainLoss" stroke="#ff9500" strokeWidth={2} name="Train Loss" />
                        <Line yAxisId="left" type="monotone" dataKey="valLoss" stroke="#ff3860" strokeWidth={2} strokeDasharray="5 5" name="Val Loss" />
                        <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#00ff88" strokeWidth={2} name="Accuracy" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Prediction Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={predictionDistribution}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="prob" stroke="#64748b" fontSize={12} tickFormatter={(v) => `${v}`} />
                        <YAxis stroke="#64748b" fontSize={12} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                        <Bar dataKey="count" fill="rgba(0, 212, 255, 0.5)" />
                        <Bar dataKey="correct" fill="rgba(0, 255, 136, 0.5)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Features Tab */}
          <TabsContent value="features" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Layers className="w-5 h-5 text-[#00d4ff]" />
                  Feature Importance Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={featureImportance} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
                      <XAxis type="number" stroke="#64748b" fontSize={12} />
                      <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={12} width={120} />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#0f1623',
                          border: '1px solid rgba(255,255,255,0.1)',
                        }}
                      />
                      <Bar dataKey="importance" fill="#00d4ff" radius={[0, 4, 4, 0]}>
                        {featureImportance.map((_entry, index) => (
                          <Cell key={`cell-${index}`} fill={index < 3 ? '#00ff88' : index < 6 ? '#00d4ff' : '#64748b'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Calibration Tab */}
          <TabsContent value="calibration" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-[#00d4ff]" />
                  Calibration Plot: Predicted vs Actual
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Perfect calibration follows the diagonal line
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis 
                        type="number" 
                        dataKey="predicted" 
                        name="Predicted" 
                        domain={[50, 100]} 
                        stroke="#64748b"
                        label={{ value: 'Predicted Probability', position: 'bottom', fill: '#64748b' }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="actual" 
                        name="Actual" 
                        domain={[50, 100]} 
                        stroke="#64748b"
                        label={{ value: 'Actual Win Rate', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                      />
                      <ZAxis type="number" dataKey="count" range={[50, 400]} />
                      <RechartsTooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-[#0f1623] border border-white/10 p-3 rounded-lg">
                                <p className="text-white font-medium">Bin: {data.bin}%</p>
                                <p className="text-gray-400 text-sm">Predicted: {data.predicted}%</p>
                                <p className="text-gray-400 text-sm">Actual: {data.actual}%</p>
                                <p className="text-gray-400 text-sm">Samples: {data.count}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <ReferenceLine x={50} stroke="#334155" />
                      <ReferenceLine y={50} stroke="#334155" />
                      <ReferenceLine segment={[{ x: 50, y: 50 }, { x: 100, y: 100 }]} stroke="#00d4ff" strokeDasharray="5 5" />
                      <Scatter data={calibrationData} fill="#00ff88" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default ModelsPage;
