/**
 * DocumentationPage - API Reference and User Guides
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  BookOpen,
  Code,
  Copy,
  Key,
  Server,
  Shield,
  Terminal,
  Webhook,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const endpoints = [
  { method: 'GET', path: '/api/v1/predictions', description: 'Get all active predictions', auth: true },
  { method: 'GET', path: '/api/v1/predictions/{id}', description: 'Get single prediction details', auth: true },
  { method: 'GET', path: '/api/v1/matches/live', description: 'Get live matches with real-time scores', auth: true },
  { method: 'GET', path: '/api/v1/handicaps/{match_id}', description: 'Get handicap lines for a match', auth: true },
  { method: 'GET', path: '/api/v1/models/performance', description: 'Get model performance metrics', auth: true },
  { method: 'POST', path: '/api/v1/predictions/custom', description: 'Generate custom prediction', auth: true },
];

const curlExample = `curl -X GET "https://api.nexus.ai/v1/predictions" \\
  -H "Authorization: Bearer YOUR_API_KEY"`;

const jsExample = `const response = await fetch('https://api.nexus.ai/v1/predictions', {
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY'
  }
});
const predictions = await response.json();`;

const pythonExample = `import requests
headers = {'Authorization': 'Bearer YOUR_API_KEY'}
response = requests.get('https://api.nexus.ai/v1/predictions', headers=headers)
predictions = response.json()`;

const guides = [
  { title: 'Getting Started', description: 'Learn the basics of integrating with NEXUS AI API', icon: BookOpen },
  { title: 'Authentication', description: 'Understanding API keys and security best practices', icon: Shield },
  { title: 'Webhooks', description: 'Receive real-time updates for predictions', icon: Webhook },
];

const MethodBadge = ({ method }: { method: string }) => {
  const colors: Record<string, string> = {
    GET: 'bg-[#00d4ff]/20 text-[#00d4ff]',
    POST: 'bg-[#00ff88]/20 text-[#00ff88]',
    PUT: 'bg-[#ff9500]/20 text-[#ff9500]',
    DELETE: 'bg-[#ff3860]/20 text-[#ff3860]',
  };
  return (
    <Badge className={cn('border-0 font-mono text-xs', colors[method])}>
      {method}
    </Badge>
  );
};

export function DocumentationPage() {
  const [activeTab, setActiveTab] = useState('api');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLang, setSelectedLang] = useState('curl');

  const filteredEndpoints = endpoints.filter(
    ep => ep.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
          ep.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getExample = () => {
    if (selectedLang === 'curl') return curlExample;
    if (selectedLang === 'javascript') return jsExample;
    return pythonExample;
  };

  return (
    <PageLayout
      title="Documentation"
      description="API reference, guides, and integration examples"
      breadcrumbs={[{ label: 'Documentation' }]}
    >
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {guides.map((guide, idx) => {
            const Icon = guide.icon;
            return (
              <Card key={idx} className="bg-[#0f1623]/80 border-white/10 hover:border-white/20 transition-colors cursor-pointer group">
                <CardContent className="p-5">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-[#00d4ff]/10 flex items-center justify-center group-hover:bg-[#00d4ff]/20 transition-colors">
                      <Icon className="w-5 h-5 text-[#00d4ff]" />
                    </div>
                    <div>
                      <h3 className="font-medium text-white">{guide.title}</h3>
                      <p className="text-sm text-gray-500 mt-1">{guide.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="api" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Server className="w-4 h-4 mr-2" />
              API Reference
            </TabsTrigger>
            <TabsTrigger value="examples" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Code className="w-4 h-4 mr-2" />
              Code Examples
            </TabsTrigger>
            <TabsTrigger value="models" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Terminal className="w-4 h-4 mr-2" />
              Model Guide
            </TabsTrigger>
          </TabsList>

          <TabsContent value="api" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white">API Endpoints</CardTitle>
                  <div className="relative w-64">
                    <Server className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <Input
                      placeholder="Search endpoints..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 bg-white/5 border-white/10 text-white"
                    />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500 w-20">Method</TableHead>
                      <TableHead className="text-gray-500">Endpoint</TableHead>
                      <TableHead className="text-gray-500">Description</TableHead>
                      <TableHead className="text-gray-500 w-24">Auth</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredEndpoints.map((ep, idx) => (
                      <TableRow key={idx} className="border-white/5 hover:bg-white/5">
                        <TableCell><MethodBadge method={ep.method} /></TableCell>
                        <TableCell className="font-mono text-[#00d4ff]">{ep.path}</TableCell>
                        <TableCell className="text-gray-300">{ep.description}</TableCell>
                        <TableCell>
                          {ep.auth ? (
                            <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-xs">Required</Badge>
                          ) : (
                            <Badge variant="outline" className="border-white/10 text-gray-500 text-xs">Public</Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="examples" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white">Code Examples</CardTitle>
                  <div className="flex gap-2">
                    {['curl', 'javascript', 'python'].map((lang) => (
                      <Button
                        key={lang}
                        variant={selectedLang === lang ? 'default' : 'outline'}
                        size="sm"
                        className={selectedLang === lang ? 'bg-[#00d4ff] text-black' : 'border-white/10 text-gray-400'}
                        onClick={() => setSelectedLang(lang)}
                      >
                        {lang}
                      </Button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <div className="flex items-center justify-between px-4 py-2 bg-black/40 border-b border-white/10 rounded-t-lg">
                    <span className="text-xs text-gray-500">{selectedLang}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs text-gray-400 hover:text-white"
                      onClick={() => navigator.clipboard.writeText(getExample())}
                    >
                      <Copy className="w-3 h-3 mr-1" /> Copy
                    </Button>
                  </div>
                  <pre className="p-4 bg-black/30 rounded-b-lg overflow-x-auto">
                    <code className="text-sm text-gray-300 font-mono">{getExample()}</code>
                  </pre>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="models" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Available Models</CardTitle>
                  <CardDescription className="text-gray-400">ML models for predictions</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    { name: 'Ensemble', desc: 'Combined model with best overall performance', accuracy: '88.2%' },
                    { name: 'MLP + PCA', desc: 'Neural network with dimensionality reduction', accuracy: '86.7%' },
                    { name: 'Random Forest + ARA', desc: 'Tree-based model with adaptive regularization', accuracy: '81.9%' },
                    { name: 'Sports Transformer', desc: 'Attention-based architecture', accuracy: '84.3%' },
                  ].map((model, idx) => (
                    <div key={idx} className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-white">{model.name}</span>
                        <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0">{model.accuracy}</Badge>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{model.desc}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Authentication</CardTitle>
                  <CardDescription className="text-gray-400">API key requirements</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Key className="w-4 h-4 text-[#00d4ff]" />
                      <span className="font-medium text-white">API Key Header</span>
                    </div>
                    <code className="text-sm text-gray-400">Authorization: Bearer YOUR_API_KEY</code>
                  </div>
                  <div className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-4 h-4 text-[#00ff88]" />
                      <span className="font-medium text-white">Rate Limits</span>
                    </div>
                    <p className="text-sm text-gray-400">1000 requests/minute for standard plans</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default DocumentationPage;
