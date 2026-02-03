/**
 * StatisticsPage - Team and Player Statistics
 * 
 * Features:
 * - League standings and tables
 * - Team form analysis
 * - Head-to-head statistics
 * - Player performance metrics
 * - Advanced team metrics (xG, possession, etc.)
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import {
  Calendar,
  Filter,
  Shield,
  Swords,
  Trophy,
  Users,
  TrendingUp,
  Award,
  Clock,
  Target,
  Star,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data
const leagues = [
  { id: 'epl', name: 'Premier League', country: 'England' },
  { id: 'laliga', name: 'La Liga', country: 'Spain' },
  { id: 'bundesliga', name: 'Bundesliga', country: 'Germany' },
  { id: 'seriea', name: 'Serie A', country: 'Italy' },
  { id: 'ligue1', name: 'Ligue 1', country: 'France' },
];

const standings = [
  { pos: 1, team: 'Man City', played: 24, won: 18, drawn: 4, lost: 2, gf: 56, ga: 22, gd: 34, pts: 58, form: ['W', 'W', 'D', 'W', 'W'], xg: 52.4, xga: 20.1 },
  { pos: 2, team: 'Liverpool', played: 24, won: 16, drawn: 6, lost: 2, gf: 54, ga: 24, gd: 30, pts: 54, form: ['W', 'W', 'W', 'D', 'W'], xg: 48.8, xga: 22.5 },
  { pos: 3, team: 'Arsenal', played: 24, won: 15, drawn: 5, lost: 4, gf: 48, ga: 22, gd: 26, pts: 50, form: ['L', 'W', 'W', 'D', 'W'], xg: 45.2, xga: 21.8 },
  { pos: 4, team: 'Aston Villa', played: 24, won: 14, drawn: 4, lost: 6, gf: 44, ga: 32, gd: 12, pts: 46, form: ['W', 'W', 'L', 'W', 'L'], xg: 38.9, xga: 30.2 },
  { pos: 5, team: 'Tottenham', played: 24, won: 13, drawn: 5, lost: 6, gf: 46, ga: 30, gd: 16, pts: 44, form: ['W', 'L', 'W', 'W', 'L'], xg: 42.1, xga: 31.5 },
  { pos: 6, team: 'Man United', played: 24, won: 12, drawn: 3, lost: 9, gf: 32, ga: 32, gd: 0, pts: 39, form: ['L', 'W', 'L', 'W', 'W'], xg: 35.6, xga: 34.8 },
];

const teamStats = {
  attack: [
    { metric: 'Goals', value: 56, rank: 1, avg: 32 },
    { metric: 'xG', value: 52.4, rank: 1, avg: 35.2 },
    { metric: 'Shots', value: 389, rank: 2, avg: 298 },
    { metric: 'Shot Accuracy', value: 42, rank: 3, avg: 38 },
    { metric: 'Big Chances', value: 67, rank: 1, avg: 42 },
  ],
  defense: [
    { metric: 'Goals Against', value: 22, rank: 1, avg: 32 },
    { metric: 'xGA', value: 20.1, rank: 1, avg: 35.8 },
    { metric: 'Clean Sheets', value: 12, rank: 1, avg: 7 },
    { metric: 'Tackles', value: 345, rank: 8, avg: 358 },
    { metric: 'Interceptions', value: 234, rank: 5, avg: 228 },
  ],
  possession: [
    { metric: 'Possession %', value: 64.2, rank: 1, avg: 50 },
    { metric: 'Passes', value: 15234, rank: 1, avg: 11245 },
    { metric: 'Pass Accuracy', value: 89.5, rank: 1, avg: 82.3 },
    { metric: 'Progressive Passes', value: 892, rank: 2, avg: 645 },
    { metric: 'Dribbles', value: 234, rank: 4, avg: 198 },
  ],
};

const h2hData = [
  { date: '2024-03-31', competition: 'Premier League', home: 'Arsenal', away: 'Man City', score: '0-0', result: 'D' },
  { date: '2023-10-08', competition: 'Premier League', home: 'Arsenal', away: 'Man City', score: '1-0', result: 'H' },
  { date: '2023-04-26', competition: 'Premier League', home: 'Man City', away: 'Arsenal', score: '4-1', result: 'A' },
  { date: '2023-02-15', competition: 'Premier League', home: 'Arsenal', away: 'Man City', score: '1-3', result: 'A' },
  { date: '2022-08-28', competition: 'Premier League', home: 'Man City', away: 'Arsenal', score: '2-0', result: 'A' },
];

const radarData = [
  { metric: 'Attack', teamA: 92, teamB: 78 },
  { metric: 'Defense', teamA: 88, teamB: 82 },
  { metric: 'Possession', teamA: 95, teamB: 65 },
  { metric: 'Form', teamA: 85, teamB: 75 },
  { metric: 'Set Pieces', teamA: 72, teamB: 68 },
  { metric: 'Discipline', teamA: 80, teamB: 70 },
];

// Player data
interface Player {
  name: string;
  team: string;
  position: string;
  nationality: string;
  appearances: number;
  minutes: number;
  goals: number;
  assists: number;
  rating: number;
  form: number[];
  shotsPerGame: number;
  passAccuracy: number;
  duelsWon: number;
  keyPasses: number;
}

const players: Player[] = [
  { name: 'Erling Haaland', team: 'Man City', position: 'ST', nationality: 'NOR', appearances: 22, minutes: 1872, goals: 18, assists: 5, rating: 8.2, form: [8.5, 7.8, 9.1, 8.0, 7.6], shotsPerGame: 4.2, passAccuracy: 78.5, duelsWon: 52, keyPasses: 0.8 },
  { name: 'Mohamed Salah', team: 'Liverpool', position: 'RW', nationality: 'EGY', appearances: 24, minutes: 2088, goals: 16, assists: 10, rating: 8.0, form: [7.9, 8.4, 7.6, 8.8, 8.2], shotsPerGame: 3.4, passAccuracy: 82.1, duelsWon: 48, keyPasses: 2.4 },
  { name: 'Bukayo Saka', team: 'Arsenal', position: 'RW', nationality: 'ENG', appearances: 23, minutes: 1978, goals: 12, assists: 9, rating: 7.8, form: [7.4, 8.1, 7.9, 7.2, 8.5], shotsPerGame: 2.8, passAccuracy: 84.3, duelsWon: 55, keyPasses: 2.8 },
  { name: 'Ollie Watkins', team: 'Aston Villa', position: 'ST', nationality: 'ENG', appearances: 24, minutes: 2112, goals: 14, assists: 8, rating: 7.6, form: [7.8, 7.2, 8.0, 7.5, 7.1], shotsPerGame: 2.9, passAccuracy: 76.2, duelsWon: 44, keyPasses: 1.2 },
  { name: 'Son Heung-min', team: 'Tottenham', position: 'LW', nationality: 'KOR', appearances: 22, minutes: 1848, goals: 13, assists: 6, rating: 7.7, form: [8.2, 7.0, 7.8, 7.5, 8.1], shotsPerGame: 3.1, passAccuracy: 80.5, duelsWon: 38, keyPasses: 1.9 },
  { name: 'Phil Foden', team: 'Man City', position: 'AM', nationality: 'ENG', appearances: 20, minutes: 1620, goals: 10, assists: 8, rating: 7.9, form: [8.0, 8.6, 7.4, 8.2, 7.8], shotsPerGame: 2.5, passAccuracy: 88.4, duelsWon: 42, keyPasses: 2.6 },
  { name: 'Bruno Fernandes', team: 'Man United', position: 'AM', nationality: 'POR', appearances: 24, minutes: 2136, goals: 8, assists: 11, rating: 7.5, form: [7.2, 7.8, 6.9, 7.4, 8.0], shotsPerGame: 2.2, passAccuracy: 83.8, duelsWon: 40, keyPasses: 3.1 },
  { name: 'Kevin De Bruyne', team: 'Man City', position: 'CM', nationality: 'BEL', appearances: 18, minutes: 1422, goals: 6, assists: 12, rating: 8.1, form: [8.4, 8.8, 7.9, 8.0, 8.6], shotsPerGame: 1.8, passAccuracy: 90.2, duelsWon: 45, keyPasses: 3.4 },
  { name: 'Alexander Isak', team: 'Newcastle', position: 'ST', nationality: 'SWE', appearances: 21, minutes: 1764, goals: 15, assists: 3, rating: 7.7, form: [7.9, 8.3, 7.1, 7.6, 8.0], shotsPerGame: 3.5, passAccuracy: 79.8, duelsWon: 46, keyPasses: 0.9 },
  { name: 'Martin Odegaard', team: 'Arsenal', position: 'AM', nationality: 'NOR', appearances: 19, minutes: 1596, goals: 7, assists: 9, rating: 7.9, form: [8.1, 7.6, 8.4, 7.8, 8.2], shotsPerGame: 1.9, passAccuracy: 89.1, duelsWon: 39, keyPasses: 3.2 },
  { name: 'Jarrod Bowen', team: 'West Ham', position: 'RW', nationality: 'ENG', appearances: 24, minutes: 2064, goals: 11, assists: 7, rating: 7.3, form: [7.0, 7.5, 6.8, 7.8, 7.2], shotsPerGame: 2.6, passAccuracy: 78.9, duelsWon: 50, keyPasses: 1.8 },
  { name: 'Dominik Szoboszlai', team: 'Liverpool', position: 'CM', nationality: 'HUN', appearances: 23, minutes: 1886, goals: 5, assists: 7, rating: 7.4, form: [7.6, 7.2, 7.8, 7.0, 7.5], shotsPerGame: 1.6, passAccuracy: 85.6, duelsWon: 56, keyPasses: 2.0 },
];

type PlayerSortKey = 'goals' | 'assists' | 'rating' | 'minutes' | 'keyPasses';

const playerCategories: { key: PlayerSortKey; label: string; icon: typeof Trophy }[] = [
  { key: 'goals', label: 'Top Scorers', icon: Target },
  { key: 'assists', label: 'Top Assists', icon: TrendingUp },
  { key: 'rating', label: 'Top Rated', icon: Star },
  { key: 'minutes', label: 'Minutes Leaders', icon: Clock },
  { key: 'keyPasses', label: 'Playmakers', icon: Award },
];

const topScorersChart = players
  .sort((a, b) => b.goals - a.goals)
  .slice(0, 8)
  .map(p => ({ name: p.name.split(' ').pop(), goals: p.goals, assists: p.assists, team: p.team }));

const PlayerFormBadge = ({ ratings }: { ratings: number[] }) => (
  <div className="flex items-center gap-1">
    {ratings.map((r, i) => (
      <div
        key={i}
        className={cn(
          'w-6 h-5 rounded text-[10px] font-bold flex items-center justify-center',
          r >= 8.0 ? 'bg-[#00ff88]/20 text-[#00ff88]' :
          r >= 7.0 ? 'bg-[#00d4ff]/20 text-[#00d4ff]' :
          r >= 6.0 ? 'bg-[#ff9500]/20 text-[#ff9500]' :
          'bg-[#ff3860]/20 text-[#ff3860]'
        )}
      >
        {r.toFixed(1)}
      </div>
    ))}
  </div>
);

const FormIndicator = ({ results }: { results: string[] }) => (
  <div className="flex items-center gap-1">
    {results.map((result, idx) => (
      <div
        key={idx}
        className={cn(
          'w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold',
          result === 'W' ? 'bg-[#00ff88]/20 text-[#00ff88]' :
          result === 'D' ? 'bg-[#ff9500]/20 text-[#ff9500]' :
          'bg-[#ff3860]/20 text-[#ff3860]'
        )}
      >
        {result}
      </div>
    ))}
  </div>
);

export function StatisticsPage() {
  const [selectedLeague, setSelectedLeague] = useState('epl');
  const [activeTab, setActiveTab] = useState('standings');
  const [playerSort, setPlayerSort] = useState<PlayerSortKey>('goals');

  return (
    <PageLayout
      title="Statistics"
      description="Team standings, form analysis, and head-to-head comparisons"
      breadcrumbs={[{ label: 'Statistics' }]}
    >
      <div className="space-y-6">
        {/* League Selector */}
        <Card className="bg-[#0f1623]/80 border-white/10">
          <CardContent className="p-4">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
              <Select value={selectedLeague} onValueChange={setSelectedLeague}>
                <SelectTrigger className="w-[280px] bg-white/5 border-white/10 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#0f1623] border-white/10">
                  {leagues.map((league) => (
                    <SelectItem key={league.id} value={league.id} className="text-white">
                      <div className="flex items-center gap-2">
                        <Trophy className="w-4 h-4 text-[#00d4ff]" />
                        <span>{league.name}</span>
                        <span className="text-gray-500">({league.country})</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button variant="outline" size="sm" className="border-white/10 text-gray-400">
                  <Calendar className="w-4 h-4 mr-2" />
                  Season
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="standings" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Trophy className="w-4 h-4 mr-2" />
              Standings
            </TabsTrigger>
            <TabsTrigger value="team" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Shield className="w-4 h-4 mr-2" />
              Team Analysis
            </TabsTrigger>
            <TabsTrigger value="h2h" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Swords className="w-4 h-4 mr-2" />
              H2H
            </TabsTrigger>
            <TabsTrigger value="players" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Users className="w-4 h-4 mr-2" />
              Players
            </TabsTrigger>
          </TabsList>

          {/* Standings Tab */}
          <TabsContent value="standings" className="mt-6">
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-[#00d4ff]" />
                  League Table
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500 w-12">Pos</TableHead>
                      <TableHead className="text-gray-500">Team</TableHead>
                      <TableHead className="text-gray-500 text-center">P</TableHead>
                      <TableHead className="text-gray-500 text-center">W</TableHead>
                      <TableHead className="text-gray-500 text-center">D</TableHead>
                      <TableHead className="text-gray-500 text-center">L</TableHead>
                      <TableHead className="text-gray-500 text-center">GF</TableHead>
                      <TableHead className="text-gray-500 text-center">GA</TableHead>
                      <TableHead className="text-gray-500 text-center">GD</TableHead>
                      <TableHead className="text-gray-500 text-center">Pts</TableHead>
                      <TableHead className="text-gray-500">Form</TableHead>
                      <TableHead className="text-gray-500 text-center">xG</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {standings.map((team) => (
                      <TableRow key={team.pos} className="border-white/5 hover:bg-white/5">
                        <TableCell className="font-medium text-white">{team.pos}</TableCell>
                        <TableCell className="font-medium text-white">{team.team}</TableCell>
                        <TableCell className="text-center text-gray-400">{team.played}</TableCell>
                        <TableCell className="text-center text-[#00ff88]">{team.won}</TableCell>
                        <TableCell className="text-center text-[#ff9500]">{team.drawn}</TableCell>
                        <TableCell className="text-center text-[#ff3860]">{team.lost}</TableCell>
                        <TableCell className="text-center text-gray-300">{team.gf}</TableCell>
                        <TableCell className="text-center text-gray-300">{team.ga}</TableCell>
                        <TableCell className="text-center text-white font-medium">{team.gd > 0 ? '+' : ''}{team.gd}</TableCell>
                        <TableCell className="text-center font-bold text-[#00d4ff]">{team.pts}</TableCell>
                        <TableCell><FormIndicator results={team.form} /></TableCell>
                        <TableCell className="text-center text-gray-400">{team.xg.toFixed(1)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Team Analysis Tab */}
          <TabsContent value="team" className="mt-6 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Radar Chart */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white text-base">Team Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
                        <Radar name="Man City" dataKey="teamA" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.3} />
                        <Radar name="Arsenal" dataKey="teamB" stroke="#00ff88" fill="#00ff88" fillOpacity={0.3} />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f1623',
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Attack Stats */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white text-base flex items-center gap-2">
                    <Swords className="w-4 h-4 text-[#00ff88]" />
                    Attack
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {teamStats.attack.map((stat) => (
                    <div key={stat.metric} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">{stat.metric}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-white font-medium">{stat.value}</span>
                          <Badge variant="outline" className="text-[10px] border-[#00ff88]/30 text-[#00ff88]">
                            #{stat.rank}
                          </Badge>
                        </div>
                      </div>
                      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#00ff88] rounded-full"
                          style={{ width: `${(stat.value / (stat.avg * 1.5)) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Defense Stats */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white text-base flex items-center gap-2">
                    <Shield className="w-4 h-4 text-[#00d4ff]" />
                    Defense
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {teamStats.defense.map((stat) => (
                    <div key={stat.metric} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">{stat.metric}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-white font-medium">{stat.value}</span>
                          <Badge variant="outline" className="text-[10px] border-[#00d4ff]/30 text-[#00d4ff]">
                            #{stat.rank}
                          </Badge>
                        </div>
                      </div>
                      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#00d4ff] rounded-full"
                          style={{ width: `${(stat.value / (stat.avg * 1.5)) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* H2H Tab */}
          <TabsContent value="h2h" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Head-to-Head History</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {h2hData.map((match, idx) => (
                      <div key={idx} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                        <div className="flex items-center gap-4">
                          <div className="text-xs text-gray-500 w-20">{match.date}</div>
                          <Badge variant="outline" className="text-xs border-white/20 text-gray-400">
                            {match.competition}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 flex-1 justify-center">
                          <span className="text-white font-medium">{match.home}</span>
                          <Badge className={cn(
                            'border-0',
                            match.result === 'H' ? 'bg-[#00ff88]/20 text-[#00ff88]' :
                            match.result === 'A' ? 'bg-[#ff3860]/20 text-[#ff3860]' :
                            'bg-[#ff9500]/20 text-[#ff9500]'
                          )}>
                            {match.score}
                          </Badge>
                          <span className="text-white font-medium">{match.away}</span>
                        </div>
                        <Badge variant="outline" className={cn(
                          'text-xs border-0',
                          match.result === 'H' ? 'bg-[#00ff88]/10 text-[#00ff88]' :
                          match.result === 'A' ? 'bg-[#ff3860]/10 text-[#ff3860]' :
                          'bg-[#ff9500]/10 text-[#ff9500]'
                        )}>
                          {match.result === 'H' ? 'Home Win' : match.result === 'A' ? 'Away Win' : 'Draw'}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">H2H Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center p-4 bg-white/5 rounded-lg">
                    <div className="text-3xl font-bold text-white">24</div>
                    <div className="text-sm text-gray-500">Total Matches</div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-3 bg-[#00ff88]/10 rounded-lg">
                      <div className="text-xl font-bold text-[#00ff88]">12</div>
                      <div className="text-xs text-gray-400">Home Wins</div>
                    </div>
                    <div className="p-3 bg-[#ff9500]/10 rounded-lg">
                      <div className="text-xl font-bold text-[#ff9500]">6</div>
                      <div className="text-xs text-gray-400">Draws</div>
                    </div>
                    <div className="p-3 bg-[#ff3860]/10 rounded-lg">
                      <div className="text-xl font-bold text-[#ff3860]">6</div>
                      <div className="text-xs text-gray-400">Away Wins</div>
                    </div>
                  </div>
                  <div className="pt-4 border-t border-white/10">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Avg Goals</span>
                      <span className="text-white">2.8</span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-2">
                      <span className="text-gray-400">BTTS %</span>
                      <span className="text-white">62.5%</span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-2">
                      <span className="text-gray-400">Over 2.5 %</span>
                      <span className="text-white">58.3%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Players Tab */}
          <TabsContent value="players" className="mt-6 space-y-6">
            {/* Top 3 Spotlight Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {players
                .sort((a, b) => b.rating - a.rating)
                .slice(0, 3)
                .map((player, idx) => (
                <Card key={player.name} className="bg-[#0f1623]/80 border-white/10">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <div className={cn(
                          'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold',
                          idx === 0 ? 'bg-[#ffd700]/20 text-[#ffd700]' :
                          idx === 1 ? 'bg-gray-400/20 text-gray-300' :
                          'bg-[#cd7f32]/20 text-[#cd7f32]'
                        )}>
                          #{idx + 1}
                        </div>
                        <div>
                          <div className="text-white font-semibold text-sm">{player.name}</div>
                          <div className="text-gray-500 text-xs">{player.team} · {player.position}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={cn(
                          'text-lg font-bold',
                          player.rating >= 8.0 ? 'text-[#00ff88]' : 'text-[#00d4ff]'
                        )}>
                          {player.rating.toFixed(1)}
                        </div>
                        <div className="text-gray-500 text-[10px] uppercase tracking-wider">Rating</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-4 gap-2 mb-3">
                      <div className="text-center p-2 bg-white/5 rounded">
                        <div className="text-white font-bold text-sm">{player.goals}</div>
                        <div className="text-gray-500 text-[10px]">Goals</div>
                      </div>
                      <div className="text-center p-2 bg-white/5 rounded">
                        <div className="text-white font-bold text-sm">{player.assists}</div>
                        <div className="text-gray-500 text-[10px]">Assists</div>
                      </div>
                      <div className="text-center p-2 bg-white/5 rounded">
                        <div className="text-white font-bold text-sm">{player.appearances}</div>
                        <div className="text-gray-500 text-[10px]">Apps</div>
                      </div>
                      <div className="text-center p-2 bg-white/5 rounded">
                        <div className="text-white font-bold text-sm">{player.passAccuracy.toFixed(0)}%</div>
                        <div className="text-gray-500 text-[10px]">Pass %</div>
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">Last 5 Match Ratings</div>
                      <PlayerFormBadge ratings={player.form} />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Goals & Assists Chart */}
              <Card className="lg:col-span-2 bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white text-base flex items-center gap-2">
                    <Target className="w-4 h-4 text-[#00ff88]" />
                    Goals & Assists Leaders
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={topScorersChart} layout="vertical" margin={{ left: 10, right: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                        <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} width={80} axisLine={false} tickLine={false} />
                        <RechartsTooltip
                          contentStyle={{ backgroundColor: '#0f1623', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                          itemStyle={{ color: '#fff' }}
                          labelStyle={{ color: '#94a3b8' }}
                        />
                        <Bar dataKey="goals" fill="#00ff88" radius={[0, 4, 4, 0]} name="Goals" />
                        <Bar dataKey="assists" fill="#00d4ff" radius={[0, 4, 4, 0]} name="Assists" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Quick Stats Sidebar */}
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white text-base flex items-center gap-2">
                    <Award className="w-4 h-4 text-[#ffd700]" />
                    League Leaders
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {playerCategories.map((cat) => {
                    const sorted = [...players].sort((a, b) => b[cat.key] - a[cat.key]);
                    const leader = sorted[0];
                    const Icon = cat.icon;
                    return (
                      <div key={cat.key} className="flex items-center gap-3 p-3 bg-white/5 rounded-lg">
                        <div className="p-2 bg-[#00d4ff]/10 rounded-lg">
                          <Icon className="w-4 h-4 text-[#00d4ff]" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs text-gray-500 uppercase tracking-wider">{cat.label}</div>
                          <div className="text-white text-sm font-medium truncate">{leader.name}</div>
                          <div className="text-gray-500 text-xs">{leader.team}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[#00ff88] font-bold">
                            {cat.key === 'rating' ? leader[cat.key].toFixed(1) :
                             cat.key === 'keyPasses' ? leader[cat.key].toFixed(1) :
                             leader[cat.key]}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            </div>

            {/* Full Player Table */}
            <Card className="bg-[#0f1623]/80 border-white/10">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white flex items-center gap-2">
                    <Users className="w-5 h-5 text-[#00d4ff]" />
                    Player Performance Table
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {playerCategories.slice(0, 4).map((cat) => (
                      <Button
                        key={cat.key}
                        variant="outline"
                        size="sm"
                        onClick={() => setPlayerSort(cat.key)}
                        className={cn(
                          'text-xs border-white/10',
                          playerSort === cat.key
                            ? 'bg-[#00d4ff]/10 text-[#00d4ff] border-[#00d4ff]/30'
                            : 'text-gray-400'
                        )}
                      >
                        {cat.label}
                      </Button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-gray-500 w-8">#</TableHead>
                      <TableHead className="text-gray-500">Player</TableHead>
                      <TableHead className="text-gray-500">Team</TableHead>
                      <TableHead className="text-gray-500 text-center">Pos</TableHead>
                      <TableHead className="text-gray-500 text-center">Apps</TableHead>
                      <TableHead className="text-gray-500 text-center">Min</TableHead>
                      <TableHead className="text-gray-500 text-center">Goals</TableHead>
                      <TableHead className="text-gray-500 text-center">Assists</TableHead>
                      <TableHead className="text-gray-500 text-center">G+A</TableHead>
                      <TableHead className="text-gray-500 text-center">Shots/G</TableHead>
                      <TableHead className="text-gray-500 text-center">Pass %</TableHead>
                      <TableHead className="text-gray-500 text-center">Key P</TableHead>
                      <TableHead className="text-gray-500 text-center">Rating</TableHead>
                      <TableHead className="text-gray-500">Form</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[...players]
                      .sort((a, b) => b[playerSort] - a[playerSort])
                      .map((player, idx) => (
                      <TableRow key={player.name} className="border-white/5 hover:bg-white/5">
                        <TableCell className="text-gray-500 font-medium">{idx + 1}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span className="text-white font-medium text-sm">{player.name}</span>
                            <span className="text-gray-600 text-xs">{player.nationality}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-gray-400 text-sm">{player.team}</TableCell>
                        <TableCell className="text-center">
                          <Badge variant="outline" className={cn(
                            'text-[10px] border-0',
                            player.position === 'ST' ? 'bg-[#ff3860]/10 text-[#ff3860]' :
                            player.position === 'AM' || player.position === 'CM' ? 'bg-[#00d4ff]/10 text-[#00d4ff]' :
                            'bg-[#00ff88]/10 text-[#00ff88]'
                          )}>
                            {player.position}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-center text-gray-400">{player.appearances}</TableCell>
                        <TableCell className="text-center text-gray-400">{player.minutes.toLocaleString()}</TableCell>
                        <TableCell className="text-center font-bold text-white">{player.goals}</TableCell>
                        <TableCell className="text-center text-gray-300">{player.assists}</TableCell>
                        <TableCell className="text-center">
                          <span className="text-[#00ff88] font-bold">{player.goals + player.assists}</span>
                        </TableCell>
                        <TableCell className="text-center text-gray-400">{player.shotsPerGame.toFixed(1)}</TableCell>
                        <TableCell className="text-center text-gray-400">{player.passAccuracy.toFixed(1)}%</TableCell>
                        <TableCell className="text-center text-gray-400">{player.keyPasses.toFixed(1)}</TableCell>
                        <TableCell className="text-center">
                          <span className={cn(
                            'font-bold',
                            player.rating >= 8.0 ? 'text-[#00ff88]' :
                            player.rating >= 7.5 ? 'text-[#00d4ff]' :
                            player.rating >= 7.0 ? 'text-[#ff9500]' :
                            'text-gray-400'
                          )}>
                            {player.rating.toFixed(1)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <PlayerFormBadge ratings={player.form} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default StatisticsPage;
