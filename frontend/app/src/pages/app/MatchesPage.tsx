/**
 * MatchesPage - Comprehensive match browser and analysis center
 *
 * Connected to real API:
 * - /api/matches for match data
 * - /api/predictions for value bet predictions
 */

import { useState, useEffect, useCallback } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Activity,
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clock,
  Filter,
  Globe,
  Loader2,
  Search,
  Star,
  Target,
  Trophy,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import api, { type ApiMatch, type ValueBet } from '@/lib/api';

// Types
interface Match {
  id: string;
  sport: string;
  league: string;
  leagueCountry: string;
  homeTeam: string;
  awayTeam: string;
  startTime: string;
  status: 'scheduled' | 'live' | 'finished' | 'postponed';
  homeScore?: number;
  awayScore?: number;
  minute?: string;
  odds?: { home: number; draw?: number; away: number };
  prediction?: {
    result: string;
    confidence: number;
    isValue: boolean;
    edge?: number;
  };
  importance: 'low' | 'medium' | 'high';
}

interface LeagueGroup {
  league: string;
  country: string;
  sport: string;
  matches: Match[];
}

// Map API match + optional value bet to local Match
function mapApiMatch(m: ApiMatch, sport: string, valueBet?: ValueBet): Match {
  const startTime = m.start_time
    ? new Date(m.start_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
    : '';

  const status: Match['status'] = m.is_live ? 'live' : m.is_finished ? 'finished' : 'scheduled';

  let prediction: Match['prediction'] | undefined;
  let odds: Match['odds'] | undefined;

  if (valueBet) {
    prediction = {
      result: valueBet.selection,
      confidence: Math.round(valueBet.confidence * 100),
      isValue: valueBet.edge > 0.03,
      edge: Math.round(valueBet.edge * 100 * 10) / 10,
    };
    odds = { home: valueBet.odds, away: valueBet.odds };
  }

  const importance: Match['importance'] =
    (m.quality_score ?? 0) >= 75 ? 'high' : (m.quality_score ?? 0) >= 50 ? 'medium' : 'low';

  return {
    id: String(m.id),
    sport: sport.charAt(0).toUpperCase() + sport.slice(1),
    league: m.league || 'Unknown',
    leagueCountry: '',
    homeTeam: m.home_team,
    awayTeam: m.away_team,
    startTime,
    status,
    homeScore: m.home_score ?? undefined,
    awayScore: m.away_score ?? undefined,
    odds,
    prediction,
    importance,
  };
}

// Helper: group matches by league
function groupByLeague(matches: Match[]): LeagueGroup[] {
  const groups: Record<string, LeagueGroup> = {};
  matches.forEach((m) => {
    const key = `${m.sport}-${m.league}`;
    if (!groups[key]) {
      groups[key] = { league: m.league, country: m.leagueCountry, sport: m.sport, matches: [] };
    }
    groups[key].matches.push(m);
  });
  return Object.values(groups).sort((a, b) => {
    const aHasLive = a.matches.some((m) => m.status === 'live') ? 1 : 0;
    const bHasLive = b.matches.some((m) => m.status === 'live') ? 1 : 0;
    if (bHasLive !== aHasLive) return bHasLive - aHasLive;
    return a.league.localeCompare(b.league);
  });
}

// Sub-components
const StatusIndicator = ({ status, minute }: { status: Match['status']; minute?: string }) => {
  switch (status) {
    case 'live':
      return (
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#ff3860] animate-pulse" />
          <span className="text-sm font-medium text-[#ff3860]">{minute || 'LIVE'}</span>
        </div>
      );
    case 'finished':
      return <Badge variant="outline" className="text-xs border-white/20 text-gray-500">FT</Badge>;
    case 'postponed':
      return <Badge variant="outline" className="text-xs border-[#ff9500]/30 text-[#ff9500]">PPD</Badge>;
    default:
      return <span className="text-sm text-gray-400">{minute}</span>;
  }
};

const MatchRow = ({
  match,
  expanded,
  onToggle,
}: {
  match: Match;
  expanded: boolean;
  onToggle: () => void;
}) => {
  const isLive = match.status === 'live';
  const isFinished = match.status === 'finished';

  return (
    <div
      className={cn(
        'border-b border-white/5 last:border-0 transition-colors',
        isLive && 'bg-[#ff3860]/5',
        expanded && 'bg-white/[0.02]'
      )}
    >
      {/* Main row */}
      <div
        className="flex items-center gap-4 px-4 py-3 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={onToggle}
      >
        {/* Time */}
        <div className="w-16 shrink-0">
          {isLive ? (
            <StatusIndicator status="live" minute={match.minute} />
          ) : isFinished ? (
            <StatusIndicator status="finished" />
          ) : (
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-gray-600" />
              <span className="text-sm text-gray-400">{match.startTime}</span>
            </div>
          )}
        </div>

        {/* Teams & Score */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <span className={cn('text-sm font-medium truncate', isLive || isFinished ? 'text-white' : 'text-gray-300')}>
                  {match.homeTeam}
                </span>
                {(isLive || isFinished) && match.homeScore !== undefined && (
                  <span className={cn('text-lg font-bold tabular-nums w-6 text-right', isLive ? 'text-white' : 'text-gray-300')}>
                    {match.homeScore}
                  </span>
                )}
              </div>
              <div className="flex items-center justify-between gap-2 mt-0.5">
                <span className={cn('text-sm font-medium truncate', isLive || isFinished ? 'text-white' : 'text-gray-300')}>
                  {match.awayTeam}
                </span>
                {(isLive || isFinished) && match.awayScore !== undefined && (
                  <span className={cn('text-lg font-bold tabular-nums w-6 text-right', isLive ? 'text-white' : 'text-gray-300')}>
                    {match.awayScore}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Odds */}
        {match.odds && (
          <div className="hidden md:flex items-center gap-2 shrink-0">
            <div className="px-2 py-1 bg-white/5 rounded text-xs font-mono text-gray-300 w-12 text-center">
              {match.odds.home.toFixed(2)}
            </div>
            {match.odds.draw !== undefined && (
              <div className="px-2 py-1 bg-white/5 rounded text-xs font-mono text-gray-300 w-12 text-center">
                {match.odds.draw.toFixed(2)}
              </div>
            )}
            <div className="px-2 py-1 bg-white/5 rounded text-xs font-mono text-gray-300 w-12 text-center">
              {match.odds.away.toFixed(2)}
            </div>
          </div>
        )}

        {/* Prediction */}
        <div className="hidden lg:flex items-center gap-2 w-40 shrink-0">
          {match.prediction && (
            <>
              <div className="flex-1">
                <div className="text-xs text-[#00d4ff]">{match.prediction.result}</div>
                <div className="flex items-center gap-1 mt-0.5">
                  <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={cn(
                        'h-full rounded-full',
                        match.prediction.confidence >= 75 ? 'bg-[#00ff88]' :
                        match.prediction.confidence >= 60 ? 'bg-[#00d4ff]' : 'bg-[#ff9500]'
                      )}
                      style={{ width: `${match.prediction.confidence}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-gray-500 w-7">{match.prediction.confidence}%</span>
                </div>
              </div>
              {match.prediction.isValue && (
                <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-[10px] px-1.5 py-0">V</Badge>
              )}
            </>
          )}
        </div>

        {/* Importance */}
        <div className="hidden xl:block w-6 shrink-0">
          {match.importance === 'high' && <Star className="w-4 h-4 text-[#ff9500]" />}
        </div>

        {/* Expand */}
        <div className="w-5 shrink-0">
          {expanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-white/5">
          <div className="pt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Prediction Details */}
            {match.prediction && (
              <div className="space-y-3">
                <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                  <Target className="w-3.5 h-3.5" />
                  Prediction
                </h4>
                <div className="p-3 bg-white/5 rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Result</span>
                    <span className="text-sm text-[#00d4ff] font-medium">{match.prediction.result}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Confidence</span>
                    <span className={cn('text-sm font-medium',
                      match.prediction.confidence >= 75 ? 'text-[#00ff88]' :
                      match.prediction.confidence >= 60 ? 'text-[#00d4ff]' : 'text-[#ff9500]'
                    )}>
                      {match.prediction.confidence}%
                    </span>
                  </div>
                  {match.prediction.edge !== undefined && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Edge</span>
                      <Badge className={cn('border-0 text-xs',
                        match.prediction.edge >= 3 ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#00d4ff]/20 text-[#00d4ff]'
                      )}>
                        +{match.prediction.edge}%
                      </Badge>
                    </div>
                  )}
                  {match.prediction.isValue && (
                    <div className="pt-2 border-t border-white/10 flex items-center gap-2">
                      <Zap className="w-3.5 h-3.5 text-[#00ff88]" />
                      <span className="text-xs text-[#00ff88]">Value bet identified</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Match Info */}
            <div className="space-y-3">
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5" />
                Match Info
              </h4>
              <div className="p-3 bg-white/5 rounded-lg space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Competition</span>
                  <span className="text-white">{match.league}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Kick-off</span>
                  <span className="text-white">{match.startTime}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Sport</span>
                  <span className="text-white">{match.sport}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export function MatchesPage() {
  const [activeSport, setActiveSport] = useState('all');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [dateOffset, setDateOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [matches, setMatches] = useState<Match[]>([]);

  // Date navigation
  const currentDate = new Date();
  currentDate.setDate(currentDate.getDate() + dateOffset);
  const dateStr = currentDate.toISOString().split('T')[0];
  const dateLabel =
    dateOffset === 0 ? 'Today' :
    dateOffset === 1 ? 'Tomorrow' :
    dateOffset === -1 ? 'Yesterday' :
    currentDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

  const fetchMatches = useCallback(async () => {
    setLoading(true);
    try {
      const sports = activeSport === 'all'
        ? ['tennis', 'basketball', 'football']
        : [activeSport];

      const results = await Promise.allSettled(
        sports.flatMap(sport => [
          api.getMatches(sport, dateStr),
          api.getPredictions(sport, dateStr),
        ])
      );

      const allMatches: Match[] = [];

      for (let i = 0; i < sports.length; i++) {
        const matchResult = results[i * 2];
        const predResult = results[i * 2 + 1];

        const apiMatches = matchResult.status === 'fulfilled' ? matchResult.value.matches : [];
        const valueBets = predResult.status === 'fulfilled'
          ? (predResult.value as { value_bets?: ValueBet[] }).value_bets || []
          : [];

        for (const m of apiMatches) {
          const matchName = `${m.home_team} vs ${m.away_team}`;
          const vb = valueBets.find(b =>
            b.match.toLowerCase() === matchName.toLowerCase() ||
            b.match.toLowerCase().includes(m.home_team.toLowerCase())
          );
          allMatches.push(mapApiMatch(m, sports[i], vb));
        }
      }

      setMatches(allMatches);
    } catch (e) {
      console.error('Failed to fetch matches:', e);
    } finally {
      setLoading(false);
    }
  }, [activeSport, dateStr]);

  useEffect(() => {
    fetchMatches();
  }, [fetchMatches]);

  // Filter matches
  const filteredMatches = matches.filter((m) => {
    if (activeSport !== 'all' && m.sport.toLowerCase() !== activeSport) return false;
    if (statusFilter === 'live' && m.status !== 'live') return false;
    if (statusFilter === 'scheduled' && m.status !== 'scheduled') return false;
    if (statusFilter === 'finished' && m.status !== 'finished') return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return m.homeTeam.toLowerCase().includes(q) || m.awayTeam.toLowerCase().includes(q) || m.league.toLowerCase().includes(q);
    }
    return true;
  });

  const leagueGroups = groupByLeague(filteredMatches);

  // Stats
  const totalMatches = matches.length;
  const liveMatches = matches.filter((m) => m.status === 'live').length;
  const valueMatches = matches.filter((m) => m.prediction?.isValue).length;
  const sports = [...new Set(matches.map((m) => m.sport))];

  return (
    <PageLayout
      title="Matches"
      description={`All matches for ${dateLabel} across leagues and sports`}
      breadcrumbs={[{ label: 'Matches' }]}
    >
      <div className="space-y-6">
        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading matches...
          </div>
        )}

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Total Matches</p>
                  <p className="text-2xl font-bold text-white">{totalMatches}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
                  <Calendar className="w-5 h-5 text-[#00d4ff]" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Live Now</p>
                  <p className="text-2xl font-bold text-[#ff3860]">{liveMatches}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#ff3860]/10 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-[#ff3860]" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Value Opportunities</p>
                  <p className="text-2xl font-bold text-[#00ff88]">{valueMatches}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#00ff88]/10 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-[#00ff88]" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-[#0f1623]/80 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500">Sports Covered</p>
                  <p className="text-2xl font-bold text-white">{sports.length}</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center">
                  <Trophy className="w-5 h-5 text-[#00d4ff]" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Date Navigation + Filters */}
        <Card className="bg-[#0f1623]/80 border-white/10">
          <CardContent className="p-4">
            <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
              <div className="flex items-center gap-3">
                <Button variant="outline" size="icon" className="h-8 w-8 border-white/10 text-gray-400" onClick={() => setDateOffset(d => d - 1)}>
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <div className="flex items-center gap-2 min-w-[140px] justify-center">
                  <Calendar className="w-4 h-4 text-[#00d4ff]" />
                  <span className="text-sm font-medium text-white">{dateLabel}</span>
                  <span className="text-xs text-gray-500">
                    {currentDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  </span>
                </div>
                <Button variant="outline" size="icon" className="h-8 w-8 border-white/10 text-gray-400" onClick={() => setDateOffset(d => d + 1)}>
                  <ChevronRight className="w-4 h-4" />
                </Button>
                {dateOffset !== 0 && (
                  <Button variant="ghost" size="sm" className="text-[#00d4ff] h-8 text-xs" onClick={() => setDateOffset(0)}>Today</Button>
                )}
              </div>

              <div className="flex flex-col sm:flex-row gap-3 flex-1 max-w-lg">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input placeholder="Search teams, leagues..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="pl-10 bg-white/5 border-white/10 text-white h-8 text-sm" />
                </div>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-[130px] bg-white/5 border-white/10 text-white h-8 text-sm">
                    <Filter className="w-3 h-3 mr-1.5" />
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0f1623] border-white/10">
                    <SelectItem value="all" className="text-white">All Status</SelectItem>
                    <SelectItem value="live" className="text-white">Live</SelectItem>
                    <SelectItem value="scheduled" className="text-white">Scheduled</SelectItem>
                    <SelectItem value="finished" className="text-white">Finished</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sport Tabs */}
        <Tabs value={activeSport} onValueChange={setActiveSport}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger value="all" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              <Globe className="w-4 h-4 mr-2" />
              All Sports
              <Badge variant="secondary" className="ml-2 bg-white/10 text-gray-400 border-0">{totalMatches}</Badge>
            </TabsTrigger>
            <TabsTrigger value="football" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              Football
            </TabsTrigger>
            <TabsTrigger value="basketball" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              Basketball
            </TabsTrigger>
            <TabsTrigger value="tennis" className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black">
              Tennis
            </TabsTrigger>
          </TabsList>

          <TabsContent value={activeSport} className="mt-6 space-y-4">
            {leagueGroups.length > 0 ? (
              leagueGroups.map((group) => (
                <Card key={`${group.sport}-${group.league}`} className="bg-[#0f1623]/80 border-white/10 overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-2.5 bg-white/[0.02] border-b border-white/5">
                    <div className="flex items-center gap-3">
                      <Trophy className="w-4 h-4 text-[#00d4ff]" />
                      <div>
                        <span className="text-sm font-medium text-white">{group.league}</span>
                        {group.country && <span className="text-xs text-gray-500 ml-2">{group.country}</span>}
                      </div>
                      {group.matches.some(m => m.status === 'live') && (
                        <Badge className="bg-[#ff3860]/20 text-[#ff3860] border-0 text-xs">LIVE</Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>{group.matches.length} matches</span>
                      <span>|</span>
                      <span>{group.sport}</span>
                    </div>
                  </div>
                  <div>
                    {group.matches.map((match) => (
                      <MatchRow key={match.id} match={match} expanded={expandedId === match.id} onToggle={() => setExpandedId(expandedId === match.id ? null : match.id)} />
                    ))}
                  </div>
                </Card>
              ))
            ) : !loading ? (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center">
                  <Search className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No matches found</p>
                  <p className="text-xs text-gray-500 mt-1">Run an analysis to collect match data, or try a different date.</p>
                  <Button variant="link" className="text-[#00d4ff] mt-2" onClick={() => { setSearchQuery(''); setStatusFilter('all'); setActiveSport('all'); }}>
                    Clear filters
                  </Button>
                </CardContent>
              </Card>
            ) : null}
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default MatchesPage;
