/**
 * MatchesPage - Comprehensive match browser and analysis center
 *
 * Features:
 * - All matches organized by date, league, and sport
 * - Live match tracking with real-time scores
 * - Match detail panels with team stats and head-to-head
 * - Quick prediction access per match
 * - Sport sub-tabs (Football, Basketball, Tennis, etc.)
 * - Date navigation with calendar
 * - League grouping with collapsible sections
 * - Form indicators and key stats inline
 */

import { useState } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
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
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Activity,
  ArrowRight,
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clock,
  Crosshair,
  Filter,
  Globe,
  MapPin,
  Search,
  Shield,
  Star,
  Swords,
  Target,
  TrendingUp,
  Trophy,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

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
  odds: { home: number; draw?: number; away: number };
  prediction?: {
    result: string;
    confidence: number;
    isValue: boolean;
    edge?: number;
  };
  homeForm: string[];
  awayForm: string[];
  h2h: { homeWins: number; draws: number; awayWins: number; total: number };
  venue?: string;
  importance: 'low' | 'medium' | 'high';
}

interface LeagueGroup {
  league: string;
  country: string;
  sport: string;
  matches: Match[];
}

// Mock data
const allMatches: Match[] = [
  // Football - Premier League
  {
    id: 'f1',
    sport: 'Football',
    league: 'Premier League',
    leagueCountry: 'England',
    homeTeam: 'Manchester City',
    awayTeam: 'Arsenal',
    startTime: '18:30',
    status: 'scheduled',
    odds: { home: 1.65, draw: 3.80, away: 4.20 },
    prediction: { result: 'Home Win', confidence: 82, isValue: true, edge: 4.2 },
    homeForm: ['W', 'W', 'D', 'W', 'W'],
    awayForm: ['L', 'W', 'W', 'D', 'W'],
    h2h: { homeWins: 12, draws: 6, awayWins: 6, total: 24 },
    venue: 'Etihad Stadium',
    importance: 'high',
  },
  {
    id: 'f2',
    sport: 'Football',
    league: 'Premier League',
    leagueCountry: 'England',
    homeTeam: 'Liverpool',
    awayTeam: 'Chelsea',
    startTime: '16:00',
    status: 'live',
    homeScore: 2,
    awayScore: 1,
    minute: "67'",
    odds: { home: 1.80, draw: 3.60, away: 3.50 },
    prediction: { result: 'Home Win', confidence: 75, isValue: false, edge: 1.8 },
    homeForm: ['W', 'W', 'W', 'D', 'W'],
    awayForm: ['W', 'L', 'W', 'W', 'L'],
    h2h: { homeWins: 10, draws: 8, awayWins: 6, total: 24 },
    venue: 'Anfield',
    importance: 'high',
  },
  {
    id: 'f3',
    sport: 'Football',
    league: 'Premier League',
    leagueCountry: 'England',
    homeTeam: 'Tottenham',
    awayTeam: 'Aston Villa',
    startTime: '14:00',
    status: 'finished',
    homeScore: 1,
    awayScore: 3,
    odds: { home: 2.10, draw: 3.40, away: 3.20 },
    prediction: { result: 'Home Win', confidence: 62, isValue: false },
    homeForm: ['W', 'L', 'W', 'W', 'L'],
    awayForm: ['W', 'W', 'L', 'W', 'L'],
    h2h: { homeWins: 8, draws: 4, awayWins: 4, total: 16 },
    venue: 'Tottenham Hotspur Stadium',
    importance: 'medium',
  },
  // Football - La Liga
  {
    id: 'f4',
    sport: 'Football',
    league: 'La Liga',
    leagueCountry: 'Spain',
    homeTeam: 'Real Madrid',
    awayTeam: 'Barcelona',
    startTime: '21:00',
    status: 'scheduled',
    odds: { home: 2.10, draw: 3.40, away: 3.25 },
    prediction: { result: 'Draw', confidence: 58, isValue: true, edge: 3.1 },
    homeForm: ['W', 'D', 'W', 'W', 'W'],
    awayForm: ['W', 'W', 'L', 'W', 'D'],
    h2h: { homeWins: 10, draws: 8, awayWins: 6, total: 24 },
    venue: 'Santiago Bernabeu',
    importance: 'high',
  },
  {
    id: 'f5',
    sport: 'Football',
    league: 'La Liga',
    leagueCountry: 'Spain',
    homeTeam: 'Atletico Madrid',
    awayTeam: 'Sevilla',
    startTime: '18:30',
    status: 'scheduled',
    odds: { home: 1.70, draw: 3.60, away: 4.50 },
    prediction: { result: 'Home Win', confidence: 73, isValue: false },
    homeForm: ['W', 'W', 'W', 'D', 'W'],
    awayForm: ['L', 'D', 'W', 'L', 'W'],
    h2h: { homeWins: 9, draws: 5, awayWins: 2, total: 16 },
    venue: 'Civitas Metropolitano',
    importance: 'medium',
  },
  // Football - Bundesliga
  {
    id: 'f6',
    sport: 'Football',
    league: 'Bundesliga',
    leagueCountry: 'Germany',
    homeTeam: 'Bayern Munich',
    awayTeam: 'Dortmund',
    startTime: '17:30',
    status: 'scheduled',
    odds: { home: 1.45, draw: 4.20, away: 5.50 },
    prediction: { result: 'Home Win', confidence: 78, isValue: true, edge: 3.5 },
    homeForm: ['W', 'W', 'W', 'W', 'D'],
    awayForm: ['W', 'L', 'W', 'D', 'W'],
    h2h: { homeWins: 14, draws: 4, awayWins: 6, total: 24 },
    venue: 'Allianz Arena',
    importance: 'high',
  },
  // Basketball - NBA
  {
    id: 'b1',
    sport: 'Basketball',
    league: 'NBA',
    leagueCountry: 'USA',
    homeTeam: 'Celtics',
    awayTeam: 'Heat',
    startTime: '01:30',
    status: 'scheduled',
    odds: { home: 1.45, away: 2.65 },
    prediction: { result: 'Home Win', confidence: 78, isValue: true, edge: 3.0 },
    homeForm: ['W', 'W', 'W', 'L', 'W'],
    awayForm: ['W', 'L', 'L', 'W', 'L'],
    h2h: { homeWins: 5, draws: 0, awayWins: 3, total: 8 },
    venue: 'TD Garden',
    importance: 'medium',
  },
  {
    id: 'b2',
    sport: 'Basketball',
    league: 'NBA',
    leagueCountry: 'USA',
    homeTeam: 'Lakers',
    awayTeam: 'Warriors',
    startTime: '04:00',
    status: 'live',
    homeScore: 98,
    awayScore: 102,
    minute: 'Q4 2:45',
    odds: { home: 2.15, away: 1.75 },
    prediction: { result: 'Away Win', confidence: 65, isValue: false },
    homeForm: ['L', 'W', 'W', 'L', 'W'],
    awayForm: ['W', 'W', 'L', 'W', 'W'],
    h2h: { homeWins: 4, draws: 0, awayWins: 4, total: 8 },
    venue: 'Crypto.com Arena',
    importance: 'high',
  },
  // Tennis - ATP
  {
    id: 't1',
    sport: 'Tennis',
    league: 'ATP Finals',
    leagueCountry: 'International',
    homeTeam: 'Alcaraz',
    awayTeam: 'Djokovic',
    startTime: '09:00',
    status: 'scheduled',
    odds: { home: 2.10, away: 1.75 },
    prediction: { result: 'Djokovic Win', confidence: 71, isValue: true, edge: 3.9 },
    homeForm: ['W', 'W', 'L', 'W', 'W'],
    awayForm: ['W', 'W', 'W', 'W', 'L'],
    h2h: { homeWins: 4, draws: 0, awayWins: 7, total: 11 },
    importance: 'high',
  },
  {
    id: 't2',
    sport: 'Tennis',
    league: 'ATP Finals',
    leagueCountry: 'International',
    homeTeam: 'Sinner',
    awayTeam: 'Medvedev',
    startTime: '14:00',
    status: 'scheduled',
    odds: { home: 1.55, away: 2.40 },
    prediction: { result: 'Sinner Win', confidence: 68, isValue: false },
    homeForm: ['W', 'W', 'W', 'L', 'W'],
    awayForm: ['L', 'W', 'L', 'W', 'W'],
    h2h: { homeWins: 6, draws: 0, awayWins: 7, total: 13 },
    importance: 'medium',
  },
];

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
  // Sort: live matches first, then by importance, then alphabetical
  return Object.values(groups).sort((a, b) => {
    const aHasLive = a.matches.some((m) => m.status === 'live') ? 1 : 0;
    const bHasLive = b.matches.some((m) => m.status === 'live') ? 1 : 0;
    if (bHasLive !== aHasLive) return bHasLive - aHasLive;
    return a.league.localeCompare(b.league);
  });
}

// Sub-components
const FormIndicator = ({ results }: { results: string[] }) => (
  <div className="flex items-center gap-0.5">
    {results.map((result, idx) => (
      <div
        key={idx}
        className={cn(
          'w-4 h-4 rounded-sm flex items-center justify-center text-[9px] font-bold',
          result === 'W'
            ? 'bg-[#00ff88]/20 text-[#00ff88]'
            : result === 'D'
              ? 'bg-[#ff9500]/20 text-[#ff9500]'
              : 'bg-[#ff3860]/20 text-[#ff3860]'
        )}
      >
        {result}
      </div>
    ))}
  </div>
);

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
                <span
                  className={cn(
                    'text-sm font-medium truncate',
                    isLive || isFinished ? 'text-white' : 'text-gray-300'
                  )}
                >
                  {match.homeTeam}
                </span>
                {(isLive || isFinished) && (
                  <span
                    className={cn(
                      'text-lg font-bold tabular-nums w-6 text-right',
                      isLive ? 'text-white' : 'text-gray-300'
                    )}
                  >
                    {match.homeScore}
                  </span>
                )}
              </div>
              <div className="flex items-center justify-between gap-2 mt-0.5">
                <span
                  className={cn(
                    'text-sm font-medium truncate',
                    isLive || isFinished ? 'text-white' : 'text-gray-300'
                  )}
                >
                  {match.awayTeam}
                </span>
                {(isLive || isFinished) && (
                  <span
                    className={cn(
                      'text-lg font-bold tabular-nums w-6 text-right',
                      isLive ? 'text-white' : 'text-gray-300'
                    )}
                  >
                    {match.awayScore}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Odds */}
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
                        match.prediction.confidence >= 75
                          ? 'bg-[#00ff88]'
                          : match.prediction.confidence >= 60
                            ? 'bg-[#00d4ff]'
                            : 'bg-[#ff9500]'
                      )}
                      style={{ width: `${match.prediction.confidence}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-gray-500 w-7">
                    {match.prediction.confidence}%
                  </span>
                </div>
              </div>
              {match.prediction.isValue && (
                <Badge className="bg-[#00ff88]/20 text-[#00ff88] border-0 text-[10px] px-1.5 py-0">
                  V
                </Badge>
              )}
            </>
          )}
        </div>

        {/* Importance */}
        <div className="hidden xl:block w-6 shrink-0">
          {match.importance === 'high' && (
            <Star className="w-4 h-4 text-[#ff9500]" />
          )}
        </div>

        {/* Expand */}
        <div className="w-5 shrink-0">
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-white/5">
          <div className="pt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
                    <span
                      className={cn(
                        'text-sm font-medium',
                        match.prediction.confidence >= 75
                          ? 'text-[#00ff88]'
                          : match.prediction.confidence >= 60
                            ? 'text-[#00d4ff]'
                            : 'text-[#ff9500]'
                      )}
                    >
                      {match.prediction.confidence}%
                    </span>
                  </div>
                  {match.prediction.edge !== undefined && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Edge</span>
                      <Badge
                        className={cn(
                          'border-0 text-xs',
                          match.prediction.edge >= 3
                            ? 'bg-[#00ff88]/20 text-[#00ff88]'
                            : 'bg-[#00d4ff]/20 text-[#00d4ff]'
                        )}
                      >
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

            {/* Form */}
            <div className="space-y-3">
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5" />
                Recent Form
              </h4>
              <div className="p-3 bg-white/5 rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">{match.homeTeam}</span>
                  <FormIndicator results={match.homeForm} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">{match.awayTeam}</span>
                  <FormIndicator results={match.awayForm} />
                </div>
              </div>
            </div>

            {/* H2H */}
            <div className="space-y-3">
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                <Swords className="w-3.5 h-3.5" />
                Head-to-Head
              </h4>
              <div className="p-3 bg-white/5 rounded-lg space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Total Matches</span>
                  <span className="text-white">{match.h2h.total}</span>
                </div>
                <div className="flex gap-1 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-[#00ff88] rounded-l"
                    style={{
                      width: `${(match.h2h.homeWins / match.h2h.total) * 100}%`,
                    }}
                  />
                  {match.h2h.draws > 0 && (
                    <div
                      className="bg-[#ff9500]"
                      style={{
                        width: `${(match.h2h.draws / match.h2h.total) * 100}%`,
                      }}
                    />
                  )}
                  <div
                    className="bg-[#ff3860] rounded-r"
                    style={{
                      width: `${(match.h2h.awayWins / match.h2h.total) * 100}%`,
                    }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{match.h2h.homeWins}W</span>
                  {match.h2h.draws > 0 && <span>{match.h2h.draws}D</span>}
                  <span>{match.h2h.awayWins}W</span>
                </div>
              </div>
            </div>

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
                {match.venue && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Venue</span>
                    <span className="text-gray-300 text-xs">{match.venue}</span>
                  </div>
                )}
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Kick-off</span>
                  <span className="text-white">{match.startTime}</span>
                </div>
                <div className="pt-2 border-t border-white/10 flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 border-white/10 text-gray-400 text-xs h-7"
                  >
                    <Crosshair className="w-3 h-3 mr-1" />
                    Handicaps
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 border-white/10 text-gray-400 text-xs h-7"
                  >
                    <Shield className="w-3 h-3 mr-1" />
                    Stats
                  </Button>
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
  const [expandedId, setExpandedId] = useState<string | null>('f1');
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [dateOffset, setDateOffset] = useState(0);

  // Date navigation
  const currentDate = new Date();
  currentDate.setDate(currentDate.getDate() + dateOffset);
  const dateLabel =
    dateOffset === 0
      ? 'Today'
      : dateOffset === 1
        ? 'Tomorrow'
        : dateOffset === -1
          ? 'Yesterday'
          : currentDate.toLocaleDateString('en-US', {
              weekday: 'short',
              month: 'short',
              day: 'numeric',
            });

  // Filter matches
  const filteredMatches = allMatches.filter((m) => {
    if (activeSport !== 'all' && m.sport.toLowerCase() !== activeSport) return false;
    if (statusFilter === 'live' && m.status !== 'live') return false;
    if (statusFilter === 'scheduled' && m.status !== 'scheduled') return false;
    if (statusFilter === 'finished' && m.status !== 'finished') return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        m.homeTeam.toLowerCase().includes(q) ||
        m.awayTeam.toLowerCase().includes(q) ||
        m.league.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const leagueGroups = groupByLeague(filteredMatches);

  // Stats
  const totalMatches = allMatches.length;
  const liveMatches = allMatches.filter((m) => m.status === 'live').length;
  const valueMatches = allMatches.filter((m) => m.prediction?.isValue).length;
  const sports = [...new Set(allMatches.map((m) => m.sport))];

  return (
    <PageLayout
      title="Matches"
      description={`All matches for ${dateLabel} across leagues and sports`}
      breadcrumbs={[{ label: 'Matches' }]}
    >
      <div className="space-y-6">
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
              {/* Date Navigation */}
              <div className="flex items-center gap-3">
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 border-white/10 text-gray-400"
                  onClick={() => setDateOffset((d) => d - 1)}
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <div className="flex items-center gap-2 min-w-[140px] justify-center">
                  <Calendar className="w-4 h-4 text-[#00d4ff]" />
                  <span className="text-sm font-medium text-white">{dateLabel}</span>
                  <span className="text-xs text-gray-500">
                    {currentDate.toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </span>
                </div>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 border-white/10 text-gray-400"
                  onClick={() => setDateOffset((d) => d + 1)}
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
                {dateOffset !== 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-[#00d4ff] h-8 text-xs"
                    onClick={() => setDateOffset(0)}
                  >
                    Today
                  </Button>
                )}
              </div>

              {/* Search and Filters */}
              <div className="flex flex-col sm:flex-row gap-3 flex-1 max-w-lg">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    placeholder="Search teams, leagues..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 bg-white/5 border-white/10 text-white h-8 text-sm"
                  />
                </div>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-[130px] bg-white/5 border-white/10 text-white h-8 text-sm">
                    <Filter className="w-3 h-3 mr-1.5" />
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0f1623] border-white/10">
                    <SelectItem value="all" className="text-white">
                      All Status
                    </SelectItem>
                    <SelectItem value="live" className="text-white">
                      Live
                    </SelectItem>
                    <SelectItem value="scheduled" className="text-white">
                      Scheduled
                    </SelectItem>
                    <SelectItem value="finished" className="text-white">
                      Finished
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sport Tabs */}
        <Tabs value={activeSport} onValueChange={setActiveSport}>
          <TabsList className="bg-[#0f1623]/80 border border-white/10">
            <TabsTrigger
              value="all"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              <Globe className="w-4 h-4 mr-2" />
              All Sports
              <Badge
                variant="secondary"
                className="ml-2 bg-white/10 text-gray-400 border-0"
              >
                {totalMatches}
              </Badge>
            </TabsTrigger>
            <TabsTrigger
              value="football"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              Football
              <Badge
                variant="secondary"
                className="ml-2 bg-white/10 text-gray-400 border-0"
              >
                {allMatches.filter((m) => m.sport === 'Football').length}
              </Badge>
            </TabsTrigger>
            <TabsTrigger
              value="basketball"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              Basketball
              <Badge
                variant="secondary"
                className="ml-2 bg-white/10 text-gray-400 border-0"
              >
                {allMatches.filter((m) => m.sport === 'Basketball').length}
              </Badge>
            </TabsTrigger>
            <TabsTrigger
              value="tennis"
              className="data-[state=active]:bg-[#00d4ff] data-[state=active]:text-black"
            >
              Tennis
              <Badge
                variant="secondary"
                className="ml-2 bg-white/10 text-gray-400 border-0"
              >
                {allMatches.filter((m) => m.sport === 'Tennis').length}
              </Badge>
            </TabsTrigger>
          </TabsList>

          {/* Match List (content is the same for all tabs; filtering is handled above) */}
          <TabsContent value={activeSport} className="mt-6 space-y-4">
            {leagueGroups.length > 0 ? (
              leagueGroups.map((group) => (
                <Card key={`${group.sport}-${group.league}`} className="bg-[#0f1623]/80 border-white/10 overflow-hidden">
                  {/* League Header */}
                  <div className="flex items-center justify-between px-4 py-2.5 bg-white/[0.02] border-b border-white/5">
                    <div className="flex items-center gap-3">
                      <Trophy className="w-4 h-4 text-[#00d4ff]" />
                      <div>
                        <span className="text-sm font-medium text-white">{group.league}</span>
                        <span className="text-xs text-gray-500 ml-2">{group.country}</span>
                      </div>
                      {group.matches.some((m) => m.status === 'live') && (
                        <Badge className="bg-[#ff3860]/20 text-[#ff3860] border-0 text-xs">
                          LIVE
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>{group.matches.length} matches</span>
                      <span>|</span>
                      <span>{group.sport}</span>
                    </div>
                  </div>

                  {/* Matches */}
                  <div>
                    {group.matches.map((match) => (
                      <MatchRow
                        key={match.id}
                        match={match}
                        expanded={expandedId === match.id}
                        onToggle={() =>
                          setExpandedId(expandedId === match.id ? null : match.id)
                        }
                      />
                    ))}
                  </div>
                </Card>
              ))
            ) : (
              <Card className="bg-[#0f1623]/80 border-white/10">
                <CardContent className="p-12 text-center">
                  <Search className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No matches found for your filters</p>
                  <Button
                    variant="link"
                    className="text-[#00d4ff] mt-2"
                    onClick={() => {
                      setSearchQuery('');
                      setStatusFilter('all');
                      setActiveSport('all');
                    }}
                  >
                    Clear filters
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

export default MatchesPage;
