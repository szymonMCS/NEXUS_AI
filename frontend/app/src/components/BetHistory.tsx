// components/BetHistory.tsx
/**
 * Bet History - Track placed bets and results
 */

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  History,
  Search,
  Download,
  CheckCircle,
  XCircle,
  Clock,
  Calendar,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

// Types
type BetStatus = 'pending' | 'won' | 'lost' | 'void';

interface Bet {
  id: string;
  date: string;
  sport: string;
  match: string;
  selection: string;
  odds: number;
  stake: number;
  potentialWin: number;
  status: BetStatus;
  profit?: number;
  edge: number;
  quality: number;
  confidence: number;
  bookmaker: string;
}

interface BetHistoryProps {
  bets: Bet[];
  onBetClick?: (bet: Bet) => void;
  onExport?: () => void;
}

const StatusBadge = ({ status }: { status: BetStatus }) => {
  const config = {
    pending: { icon: Clock, className: 'bg-yellow-500/20 text-yellow-400', label: 'Pending' },
    won: { icon: CheckCircle, className: 'bg-green-500/20 text-green-400', label: 'Won' },
    lost: { icon: XCircle, className: 'bg-red-500/20 text-red-400', label: 'Lost' },
    void: { icon: XCircle, className: 'bg-gray-500/20 text-gray-400', label: 'Void' },
  };

  const { icon: Icon, className, label } = config[status];

  return (
    <Badge className={`${className} flex items-center gap-1`}>
      <Icon className="w-3 h-3" />
      {label}
    </Badge>
  );
};

export function BetHistory({ bets, onBetClick, onExport }: BetHistoryProps) {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<BetStatus | 'all'>('all');
  const [sportFilter, setSportFilter] = useState<string>('all');
  const [page, setPage] = useState(1);
  const perPage = 10;

  // Get unique sports
  const sports = Array.from(new Set(bets.map(b => b.sport)));

  // Filter bets
  const filteredBets = bets.filter(bet => {
    const matchesSearch = bet.match.toLowerCase().includes(search.toLowerCase()) ||
                         bet.selection.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === 'all' || bet.status === statusFilter;
    const matchesSport = sportFilter === 'all' || bet.sport === sportFilter;
    return matchesSearch && matchesStatus && matchesSport;
  });

  // Pagination
  const totalPages = Math.ceil(filteredBets.length / perPage);
  const paginatedBets = filteredBets.slice((page - 1) * perPage, page * perPage);

  // Calculate stats
  const stats = {
    total: filteredBets.length,
    pending: filteredBets.filter(b => b.status === 'pending').length,
    won: filteredBets.filter(b => b.status === 'won').length,
    lost: filteredBets.filter(b => b.status === 'lost').length,
    totalProfit: filteredBets.reduce((acc, b) => acc + (b.profit || 0), 0),
    totalStaked: filteredBets.reduce((acc, b) => acc + b.stake, 0),
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <History className="w-6 h-6 text-violet-400" />
            Bet History
          </h2>
          <p className="text-gray-400 text-sm">Track and analyze your betting history</p>
        </div>
        {onExport && (
          <Button size="sm" variant="outline" className="bg-white/5 border-white/10" onClick={onExport}>
            <Download className="w-4 h-4 mr-1" />
            Export CSV
          </Button>
        )}
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Card className="bg-glass-card border-white/5">
          <CardContent className="p-3 text-center">
            <div className="text-xl font-bold text-white">{stats.total}</div>
            <div className="text-xs text-gray-400">Total Bets</div>
          </CardContent>
        </Card>
        <Card className="bg-glass-card border-white/5">
          <CardContent className="p-3 text-center">
            <div className="text-xl font-bold text-yellow-400">{stats.pending}</div>
            <div className="text-xs text-gray-400">Pending</div>
          </CardContent>
        </Card>
        <Card className="bg-glass-card border-white/5">
          <CardContent className="p-3 text-center">
            <div className="text-xl font-bold text-green-400">{stats.won}</div>
            <div className="text-xs text-gray-400">Won</div>
          </CardContent>
        </Card>
        <Card className="bg-glass-card border-white/5">
          <CardContent className="p-3 text-center">
            <div className="text-xl font-bold text-red-400">{stats.lost}</div>
            <div className="text-xs text-gray-400">Lost</div>
          </CardContent>
        </Card>
        <Card className="bg-glass-card border-white/5">
          <CardContent className="p-3 text-center">
            <div className={`text-xl font-bold ${stats.totalProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stats.totalProfit >= 0 ? '+' : ''}${stats.totalProfit.toFixed(2)}
            </div>
            <div className="text-xs text-gray-400">Profit</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card className="bg-glass-card border-white/5">
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search matches..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 bg-white/5 border-white/10 text-white"
              />
            </div>
            <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as BetStatus | 'all')}>
              <SelectTrigger className="w-[140px] bg-white/5 border-white/10 text-white">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="won">Won</SelectItem>
                <SelectItem value="lost">Lost</SelectItem>
                <SelectItem value="void">Void</SelectItem>
              </SelectContent>
            </Select>
            <Select value={sportFilter} onValueChange={setSportFilter}>
              <SelectTrigger className="w-[140px] bg-white/5 border-white/10 text-white">
                <SelectValue placeholder="Sport" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sports</SelectItem>
                {sports.map(sport => (
                  <SelectItem key={sport} value={sport}>{sport}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Bets Table */}
      <Card className="bg-glass-card border-white/5 overflow-hidden">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-white/10 hover:bg-transparent">
                  <TableHead className="text-gray-400">Date</TableHead>
                  <TableHead className="text-gray-400">Match</TableHead>
                  <TableHead className="text-gray-400">Selection</TableHead>
                  <TableHead className="text-gray-400 text-right">Odds</TableHead>
                  <TableHead className="text-gray-400 text-right">Stake</TableHead>
                  <TableHead className="text-gray-400 text-right">Edge</TableHead>
                  <TableHead className="text-gray-400 text-center">Status</TableHead>
                  <TableHead className="text-gray-400 text-right">Profit</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedBets.map((bet) => (
                  <TableRow
                    key={bet.id}
                    className="border-white/5 hover:bg-white/5 cursor-pointer"
                    onClick={() => onBetClick?.(bet)}
                  >
                    <TableCell className="text-gray-300">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4 text-gray-500" />
                        {bet.date}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div>
                        <div className="text-white font-medium">{bet.match}</div>
                        <div className="text-xs text-gray-500">{bet.sport} | {bet.bookmaker}</div>
                      </div>
                    </TableCell>
                    <TableCell className="text-violet-400 font-medium">{bet.selection}</TableCell>
                    <TableCell className="text-right text-white">{bet.odds.toFixed(2)}</TableCell>
                    <TableCell className="text-right text-white">${bet.stake.toFixed(2)}</TableCell>
                    <TableCell className="text-right text-green-400">+{(bet.edge * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-center">
                      <StatusBadge status={bet.status} />
                    </TableCell>
                    <TableCell className="text-right">
                      {bet.profit !== undefined ? (
                        <span className={bet.profit >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {bet.profit >= 0 ? '+' : ''}${bet.profit.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between p-4 border-t border-white/10">
              <div className="text-sm text-gray-400">
                Showing {(page - 1) * perPage + 1}-{Math.min(page * perPage, filteredBets.length)} of {filteredBets.length}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="bg-white/5 border-white/10"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-gray-400">
                  Page {page} of {totalPages}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  className="bg-white/5 border-white/10"
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Empty State */}
          {filteredBets.length === 0 && (
            <div className="text-center py-12">
              <History className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">No bets found</h3>
              <p className="text-gray-400">
                {search || statusFilter !== 'all' || sportFilter !== 'all'
                  ? 'Try adjusting your filters'
                  : 'Your betting history will appear here'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default BetHistory;
