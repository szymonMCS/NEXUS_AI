// components/AdvancedFilters.tsx
/**
 * Advanced Filters - Comprehensive filtering for value bets
 * Inspired by sports-ai.dev
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Filter,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Target,
  Percent,
  Shield,
  DollarSign,
  Clock,
  Zap
} from 'lucide-react';
import type { SportId } from '@/lib/api';

// Types
export interface FilterValues {
  sports: SportId[];
  minEdge: number;
  maxEdge: number;
  minQuality: number;
  minConfidence: number;
  minOdds: number;
  maxOdds: number;
  bookmakers: string[];
  timeRange: 'all' | '1h' | '3h' | '6h' | '12h' | '24h';
  onlyTopPicks: boolean;
  hideLowQuality: boolean;
  sortBy: 'edge' | 'quality' | 'confidence' | 'odds' | 'time';
  sortOrder: 'asc' | 'desc';
}

interface AdvancedFiltersProps {
  filters: FilterValues;
  onChange: (filters: FilterValues) => void;
  availableSports: { id: SportId; name: string; icon: string }[];
  availableBookmakers: string[];
  resultCount?: number;
  compact?: boolean;
}

const defaultFilters: FilterValues = {
  sports: [],
  minEdge: 0,
  maxEdge: 30,
  minQuality: 0,
  minConfidence: 0,
  minOdds: 1.0,
  maxOdds: 10.0,
  bookmakers: [],
  timeRange: 'all',
  onlyTopPicks: false,
  hideLowQuality: false,
  sortBy: 'edge',
  sortOrder: 'desc',
};

export function AdvancedFilters({
  filters,
  onChange,
  availableSports,
  availableBookmakers,
  resultCount,
  compact = false
}: AdvancedFiltersProps) {
  const [isOpen, setIsOpen] = useState(!compact);

  const updateFilter = <K extends keyof FilterValues>(key: K, value: FilterValues[K]) => {
    onChange({ ...filters, [key]: value });
  };

  const toggleSport = (sportId: SportId) => {
    const newSports = filters.sports.includes(sportId)
      ? filters.sports.filter(s => s !== sportId)
      : [...filters.sports, sportId];
    updateFilter('sports', newSports);
  };

  const toggleBookmaker = (bookmaker: string) => {
    const newBookmakers = filters.bookmakers.includes(bookmaker)
      ? filters.bookmakers.filter(b => b !== bookmaker)
      : [...filters.bookmakers, bookmaker];
    updateFilter('bookmakers', newBookmakers);
  };

  const resetFilters = () => {
    onChange(defaultFilters);
  };

  const activeFilterCount = [
    filters.sports.length > 0,
    filters.minEdge > 0,
    filters.minQuality > 0,
    filters.minConfidence > 0,
    filters.minOdds > 1.0 || filters.maxOdds < 10.0,
    filters.bookmakers.length > 0,
    filters.timeRange !== 'all',
    filters.onlyTopPicks,
    filters.hideLowQuality,
  ].filter(Boolean).length;

  return (
    <Card className="bg-glass-card border-white/5">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-white/5 transition-colors pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
                <Filter className="w-5 h-5 text-violet-400" />
                Advanced Filters
                {activeFilterCount > 0 && (
                  <Badge className="bg-violet-500/20 text-violet-300 ml-2">
                    {activeFilterCount} active
                  </Badge>
                )}
              </CardTitle>
              <div className="flex items-center gap-2">
                {resultCount !== undefined && (
                  <span className="text-sm text-gray-400">{resultCount} results</span>
                )}
                {isOpen ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="space-y-6 pt-0">
            {/* Sports Filter */}
            <div>
              <Label className="text-sm text-gray-400 mb-2 block">Sports</Label>
              <div className="flex flex-wrap gap-2">
                {availableSports.map((sport) => (
                  <Button
                    key={sport.id}
                    size="sm"
                    variant={filters.sports.includes(sport.id) ? 'default' : 'outline'}
                    className={filters.sports.includes(sport.id)
                      ? 'bg-violet-500'
                      : 'bg-white/5 border-white/10 text-white'}
                    onClick={() => toggleSport(sport.id)}
                  >
                    {sport.icon} {sport.name}
                  </Button>
                ))}
              </div>
            </div>

            {/* Edge Range */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm text-gray-400 flex items-center gap-1">
                  <Percent className="w-4 h-4" />
                  Edge Range
                </Label>
                <span className="text-sm text-white">
                  {filters.minEdge}% - {filters.maxEdge}%
                </span>
              </div>
              <div className="px-2">
                <Slider
                  value={[filters.minEdge, filters.maxEdge]}
                  min={0}
                  max={30}
                  step={1}
                  onValueChange={([min, max]) => {
                    updateFilter('minEdge', min);
                    updateFilter('maxEdge', max);
                  }}
                  className="w-full"
                />
              </div>
            </div>

            {/* Quality Threshold */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm text-gray-400 flex items-center gap-1">
                  <Shield className="w-4 h-4" />
                  Min Quality Score
                </Label>
                <span className="text-sm text-white">{filters.minQuality}%</span>
              </div>
              <div className="px-2">
                <Slider
                  value={[filters.minQuality]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={([val]) => updateFilter('minQuality', val)}
                  className="w-full"
                />
              </div>
            </div>

            {/* Confidence Threshold */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm text-gray-400 flex items-center gap-1">
                  <Target className="w-4 h-4" />
                  Min Confidence
                </Label>
                <span className="text-sm text-white">{filters.minConfidence}%</span>
              </div>
              <div className="px-2">
                <Slider
                  value={[filters.minConfidence]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={([val]) => updateFilter('minConfidence', val)}
                  className="w-full"
                />
              </div>
            </div>

            {/* Odds Range */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm text-gray-400 flex items-center gap-1">
                  <DollarSign className="w-4 h-4" />
                  Odds Range
                </Label>
                <span className="text-sm text-white">
                  {filters.minOdds.toFixed(2)} - {filters.maxOdds.toFixed(2)}
                </span>
              </div>
              <div className="px-2">
                <Slider
                  value={[filters.minOdds, filters.maxOdds]}
                  min={1.0}
                  max={10.0}
                  step={0.1}
                  onValueChange={([min, max]) => {
                    updateFilter('minOdds', min);
                    updateFilter('maxOdds', max);
                  }}
                  className="w-full"
                />
              </div>
            </div>

            {/* Time Range */}
            <div>
              <Label className="text-sm text-gray-400 mb-2 block flex items-center gap-1">
                <Clock className="w-4 h-4" />
                Time to Match
              </Label>
              <Select
                value={filters.timeRange}
                onValueChange={(val) => updateFilter('timeRange', val as FilterValues['timeRange'])}
              >
                <SelectTrigger className="w-full bg-white/5 border-white/10 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All upcoming</SelectItem>
                  <SelectItem value="1h">Within 1 hour</SelectItem>
                  <SelectItem value="3h">Within 3 hours</SelectItem>
                  <SelectItem value="6h">Within 6 hours</SelectItem>
                  <SelectItem value="12h">Within 12 hours</SelectItem>
                  <SelectItem value="24h">Within 24 hours</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Bookmakers */}
            {availableBookmakers.length > 0 && (
              <div>
                <Label className="text-sm text-gray-400 mb-2 block">Bookmakers</Label>
                <div className="flex flex-wrap gap-2">
                  {availableBookmakers.map((bookmaker) => (
                    <Button
                      key={bookmaker}
                      size="sm"
                      variant={filters.bookmakers.includes(bookmaker) ? 'default' : 'outline'}
                      className={filters.bookmakers.includes(bookmaker)
                        ? 'bg-blue-500'
                        : 'bg-white/5 border-white/10 text-white'}
                      onClick={() => toggleBookmaker(bookmaker)}
                    >
                      {bookmaker}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Toggle Options */}
            <div className="space-y-3 pt-2 border-t border-white/10">
              <div className="flex items-center justify-between">
                <Label className="text-sm text-gray-300 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  Only Top Picks (High Edge + Quality)
                </Label>
                <Switch
                  checked={filters.onlyTopPicks}
                  onCheckedChange={(val) => updateFilter('onlyTopPicks', val)}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-sm text-gray-300 flex items-center gap-2">
                  <Shield className="w-4 h-4 text-red-400" />
                  Hide Low Quality (&lt;50%)
                </Label>
                <Switch
                  checked={filters.hideLowQuality}
                  onCheckedChange={(val) => updateFilter('hideLowQuality', val)}
                />
              </div>
            </div>

            {/* Sort Options */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-sm text-gray-400 mb-2 block">Sort By</Label>
                <Select
                  value={filters.sortBy}
                  onValueChange={(val) => updateFilter('sortBy', val as FilterValues['sortBy'])}
                >
                  <SelectTrigger className="w-full bg-white/5 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="edge">Edge</SelectItem>
                    <SelectItem value="quality">Quality</SelectItem>
                    <SelectItem value="confidence">Confidence</SelectItem>
                    <SelectItem value="odds">Odds</SelectItem>
                    <SelectItem value="time">Time</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-sm text-gray-400 mb-2 block">Order</Label>
                <Select
                  value={filters.sortOrder}
                  onValueChange={(val) => updateFilter('sortOrder', val as FilterValues['sortOrder'])}
                >
                  <SelectTrigger className="w-full bg-white/5 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="desc">Highest First</SelectItem>
                    <SelectItem value="asc">Lowest First</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Reset Button */}
            <div className="flex justify-end pt-2">
              <Button
                variant="outline"
                size="sm"
                className="bg-white/5 border-white/10 text-gray-400 hover:text-white"
                onClick={resetFilters}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Reset Filters
              </Button>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

export default AdvancedFilters;
