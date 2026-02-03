"""
Scheduler for automatic match analysis.
Runs predictions every X hours for upcoming matches.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisFrequency(Enum):
    """Analysis frequency options."""
    HOURLY = "hourly"
    EVERY_3_HOURS = "every_3_hours"
    EVERY_6_HOURS = "every_6_hours"
    EVERY_12_HOURS = "every_12_hours"
    DAILY = "daily"
    CUSTOM = "custom"


@dataclass
class ScheduledAnalysis:
    """Scheduled analysis task."""
    id: str
    sport: str
    league: Optional[str]
    frequency: AnalysisFrequency
    custom_hours: Optional[int] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    
    def get_interval_hours(self) -> int:
        """Get interval in hours."""
        intervals = {
            AnalysisFrequency.HOURLY: 1,
            AnalysisFrequency.EVERY_3_HOURS: 3,
            AnalysisFrequency.EVERY_6_HOURS: 6,
            AnalysisFrequency.EVERY_12_HOURS: 12,
            AnalysisFrequency.DAILY: 24,
        }
        if self.frequency == AnalysisFrequency.CUSTOM and self.custom_hours:
            return self.custom_hours
        return intervals.get(self.frequency, 6)


class AnalysisScheduler:
    """
    Scheduler for automatic match analysis.
    Runs analysis tasks at configured intervals.
    """
    
    def __init__(self, data_dir: str = "data/scheduler"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.schedules_file = self.data_dir / "schedules.json"
        self.analysis_log_file = self.data_dir / "analysis_log.json"
        
        self.schedules: Dict[str, ScheduledAnalysis] = {}
        self._running = False
        self._task = None
        self._analysis_callback: Optional[Callable] = None
        
        self._load_schedules()
    
    def _load_schedules(self):
        """Load saved schedules."""
        if self.schedules_file.exists():
            try:
                with open(self.schedules_file, 'r') as f:
                    data = json.load(f)
                
                for item in data:
                    schedule = ScheduledAnalysis(
                        id=item['id'],
                        sport=item['sport'],
                        league=item.get('league'),
                        frequency=AnalysisFrequency(item['frequency']),
                        custom_hours=item.get('custom_hours'),
                        last_run=datetime.fromisoformat(item['last_run']) if item.get('last_run') else None,
                        next_run=datetime.fromisoformat(item['next_run']) if item.get('next_run') else None,
                        enabled=item.get('enabled', True)
                    )
                    self.schedules[schedule.id] = schedule
                    
                logger.info(f"Loaded {len(self.schedules)} schedules")
            except Exception as e:
                logger.error(f"Error loading schedules: {e}")
    
    def _save_schedules(self):
        """Save schedules to file."""
        data = []
        for schedule in self.schedules.values():
            data.append({
                'id': schedule.id,
                'sport': schedule.sport,
                'league': schedule.league,
                'frequency': schedule.frequency.value,
                'custom_hours': schedule.custom_hours,
                'last_run': schedule.last_run.isoformat() if schedule.last_run else None,
                'next_run': schedule.next_run.isoformat() if schedule.next_run else None,
                'enabled': schedule.enabled
            })
        
        with open(self.schedules_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_schedule(self, sport: str, 
                       frequency: AnalysisFrequency = AnalysisFrequency.EVERY_6_HOURS,
                       league: Optional[str] = None,
                       custom_hours: Optional[int] = None) -> ScheduledAnalysis:
        """
        Create a new analysis schedule.
        
        Args:
            sport: Sport to analyze
            frequency: How often to run
            league: Specific league (optional)
            custom_hours: Custom interval for CUSTOM frequency
        
        Returns:
            Created schedule
        """
        schedule_id = f"{sport}_{league or 'all'}_{frequency.value}"
        
        schedule = ScheduledAnalysis(
            id=schedule_id,
            sport=sport,
            league=league,
            frequency=frequency,
            custom_hours=custom_hours,
            next_run=datetime.now() + timedelta(hours=1)  # First run in 1 hour
        )
        
        self.schedules[schedule_id] = schedule
        self._save_schedules()
        
        logger.info(f"Created schedule: {schedule_id}")
        return schedule
    
    def delete_schedule(self, schedule_id: str):
        """Delete a schedule."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self._save_schedules()
            logger.info(f"Deleted schedule: {schedule_id}")
    
    def enable_schedule(self, schedule_id: str, enabled: bool = True):
        """Enable/disable a schedule."""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = enabled
            self._save_schedules()
    
    def set_analysis_callback(self, callback: Callable):
        """
        Set the callback function for analysis.
        
        Args:
            callback: async function(schedule: ScheduledAnalysis) -> results
        """
        self._analysis_callback = callback
    
    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_schedules()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_schedules(self):
        """Check and execute due schedules."""
        now = datetime.now()
        
        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue
            
            if schedule.next_run and schedule.next_run <= now:
                # Execute analysis
                await self._run_analysis(schedule)
                
                # Update next run time
                schedule.last_run = now
                interval = schedule.get_interval_hours()
                schedule.next_run = now + timedelta(hours=interval)
                
                self._save_schedules()
    
    async def _run_analysis(self, schedule: ScheduledAnalysis):
        """Run analysis for a schedule."""
        logger.info(f"Running scheduled analysis: {schedule.id}")
        
        try:
            if self._analysis_callback:
                results = await self._analysis_callback(schedule)
                
                # Log results
                self._log_analysis(schedule, results)
                
                logger.info(f"Analysis complete: {schedule.id} - {len(results)} matches analyzed")
            else:
                logger.warning(f"No analysis callback set for schedule: {schedule.id}")
                
        except Exception as e:
            logger.error(f"Analysis failed for {schedule.id}: {e}")
            self._log_analysis(schedule, [], error=str(e))
    
    def _log_analysis(self, schedule: ScheduledAnalysis, results: List[Dict], error: Optional[str] = None):
        """Log analysis results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'schedule_id': schedule.id,
            'sport': schedule.sport,
            'league': schedule.league,
            'results_count': len(results),
            'error': error,
            'results': results[:10]  # Store first 10 results
        }
        
        # Append to log
        logs = []
        if self.analysis_log_file.exists():
            try:
                with open(self.analysis_log_file, 'r') as f:
                    logs = json.load(f)
            except:
                pass
        
        logs.append(log_entry)
        
        # Keep last 1000 entries
        logs = logs[-1000:]
        
        with open(self.analysis_log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    
    def get_schedules(self) -> List[ScheduledAnalysis]:
        """Get all schedules."""
        return list(self.schedules.values())
    
    def get_analysis_log(self, limit: int = 100) -> List[Dict]:
        """Get analysis log."""
        if not self.analysis_log_file.exists():
            return []
        
        with open(self.analysis_log_file, 'r') as f:
            logs = json.load(f)
        
        return logs[-limit:]
    
    def get_upcoming_analyses(self) -> List[Dict]:
        """Get list of upcoming scheduled analyses."""
        now = datetime.now()
        upcoming = []
        
        for schedule in self.schedules.values():
            if schedule.enabled and schedule.next_run:
                upcoming.append({
                    'id': schedule.id,
                    'sport': schedule.sport,
                    'league': schedule.league,
                    'frequency': schedule.frequency.value,
                    'next_run': schedule.next_run.isoformat(),
                    'minutes_until': max(0, int((schedule.next_run - now).total_seconds() / 60))
                })
        
        # Sort by next run time
        upcoming.sort(key=lambda x: x['next_run'])
        return upcoming


# Singleton instance
_scheduler = None

def get_scheduler() -> AnalysisScheduler:
    """Get singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AnalysisScheduler()
    return _scheduler


# Default analysis function
async def default_analysis_function(schedule: ScheduledAnalysis) -> List[Dict]:
    """
    Default analysis function that uses available scrapers.
    """
    from core.ml import get_prediction_service
    
    results = []
    
    try:
        # Try to get upcoming matches from scrapers
        from data.scrapers import get_upcoming_matches
        
        matches = await get_upcoming_matches(schedule.sport, schedule.league)
        
        service = get_prediction_service()
        
        for match in matches[:20]:  # Analyze up to 20 matches
            prediction = service.predict(
                sport=schedule.sport,
                home_player=match.get('home_team'),
                away_player=match.get('away_team'),
                odds=match.get('odds', {})
            )
            
            results.append({
                'match_id': match.get('id'),
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'prediction': {
                    'home_prob': prediction.home_win_prob,
                    'away_prob': prediction.away_win_prob,
                    'confidence': prediction.confidence
                }
            })
            
    except Exception as e:
        logger.error(f"Default analysis failed: {e}")
    
    return results
