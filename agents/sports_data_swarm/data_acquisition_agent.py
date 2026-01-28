"""
Data Acquisition Agent - Handles web scraping and API calls.
Uses Brave Search and Serper API for data discovery and web scraping for extraction.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import random

from base_agent import BaseAgent, TaskResult, AgentStatus
import logging

logger = logging.getLogger(__name__)


class DataAcquisitionAgent(BaseAgent):
    """
    Agent responsible for acquiring sports data from web sources.
    
    Features:
    - Search-based data discovery (Brave, Serper)
    - Web scraping with BeautifulSoup
    - Multiple source aggregation
    - Rate limiting and retry logic
    - Data caching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("DataAcquisitionAgent", config)
        self.cache = {}
        self.rate_limit_delay = config.get('rate_limit_delay', 1) if config else 1
        self.max_retries = config.get('max_retries', 3) if config else 3
        
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute data acquisition task.
        
        Task params:
        - sport: Sport to collect data for
        - strategy: Collection strategy from sport agent
        - target_records: Target number of records
        - date_range: Date range for data
        """
        self.status = AgentStatus.RUNNING
        
        try:
            sport = task['sport']
            strategy = task.get('strategy', {})
            target = task.get('target_records', 10000)
            date_range = task.get('date_range', {})
            
            logger.info(f"[{self.name}] Starting data acquisition for {sport}")
            
            all_records = []
            
            # Method 1: Search-based data discovery
            search_records = await self._acquire_via_search(sport, strategy, target // 2, date_range)
            all_records.extend(search_records)
            logger.info(f"[{self.name}] Acquired {len(search_records)} records via search")
            
            # Method 2: Direct web scraping from known sources
            if len(all_records) < target:
                remaining = target - len(all_records)
                web_records = await self._acquire_via_web_scraping(sport, strategy, remaining, date_range)
                all_records.extend(web_records)
                logger.info(f"[{self.name}] Acquired {len(web_records)} records via web scraping")
            
            # Method 3: API-based sources (if available)
            if len(all_records) < target:
                remaining = target - len(all_records)
                api_records = await self._acquire_via_api(sport, strategy, remaining, date_range)
                all_records.extend(api_records)
                logger.info(f"[{self.name}] Acquired {len(api_records)} records via API")
            
            # Deduplicate records
            unique_records = self._deduplicate_records(all_records)
            
            logger.info(f"[{self.name}] Total unique records acquired: {len(unique_records)}")
            
            self.status = AgentStatus.COMPLETED
            return TaskResult(
                success=True,
                data={'records': unique_records, 'raw_count': len(all_records)},
                records_processed=len(unique_records),
                metadata={'sources_used': strategy.get('data_sources', [])}
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            self.status = AgentStatus.ERROR
            return TaskResult(success=False, error=str(e))
    
    async def _acquire_via_search(self, sport: str, strategy: Dict, target: int, 
                                   date_range: Dict) -> List[Dict]:
        """Acquire data using search APIs."""
        records = []
        queries = strategy.get('search_queries', [])
        
        # Limit queries based on target
        queries_needed = min(len(queries), max(1, target // 100))
        selected_queries = random.sample(queries, min(queries_needed, len(queries))) if len(queries) > queries_needed else queries
        
        logger.info(f"[{self.name}] Searching with {len(selected_queries)} queries")
        
        for i, query in enumerate(selected_queries):
            if len(records) >= target:
                break
                
            self.log_progress(f"Search query {i+1}/{len(selected_queries)}", i+1, len(selected_queries))
            
            # Try Brave Search
            try:
                brave_results = await self.search_brave(query, count=20)
                await asyncio.sleep(self.rate_limit_delay)
                
                for result in brave_results[:5]:  # Process top 5 results
                    parsed = await self._parse_search_result(result, sport)
                    if parsed:
                        records.extend(parsed)
                        
            except Exception as e:
                logger.warning(f"Brave search error: {e}")
            
            # Try Serper if needed
            if len(records) < target * (i + 1) // len(selected_queries):
                try:
                    serper_results = await self.search_serper(query, num_results=20)
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    for result in serper_results[:5]:
                        parsed = await self._parse_search_result(result, sport)
                        if parsed:
                            records.extend(parsed)
                            
                except Exception as e:
                    logger.warning(f"Serper search error: {e}")
        
        return records[:target]
    
    async def _acquire_via_web_scraping(self, sport: str, strategy: Dict, target: int,
                                         date_range: Dict) -> List[Dict]:
        """Acquire data by scraping known web sources."""
        records = []
        sources = strategy.get('web_sources', [])
        
        logger.info(f"[{self.name}] Scraping {len(sources)} web sources")
        
        for i, source in enumerate(sources):
            if len(records) >= target:
                break
                
            url = source.get('url', '')
            priority = source.get('priority', 3)
            
            self.log_progress(f"Scraping {url}", i+1, len(sources))
            
            try:
                content = await self.fetch_url(url)
                if content:
                    parsed = self._parse_html(content, sport, url)
                    if parsed:
                        records.extend(parsed)
                        logger.info(f"[{self.name}] Parsed {len(parsed)} records from {url}")
                        
                await asyncio.sleep(self.rate_limit_delay * priority)
                
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")
        
        return records[:target]
    
    async def _acquire_via_api(self, sport: str, strategy: Dict, target: int,
                                date_range: Dict) -> List[Dict]:
        """Acquire data from free APIs."""
        records = []
        
        # Try TheSportsDB (free tier)
        try:
            thesportsdb_records = await self._fetch_thesportsdb(sport, target // 3)
            records.extend(thesportsdb_records)
            logger.info(f"[{self.name}] Acquired {len(thesportsdb_records)} from TheSportsDB")
        except Exception as e:
            logger.warning(f"TheSportsDB error: {e}")
        
        return records[:target]
    
    async def _parse_search_result(self, result: Dict, sport: str) -> List[Dict]:
        """Parse a search result for data extraction."""
        records = []
        url = result.get('url', '')
        
        # Skip if already cached
        if url in self.cache:
            return self.cache[url]
        
        try:
            content = await self.fetch_url(url)
            if content:
                parsed = self._parse_html(content, sport, url)
                self.cache[url] = parsed
                return parsed
        except Exception as e:
            logger.debug(f"Error parsing {url}: {e}")
        
        return records
    
    def _parse_html(self, html: str, sport: str, url: str) -> List[Dict]:
        """Parse HTML content based on sport type."""
        records = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        if sport == 'basketball':
            records = self._parse_basketball_html(soup, url)
        elif sport == 'volleyball':
            records = self._parse_volleyball_html(soup, url)
        elif sport == 'handball':
            records = self._parse_handball_html(soup, url)
        elif sport == 'tennis':
            records = self._parse_tennis_html(soup, url)
        
        return records
    
    def _parse_basketball_html(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Parse basketball-specific HTML structures."""
        records = []
        
        # Look for common table structures
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    try:
                        record = self._extract_basketball_record(cells)
                        if record:
                            record['source_url'] = url
                            records.append(record)
                    except:
                        continue
        
        # Also look for div-based score displays
        score_divs = soup.find_all('div', class_=re.compile(r'score|result|match', re.I))
        for div in score_divs:
            try:
                record = self._extract_from_score_div(div, 'basketball')
                if record:
                    record['source_url'] = url
                    records.append(record)
            except:
                continue
        
        return records
    
    def _parse_volleyball_html(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Parse volleyball-specific HTML structures."""
        records = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    try:
                        record = self._extract_volleyball_record(cells)
                        if record:
                            record['source_url'] = url
                            records.append(record)
                    except:
                        continue
        
        return records
    
    def _parse_handball_html(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Parse handball-specific HTML structures."""
        records = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    try:
                        record = self._extract_handball_record(cells)
                        if record:
                            record['source_url'] = url
                            records.append(record)
                    except:
                        continue
        
        return records
    
    def _parse_tennis_html(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Parse tennis-specific HTML structures."""
        records = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    try:
                        record = self._extract_tennis_record(cells)
                        if record:
                            record['source_url'] = url
                            records.append(record)
                    except:
                        continue
        
        return records
    
    def _extract_basketball_record(self, cells) -> Optional[Dict]:
        """Extract basketball record from table cells."""
        try:
            text = [c.get_text(strip=True) for c in cells]
            
            # Try to identify score pattern (e.g., "102:98" or "102-98")
            score_pattern = re.compile(r'(\d+)\s*[:\-]\s*(\d+)')
            
            for i, t in enumerate(text):
                match = score_pattern.search(t)
                if match:
                    home_score = int(match.group(1))
                    away_score = int(match.group(2))
                    
                    # Assume teams are before and after score
                    home_team = text[max(0, i-1)] if i > 0 else 'Unknown'
                    away_team = text[min(len(text)-1, i+1)] if i < len(text)-1 else 'Unknown'
                    
                    return {
                        'game_id': f"game_{hash(tuple(text))}",
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'league': 'Unknown',
                        'season': '2024-25'
                    }
            return None
        except:
            return None
    
    def _extract_volleyball_record(self, cells) -> Optional[Dict]:
        """Extract volleyball record from table cells."""
        try:
            text = [c.get_text(strip=True) for c in cells]
            
            # Look for set scores (e.g., "3:1" or "3-1")
            set_pattern = re.compile(r'(\d)\s*[:\-]\s*(\d)')
            
            for i, t in enumerate(text):
                match = set_pattern.search(t)
                if match and int(match.group(1)) <= 3 and int(match.group(2)) <= 3:
                    home_sets = int(match.group(1))
                    away_sets = int(match.group(2))
                    
                    home_team = text[max(0, i-1)] if i > 0 else 'Unknown'
                    away_team = text[min(len(text)-1, i+1)] if i < len(text)-1 else 'Unknown'
                    
                    return {
                        'match_id': f"match_{hash(tuple(text))}",
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_sets_won': home_sets,
                        'away_sets_won': away_sets,
                        'league': 'Unknown',
                        'season': '2024-25'
                    }
            return None
        except:
            return None
    
    def _extract_handball_record(self, cells) -> Optional[Dict]:
        """Extract handball record from table cells."""
        try:
            text = [c.get_text(strip=True) for c in cells]
            
            # Look for goal scores
            score_pattern = re.compile(r'(\d+)\s*[:\-]\s*(\d+)')
            
            for i, t in enumerate(text):
                match = score_pattern.search(t)
                if match:
                    home_score = int(match.group(1))
                    away_score = int(match.group(2))
                    
                    # Handball scores are typically 20-40 goals
                    if 10 <= home_score <= 50 and 10 <= away_score <= 50:
                        home_team = text[max(0, i-1)] if i > 0 else 'Unknown'
                        away_team = text[min(len(text)-1, i+1)] if i < len(text)-1 else 'Unknown'
                        
                        return {
                            'match_id': f"match_{hash(tuple(text))}",
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'league': 'Unknown',
                            'season': '2024-25'
                        }
            return None
        except:
            return None
    
    def _extract_tennis_record(self, cells) -> Optional[Dict]:
        """Extract tennis record from table cells."""
        try:
            text = [c.get_text(strip=True) for c in cells]
            
            # Look for set scores with games (e.g., "6-4 6-3 7-5")
            tennis_pattern = re.compile(r'(\d)[\-:](\d)')
            
            for i, t in enumerate(text):
                matches = tennis_pattern.findall(t)
                if len(matches) >= 2:  # At least 2 sets
                    sets_won_p1 = sum(1 for m in matches if int(m[0]) > int(m[1]))
                    sets_won_p2 = sum(1 for m in matches if int(m[1]) > int(m[0]))
                    
                    player1 = text[max(0, i-1)] if i > 0 else 'Unknown'
                    player2 = text[min(len(text)-1, i+1)] if i < len(text)-1 else 'Unknown'
                    
                    return {
                        'match_id': f"match_{hash(tuple(text))}",
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'player1_name': player1,
                        'player2_name': player2,
                        'player1_sets_won': sets_won_p1,
                        'player2_sets_won': sets_won_p2,
                        'set_scores': t,
                        'tournament': 'Unknown',
                        'surface': 'Hard'
                    }
            return None
        except:
            return None
    
    def _extract_from_score_div(self, div, sport: str) -> Optional[Dict]:
        """Extract record from score div element."""
        # This is a fallback for div-based layouts
        text = div.get_text(strip=True)
        # Implementation would depend on specific site structure
        return None
    
    def _deduplicate_records(self, records: List[Dict]) -> List[Dict]:
        """Remove duplicate records based on unique identifiers."""
        seen = set()
        unique = []
        
        for record in records:
            # Create unique key from essential fields
            key_fields = []
            for k in ['game_id', 'match_id', 'date', 'home_team', 'away_team', 
                      'player1_name', 'player2_name', 'home_score', 'away_score']:
                if k in record:
                    key_fields.append(str(record[k]))
            
            key = tuple(key_fields)
            
            if key not in seen and len(key_fields) >= 4:
                seen.add(key)
                unique.append(record)
        
        return unique
    
    async def _fetch_thesportsdb(self, sport: str, limit: int) -> List[Dict]:
        """Fetch data from TheSportsDB free API."""
        records = []
        
        # Map our sports to TheSportsDB format
        sport_map = {
            'basketball': 'Basketball',
            'volleyball': 'Volleyball',
            'handball': 'Handball',
            'tennis': 'Tennis'
        }
        
        thesport = sport_map.get(sport)
        if not thesport:
            return records
        
        try:
            # Search for past events (requires different approach for free tier)
            url = f"https://www.thesportsdb.com/api/v1/json/3/search_all_leagues.php?s={thesport}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        leagues = data.get('countrys', [])
                        
                        # Get events from top leagues
                        for league in leagues[:5]:
                            league_id = league.get('idLeague')
                            if league_id:
                                # This would need to be adapted based on actual API responses
                                pass
                                
        except Exception as e:
            logger.warning(f"TheSportsDB fetch error: {e}")
        
        return records


# Need to import aiohttp here for the API method
import aiohttp
