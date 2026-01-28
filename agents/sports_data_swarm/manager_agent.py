"""
Manager Agent - Coordinates the entire data collection process.
Orchestrates all other agents in the swarm.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from base_agent import BaseAgent, TaskResult, AgentMessage, AgentStatus
import logging

logger = logging.getLogger(__name__)


class ManagerAgent(BaseAgent):
    """
    Manager Agent that coordinates the data collection swarm.
    
    Responsibilities:
    - Initialize and manage all sport-specific agents
    - Coordinate data acquisition workflow
    - Monitor progress and handle failures
    - Collect final results
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ManagerAgent", config)
        self.sport_agents = {}
        self.acquisition_agent = None
        self.formatting_agent = None
        self.storage_agent = None
        self.evaluator_agents = {}
        self.collection_results = {}
        self.target_records = config.get('target_records', 10000) if config else 10000
        
    def register_sport_agent(self, sport: str, agent: BaseAgent):
        """Register a sport-specific agent."""
        self.sport_agents[sport] = agent
        logger.info(f"Registered agent for {sport}")
    
    def register_acquisition_agent(self, agent: BaseAgent):
        """Register data acquisition agent."""
        self.acquisition_agent = agent
        logger.info("Registered DataAcquisitionAgent")
    
    def register_formatting_agent(self, agent: BaseAgent):
        """Register formatting agent."""
        self.formatting_agent = agent
        logger.info("Registered FormattingAgent")
    
    def register_storage_agent(self, agent: BaseAgent):
        """Register storage agent."""
        self.storage_agent = agent
        logger.info("Registered StorageAgent")
    
    def register_evaluator_agent(self, sport: str, agent: BaseAgent):
        """Register evaluator agent for a sport."""
        self.evaluator_agents[sport] = agent
        logger.info(f"Registered evaluator for {sport}")
    
    async def execute(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute the full data collection workflow.
        
        Task params:
        - sports: List of sports to collect data for
        - target_records: Target number of records per sport
        - date_range: Date range for historical data
        """
        self.status = AgentStatus.RUNNING
        sports = task.get('sports', ['basketball', 'volleyball', 'handball', 'tennis'])
        target_records = task.get('target_records', self.target_records)
        date_range = task.get('date_range', {'start': '2020-01-01', 'end': '2025-12-31'})
        
        logger.info("=" * 60)
        logger.info("SPORTS DATA SWARM - Starting Collection Process")
        logger.info("=" * 60)
        logger.info(f"Sports: {', '.join(sports)}")
        logger.info(f"Target records per sport: {target_records}")
        logger.info(f"Date range: {date_range['start']} to {date_range['end']}")
        logger.info("=" * 60)
        
        overall_results = {}
        
        try:
            # Phase 1: Collect data for each sport
            for sport in sports:
                logger.info(f"\n--- Processing {sport.upper()} ---")
                result = await self._collect_sport_data(sport, target_records, date_range)
                overall_results[sport] = result
                
            # Phase 2: Final evaluation
            logger.info("\n--- Final Evaluation Phase ---")
            evaluation_results = await self._run_final_evaluation(sports, overall_results)
            
            # Save final report
            await self._save_final_report(overall_results, evaluation_results)
            
            total_records = sum(r.records_processed for r in overall_results.values() if r)
            
            self.status = AgentStatus.COMPLETED
            logger.info("\n" + "=" * 60)
            logger.info("COLLECTION PROCESS COMPLETED")
            logger.info(f"Total records collected: {total_records}")
            logger.info("=" * 60)
            
            return TaskResult(
                success=True,
                data={'results': {s: r.data if r else {} for s, r in overall_results.items()}},
                records_processed=total_records,
                metadata={'evaluation': evaluation_results}
            )
            
        except Exception as e:
            logger.error(f"Manager agent error: {e}")
            self.status = AgentStatus.ERROR
            return TaskResult(success=False, error=str(e))
    
    async def _collect_sport_data(self, sport: str, target: int, date_range: Dict) -> TaskResult:
        """Coordinate data collection for a single sport."""
        sport_agent = self.sport_agents.get(sport)
        if not sport_agent:
            logger.error(f"No agent registered for {sport}")
            return TaskResult(success=False, error=f"No agent for {sport}")
        
        # Step 1: Sport agent plans the collection strategy
        logger.info(f"[Manager] Getting collection strategy for {sport}...")
        strategy = await sport_agent.execute({
            'action': 'plan_collection',
            'target_records': target,
            'date_range': date_range
        })
        
        if not strategy.success:
            logger.error(f"Strategy planning failed for {sport}")
            return strategy
        
        # Step 2: Data acquisition
        logger.info(f"[Manager] Acquiring data for {sport}...")
        acquisition_task = {
            'sport': sport,
            'strategy': strategy.data,
            'target_records': target,
            'date_range': date_range
        }
        acquisition_result = await self.acquisition_agent.execute(acquisition_task)
        
        if not acquisition_result.success:
            logger.error(f"Data acquisition failed for {sport}")
            return acquisition_result
        
        logger.info(f"[Manager] Acquired {acquisition_result.records_processed} raw records for {sport}")
        
        # Step 3: Formatting
        logger.info(f"[Manager] Formatting data for {sport}...")
        formatting_task = {
            'sport': sport,
            'raw_data': acquisition_result.data,
            'schema': strategy.data.get('schema', {})
        }
        formatting_result = await self.formatting_agent.execute(formatting_task)
        
        if not formatting_result.success:
            logger.error(f"Data formatting failed for {sport}")
            return formatting_result
        
        logger.info(f"[Manager] Formatted {formatting_result.records_processed} records for {sport}")
        
        # Step 4: Storage
        logger.info(f"[Manager] Saving data for {sport}...")
        storage_task = {
            'sport': sport,
            'data': formatting_result.data,
            'output_dir': 'datasets/sports_data/processed'
        }
        storage_result = await self.storage_agent.execute(storage_task)
        
        if not storage_result.success:
            logger.error(f"Data storage failed for {sport}")
            return storage_result
        
        # Step 5: Evaluation
        logger.info(f"[Manager] Evaluating data quality for {sport}...")
        evaluator = self.evaluator_agents.get(sport)
        if evaluator:
            eval_task = {
                'sport': sport,
                'data': formatting_result.data,
                'dataset_path': storage_result.data.get('filepath')
            }
            eval_result = await evaluator.execute(eval_task)
            logger.info(f"[Manager] Evaluation for {sport}: {eval_result.records_processed} records passed")
        
        logger.info(f"[Manager] Completed {sport}: {storage_result.records_processed} records saved")
        
        return TaskResult(
            success=True,
            data={
                'acquisition': acquisition_result.data,
                'formatting': formatting_result.data,
                'storage': storage_result.data
            },
            records_processed=storage_result.records_processed
        )
    
    async def _run_final_evaluation(self, sports: List[str], results: Dict[str, TaskResult]) -> Dict:
        """Run final evaluation across all sports."""
        evaluation_summary = {}
        
        for sport in sports:
            evaluator = self.evaluator_agents.get(sport)
            if evaluator and sport in results:
                result = results[sport]
                if result and result.success:
                    eval_summary = await evaluator.execute({
                        'action': 'summarize',
                        'data': result.data
                    })
                    evaluation_summary[sport] = eval_summary.data if eval_summary.success else {}
        
        return evaluation_summary
    
    async def _save_final_report(self, results: Dict[str, TaskResult], evaluation: Dict):
        """Save final collection report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_sports': len(results),
                'total_records': sum(r.records_processed for r in results.values() if r),
                'sports': {}
            },
            'details': {},
            'evaluation': evaluation
        }
        
        for sport, result in results.items():
            if result:
                report['summary']['sports'][sport] = {
                    'records': result.records_processed,
                    'success': result.success
                }
                if result.data:
                    report['details'][sport] = result.data
        
        filepath = 'datasets/sports_data/collection_report.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"[Manager] Final report saved: {filepath}")
