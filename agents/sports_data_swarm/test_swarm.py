#!/usr/bin/env python3
"""
Test script for Sports Data Swarm
Quick test to verify all agents are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly from files to avoid __init__.py conflicts
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules
base_agent = load_module("base_agent", project_root / "base_agent.py")
manager_agent = load_module("manager_agent", project_root / "manager_agent.py")
sport_agents = load_module("sport_agents", project_root / "sport_agents.py")
football_agent = load_module("football_agent", project_root / "football_agent.py")
baseball_agent = load_module("baseball_agent", project_root / "baseball_agent.py")
hockey_agent = load_module("hockey_agent", project_root / "hockey_agent.py")
mma_agent = load_module("mma_agent", project_root / "mma_agent.py")
esports_agent = load_module("esports_agent", project_root / "esports_agent.py")
golf_agent = load_module("golf_agent", project_root / "golf_agent.py")
rugby_agent = load_module("rugby_agent", project_root / "rugby_agent.py")
cricket_agent = load_module("cricket_agent", project_root / "cricket_agent.py")
table_tennis_agent = load_module("table_tennis_agent", project_root / "table_tennis_agent.py")
data_acquisition = load_module("data_acquisition_agent", project_root / "data_acquisition_agent.py")
formatting = load_module("formatting_agent", project_root / "formatting_agent.py")
storage = load_module("storage_agent", project_root / "storage_agent.py")
evaluators = load_module("evaluator_agents", project_root / "evaluator_agents.py")
football_eval = load_module("football_evaluator", project_root / "football_evaluator.py")
augmentation = load_module("data_augmentation_agent", project_root / "data_augmentation_agent.py")

# Get classes
ManagerAgent = manager_agent.ManagerAgent
BasketballAgent = sport_agents.BasketballAgent
VolleyballAgent = sport_agents.VolleyballAgent
HandballAgent = sport_agents.HandballAgent
TennisAgent = sport_agents.TennisAgent
FootballAgent = football_agent.FootballAgent
BaseballAgent = baseball_agent.BaseballAgent
HockeyAgent = hockey_agent.HockeyAgent
MMAAgent = mma_agent.MMAAgent
EsportsAgent = esports_agent.EsportsAgent
GolfAgent = golf_agent.GolfAgent
RugbyAgent = rugby_agent.RugbyAgent
CricketAgent = cricket_agent.CricketAgent
TableTennisAgent = table_tennis_agent.TableTennisAgent
DataAcquisitionAgent = data_acquisition.DataAcquisitionAgent
FormattingAgent = formatting.FormattingAgent
StorageAgent = storage.StorageAgent
BasketballEvaluatorAgent = evaluators.BasketballEvaluatorAgent
VolleyballEvaluatorAgent = evaluators.VolleyballEvaluatorAgent
HandballEvaluatorAgent = evaluators.HandballEvaluatorAgent
TennisEvaluatorAgent = evaluators.TennisEvaluatorAgent
FootballEvaluatorAgent = football_eval.FootballEvaluatorAgent
DataAugmentationAgent = augmentation.DataAugmentationAgent


async def test_agents():
    """Test all agents with small sample."""
    print("\n" + "="*60)
    print("  TESTING SPORTS DATA SWARM v2.0")
    print("="*60 + "\n")
    
    # Test 1: Sport Agents (13 sports)
    print("1. Testing Sport Agents...")
    
    basketball_agent = BasketballAgent()
    result = await basketball_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Basketball agent failed"
    print("   [OK] BasketballAgent")
    
    volleyball_agent = VolleyballAgent()
    result = await volleyball_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Volleyball agent failed"
    print("   [OK] VolleyballAgent")
    
    handball_agent = HandballAgent()
    result = await handball_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Handball agent failed"
    print("   [OK] HandballAgent")
    
    tennis_agent = TennisAgent()
    result = await tennis_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Tennis agent failed"
    print("   [OK] TennisAgent")
    
    football_agent = FootballAgent()
    result = await football_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Football agent failed"
    print("   [OK] FootballAgent")
    
    # New sports for betting
    baseball_agent = BaseballAgent()
    result = await baseball_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Baseball agent failed"
    print("   [OK] BaseballAgent (NEW)")
    
    hockey_agent = HockeyAgent()
    result = await hockey_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Hockey agent failed"
    print("   [OK] HockeyAgent (NEW)")
    
    mma_agent = MMAAgent()
    result = await mma_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "MMA agent failed"
    print("   [OK] MMAAgent (NEW)")
    
    esports_agent = EsportsAgent()
    result = await esports_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Esports agent failed"
    print("   [OK] EsportsAgent (NEW)")
    
    golf_agent = GolfAgent()
    result = await golf_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Golf agent failed"
    print("   [OK] GolfAgent (NEW)")
    
    rugby_agent = RugbyAgent()
    result = await rugby_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Rugby agent failed"
    print("   [OK] RugbyAgent (NEW)")
    
    cricket_agent = CricketAgent()
    result = await cricket_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Cricket agent failed"
    print("   [OK] CricketAgent (NEW)")
    
    table_tennis_agent = TableTennisAgent()
    result = await table_tennis_agent.execute({'action': 'plan_collection', 'target_records': 100, 'date_range': {}})
    assert result.success, "Table Tennis agent failed"
    print("   [OK] TableTennisAgent (NEW)")
    
    # Test 2: Data Acquisition Agent
    print("\n2. Testing DataAcquisitionAgent...")
    acquisition_agent = DataAcquisitionAgent()
    print("   [OK] DataAcquisitionAgent initialized")
    
    # Test 3: Formatting Agent
    print("\n3. Testing FormattingAgent...")
    formatting_agent = FormattingAgent()
    
    sample_data = {
        'records': [
            {
                'game_id': 'test_1',
                'date': '2024-01-15',
                'home_team': 'Lakers',
                'away_team': 'Celtics',
                'home_score': 102,
                'away_score': 98,
                'league': 'NBA'
            }
        ]
    }
    schema = {
        'required_fields': ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score'],
        'optional_fields': ['league', 'venue']
    }
    
    result = await formatting_agent.execute({
        'sport': 'basketball',
        'raw_data': sample_data,
        'schema': schema
    })
    assert result.success, "Formatting agent failed"
    print("   [OK] FormattingAgent")
    
    # Test 4: Storage Agent
    print("\n4. Testing StorageAgent...")
    storage_agent = StorageAgent()
    result = await storage_agent.execute({
        'sport': 'basketball',
        'data': result.data,
        'format': 'json',
        'split': False
    })
    assert result.success, "Storage agent failed"
    print("   [OK] StorageAgent")
    print(f"   [FILE] Saved to: {result.data.get('filepath')}")
    
    # Test 5: Evaluator Agents
    print("\n5. Testing Evaluator Agents...")
    
    basketball_eval = BasketballEvaluatorAgent()
    result = await basketball_eval.execute({
        'sport': 'basketball',
        'data': {'records': sample_data['records']}
    })
    assert result.success, "Basketball evaluator failed"
    print("   [OK] BasketballEvaluatorAgent")
    
    football_eval = FootballEvaluatorAgent()
    result = await football_eval.execute({
        'sport': 'football',
        'data': {'records': [{'match_id': 'f1', 'home_team': 'Team A', 'away_team': 'Team B', 'home_goals': 2, 'away_goals': 1}]}
    })
    assert result.success, "Football evaluator failed"
    print("   [OK] FootballEvaluatorAgent (NEW)")
    
    # Test 6: Data Augmentation Agent
    print("\n6. Testing DataAugmentationAgent (NEW)...")
    augmentation_agent = DataAugmentationAgent()
    
    # Test with sample data
    test_records = [
        {'game_id': 'g1', 'date': '2024-01-01', 'home_team': 'A', 'away_team': 'B', 'home_score': 100, 'away_score': 90, 'home_fg_pct': 0.45},
        {'game_id': 'g2', 'date': '2024-01-02', 'home_team': 'C', 'away_team': 'D', 'home_score': 95, 'away_score': 88, 'home_fg_pct': 0.42},
    ]
    
    result = await augmentation_agent.execute({
        'sport': 'basketball',
        'data': {'records': test_records},
        'target_multiplier': 2.0,
        'techniques': ['noise', 'features']
    })
    assert result.success, "Augmentation agent failed"
    print("   [OK] DataAugmentationAgent")
    print(f"   [STATS] {result.data.get('original_count')} -> {result.data.get('augmented_count')} records")
    
    # Test 7: Manager Agent
    print("\n7. Testing ManagerAgent...")
    manager = ManagerAgent(config={'target_records': 10})
    
    manager.register_sport_agent('basketball', basketball_agent)
    manager.register_sport_agent('football', football_agent)
    manager.register_sport_agent('baseball', baseball_agent)
    manager.register_sport_agent('hockey', hockey_agent)
    manager.register_sport_agent('mma', mma_agent)
    manager.register_sport_agent('esports', esports_agent)
    manager.register_acquisition_agent(acquisition_agent)
    manager.register_formatting_agent(formatting_agent)
    manager.register_storage_agent(storage_agent)
    manager.register_evaluator_agent('basketball', basketball_eval)
    manager.register_evaluator_agent('football', football_eval)
    
    print("   [OK] ManagerAgent initialized with all sub-agents")
    
    print("\n" + "="*60)
    print("  [OK] ALL TESTS PASSED!")
    print("="*60 + "\n")
    
    print("System is ready for data collection!")
    print("\nFeatures in v3.0 (Betting Edition):")
    print("  - 13 sports for betting (most popular worldwide)")
    print("  - Football, Basketball, Tennis, Baseball, Hockey")
    print("  - MMA, Esports, Golf, Rugby, Cricket, Table Tennis")
    print("  - Data augmentation (2-5x data multiplication)")
    print("  - Advanced feature engineering")
    print("\nTop betting sports by volume:")
    print("  1. Football/Soccer (~50% of global betting)")
    print("  2. Basketball (~15%)")
    print("  3. Tennis (~12%)")
    print("  4. American Football (~10%)")
    print("  5. Baseball (~5%)")
    print("\nTo run full collection:")
    print("  python run_collection.py --all --target 1000")
    print("\nSpecific sport:")
    print("  python run_collection.py --sport baseball --target 1000 --augment 3.0")


if __name__ == '__main__':
    asyncio.run(test_agents())
