#!/usr/bin/env python3
"""
Sports Data Swarm - Demo Script
Pokazuje zaawansowane użycie systemu agentów.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from base_agent import BaseAgent, TaskResult
from manager_agent import ManagerAgent
from sport_agents import BasketballAgent, VolleyballAgent, HandballAgent, TennisAgent
from data_acquisition_agent import DataAcquisitionAgent
from formatting_agent import FormattingAgent
from storage_agent import StorageAgent
from evaluator_agents import (
    BasketballEvaluatorAgent, VolleyballEvaluatorAgent, 
    HandballEvaluatorAgent, TennisEvaluatorAgent
)


async def demo_single_collection():
    """Demo: Zbierz dane dla pojedynczego sportu."""
    print("\n" + "="*60)
    print("DEMO 1: Pojedyncza kolekcja - Koszykówka")
    print("="*60 + "\n")
    
    # Inicjalizacja agentów
    manager = ManagerAgent(config={'target_records': 50})
    
    manager.register_sport_agent('basketball', BasketballAgent())
    manager.register_acquisition_agent(DataAcquisitionAgent())
    manager.register_formatting_agent(FormattingAgent())
    manager.register_storage_agent(StorageAgent())
    manager.register_evaluator_agent('basketball', BasketballEvaluatorAgent())
    
    # Uruchomienie kolekcji
    result = await manager.execute({
        'sports': ['basketball'],
        'target_records': 50,
        'date_range': {'start': '2024-01-01', 'end': '2024-12-31'}
    })
    
    if result.success:
        print(f"\nZebrano {result.records_processed} rekordow!")
        print(f"Plik: {result.data.get('results', {}).get('basketball', {}).get('storage', {}).get('filepath')}")
    else:
        print(f"Błąd: {result.error}")


async def demo_custom_pipeline():
    """Demo: Własny pipeline bez managera."""
    print("\n" + "="*60)
    print("DEMO 2: Wlasny pipeline - Tenis")
    print("="*60 + "\n")
    
    # Step 1: Planowanie
    print("Step 1: Planowanie kolekcji...")
    tennis_agent = TennisAgent()
    plan = await tennis_agent.execute({
        'action': 'plan_collection',
        'target_records': 10,
        'date_range': {'start': '2024-01-01', 'end': '2024-12-31'}
    })
    
    if plan.success:
        print(f"Zaplanowano: {plan.data.get('target_records')} rekordow")
        print(f"Turnieje: {', '.join(plan.data.get('tournaments', [])[:3])}...")
    
    # Step 2: Pozyskiwanie danych
    print("\nStep 2: Pozyskiwanie danych...")
    acquisition = DataAcquisitionAgent()
    raw_data = await acquisition.execute({
        'sport': 'tennis',
        'strategy': plan.data,
        'target_records': 10,
        'date_range': {'start': '2024-01-01', 'end': '2024-12-31'}
    })
    
    if raw_data.success:
        print(f"Pozyskano: {raw_data.records_processed} rekordow")
    
    # Step 3: Formatowanie
    print("\nStep 3: Formatowanie danych...")
    formatter = FormattingAgent()
    formatted = await formatter.execute({
        'sport': 'tennis',
        'raw_data': raw_data.data,
        'schema': plan.data.get('schema', {})
    })
    
    if formatted.success:
        print(f"Sformatowano: {formatted.records_processed} rekordow")
        print(f"Zgodnosc ze schematem: {formatted.metadata.get('compliance_rate', 0)*100:.1f}%")
    
    # Step 4: Zapis
    print("\nStep 4: Zapis danych...")
    storage = StorageAgent()
    saved = await storage.execute({
        'sport': 'tennis',
        'data': formatted.data,
        'format': 'json',
        'split': False
    })
    
    if saved.success:
        print(f"Zapisano do: {saved.data.get('filepath')}")
    
    # Step 5: Ewaluacja
    print("\nStep 5: Ewaluacja jakosci...")
    evaluator = TennisEvaluatorAgent()
    evaluation = await evaluator.execute({
        'sport': 'tennis',
        'data': formatted.data
    })
    
    if evaluation.success:
        print(f"Ocena jakosci: {evaluation.data.get('quality_score')}/100")
        print(f"Ocena: {evaluation.data.get('quality_grade')}")


async def demo_data_analysis():
    """Demo: Analiza zebranych danych."""
    print("\n" + "="*60)
    print("DEMO 3: Analiza danych koszykowki")
    print("="*60 + "\n")
    
    # Tworzymy przykładowe dane
    sample_games = [
        {
            'game_id': 'nba_001',
            'date': '2024-01-15',
            'home_team': 'Lakers',
            'away_team': 'Celtics',
            'home_score': 102,
            'away_score': 98,
            'home_q1': 28, 'home_q2': 24, 'home_q3': 26, 'home_q4': 24,
            'away_q1': 25, 'away_q2': 22, 'away_q3': 28, 'away_q4': 23,
            'league': 'NBA'
        },
        {
            'game_id': 'nba_002',
            'date': '2024-01-16',
            'home_team': 'Warriors',
            'away_team': 'Nets',
            'home_score': 110,
            'away_score': 105,
            'home_q1': 30, 'home_q2': 25, 'home_q3': 28, 'home_q4': 27,
            'away_q1': 28, 'away_q2': 26, 'away_q3': 25, 'away_q4': 26,
            'league': 'NBA'
        },
        {
            'game_id': 'euro_001',
            'date': '2024-01-17',
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'home_score': 85,
            'away_score': 78,
            'home_q1': 22, 'home_q2': 20, 'home_q3': 21, 'home_q4': 22,
            'away_q1': 20, 'away_q2': 18, 'away_q3': 20, 'away_q4': 20,
            'league': 'EuroLeague'
        }
    ]
    
    # Formatowanie
    formatter = FormattingAgent()
    result = await formatter.execute({
        'sport': 'basketball',
        'raw_data': {'records': sample_games},
        'schema': {
            'required_fields': ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score'],
            'optional_fields': ['league', 'venue']
        }
    })
    
    if result.success:
        records = result.data.get('records', [])
        
        # Analiza
        print("Statystyki:")
        
        # Średnia punktów
        total_points = [r['home_score'] + r['away_score'] for r in records]
        avg_points = sum(total_points) / len(total_points)
        print(f"  Srednia punktow na mecz: {avg_points:.1f}")
        
        # Wygrane u siebie
        home_wins = sum(1 for r in records if r['home_score'] > r['away_score'])
        print(f"  Wygrane u siebie: {home_wins}/{len(records)} ({100*home_wins/len(records):.0f}%)")
        
        # Różnica punktów
        margins = [abs(r['home_score'] - r['away_score']) for r in records]
        avg_margin = sum(margins) / len(margins)
        print(f"  Srednia roznica punktow: {avg_margin:.1f}")
        
        # Rozkład lig
        leagues = {}
        for r in records:
            league = r.get('league', 'Unknown')
            leagues[league] = leagues.get(league, 0) + 1
        
        print(f"  Rozklad lig:")
        for league, count in leagues.items():
            print(f"    {league}: {count} meczow")
        
        # Cechy ML
        print(f"\nCechy ML:")
        for r in records[:2]:
            print(f"  {r['home_team']} vs {r['away_team']}:")
            print(f"    point_diff: {r.get('point_diff')}")
            print(f"    total_points: {r.get('total_points')}")
            print(f"    home_win: {r.get('home_win')}")


async def main():
    """Main demo function."""
    print("\n" + "="*60)
    print("SPORTS DATA SWARM - DEMOS")
    print("="*60)
    
    # Demo 2 (własny pipeline) jest najbardziej pouczające
    await demo_custom_pipeline()
    
    # Demo 3 (analiza danych)
    await demo_data_analysis()
    
    print("\n" + "="*60)
    print("DEMOS ZAKONCZONE")
    print("="*60)
    print("\nAby uruchomic pelna kolekcje:")
    print("  python run_collection.py --all --target 1000")


if __name__ == '__main__':
    asyncio.run(main())
