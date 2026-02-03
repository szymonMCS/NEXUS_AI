"""Test ensemble model system."""

import asyncio
import sys

async def test_ensemble():
    print('='*60)
    print('Testing Ensemble Model System')
    print('='*60)
    
    # Test 1: Basic ensemble prediction
    print('\n[1/4] Testing ensemble prediction...')
    from core.ml import get_prediction_service
    
    service = get_prediction_service()
    
    # Single prediction
    result = service.predict('tennis', 'Djokovic', 'Alcaraz', {'surface': 'hard'})
    print(f'  Home prob: {result.home_win_prob:.2f}')
    print(f'  Away prob: {result.away_win_prob:.2f}')
    print(f'  Confidence: {result.confidence:.2f}')
    print(f'  Model: {result.model_name}')
    
    if hasattr(result, '_ensemble_meta'):
        meta = result._ensemble_meta
        print(f'  Agreement: {meta.get("agreement_score", 0):.2f}')
        print(f'  Action: {meta.get("recommended_action", "unknown")}')
    print('  OK')
    
    # Test 2: Value betting with ensemble
    print('\n[2/4] Testing value betting with ensemble...')
    explanation = service.predict_with_value(
        'tennis', 'Djokovic', 'Alcaraz',
        home_odds=1.8, away_odds=2.1
    )
    print(f'  EV: {explanation.expected_value:.2f}')
    print(f'  Selected: {explanation.selected_outcome}')
    print(f'  Agreement: {explanation.agreement_score:.2f}')
    print(f'  Action: {explanation.recommended_action}')
    print('  OK')
    
    # Test 3: Model info
    print('\n[3/4] Testing model info...')
    info = service.get_model_info('tennis')
    print(f'  Available models: {info["available_models"]}')
    print(f'  Ensemble enabled: {info["ensemble"]["enabled"]}')
    print(f'  Primary model: {info["ensemble"]["primary_model"]}')
    print('  OK')
    
    # Test 4: Direct aggregator test
    print('\n[4/4] Testing EnsembleAggregator directly...')
    from core.ml import EnsembleAggregator, PredictionResult
    
    agg = EnsembleAggregator(min_models=1)
    
    # Create mock predictions
    predictions = {
        'Model_A': PredictionResult(home_win_prob=0.6, away_win_prob=0.4, confidence=0.7, model_name='Model_A'),
        'Model_B': PredictionResult(home_win_prob=0.55, away_win_prob=0.45, confidence=0.6, model_name='Model_B'),
        'StatisticalEnsemble': PredictionResult(home_win_prob=0.5, away_win_prob=0.5, confidence=0.4, model_name='StatisticalEnsemble'),
    }
    
    ensemble = agg.aggregate(predictions, {})
    print(f'  Aggregated home: {ensemble.home_win_prob:.2f}')
    print(f'  Confidence: {ensemble.confidence:.2f}')
    print(f'  Agreement: {ensemble.agreement_score:.2f}')
    print(f'  Primary: {ensemble.primary_model}')
    print('  OK')
    
    print('\n' + '='*60)
    print('ALL ENSEMBLE TESTS PASSED')
    print('='*60)

if __name__ == '__main__':
    try:
        asyncio.run(test_ensemble())
        sys.exit(0)
    except Exception as e:
        print(f'\nTest failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
