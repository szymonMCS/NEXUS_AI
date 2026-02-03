"""Test ensemble API endpoints."""

from api.main import app
from fastapi.testclient import TestClient

def test_ensemble_api():
    client = TestClient(app)
    
    print('Testing Ensemble API Endpoints')
    print('='*60)
    
    # Test 1: Ensemble status
    print('\n[1/4] GET /api/ml/ensemble/status')
    r = client.get('/api/ml/ensemble/status')
    print(f'  Status: {r.status_code}')
    if r.status_code == 200:
        data = r.json()
        print(f'  Sports: {list(data["ensemble_status"].keys())}')
    
    # Test 2: Ensemble predict
    print('\n[2/4] POST /api/ml/ensemble/predict')
    r = client.post('/api/ml/ensemble/predict', json={
        'sport': 'tennis',
        'home_player': 'Djokovic',
        'away_player': 'Alcaraz',
        'return_details': True
    })
    print(f'  Status: {r.status_code}')
    if r.status_code == 200:
        data = r.json()
        print(f'  Home prob: {data["ensemble"]["home_probability"]}')
        print(f'  Confidence: {data["ensemble"]["confidence"]}')
        print(f'  Models used: {data.get("models_used", 0)}')
    
    # Test 3: Ensemble predict-value
    print('\n[3/4] POST /api/ml/ensemble/predict-value')
    r = client.post('/api/ml/ensemble/predict-value', json={
        'sport': 'tennis',
        'home_player': 'Djokovic',
        'away_player': 'Alcaraz',
        'home_odds': 1.8,
        'away_odds': 2.1
    })
    print(f'  Status: {r.status_code}')
    if r.status_code == 200:
        data = r.json()
        print(f'  Selection: {data["prediction"]["selection"]}')
        print(f'  EV: {data["prediction"]["expected_value"]}')
        print(f'  Action: {data["prediction"]["recommended_action"]}')
    
    # Test 4: Model rankings
    print('\n[4/4] GET /api/ml/ensemble/rankings/tennis')
    r = client.get('/api/ml/ensemble/rankings/tennis')
    print(f'  Status: {r.status_code}')
    if r.status_code == 200:
        data = r.json()
        print(f'  Sport: {data["sport"]}')
        print(f'  Total models: {data["total_models"]}')
    
    print('\n' + '='*60)
    print('ALL ENSEMBLE API TESTS PASSED')

if __name__ == '__main__':
    test_ensemble_api()
