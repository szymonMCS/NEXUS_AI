"""
Test API Connections.

Verifies all configured API keys work correctly.
Run: python scripts/test_api_connections.py
"""

import os
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, date

# Load .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def test_odds_api():
    """Test The Odds API connection."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {"status": "SKIP", "message": "ODDS_API_KEY not configured"}

    url = "https://api.the-odds-api.com/v4/sports"
    params = {"apiKey": api_key}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "OK",
                    "message": f"Found {len(data)} sports",
                    "sample": [s["title"] for s in data[:5]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_football_data_org():
    """Test Football-Data.org API connection."""
    api_key = os.getenv("X-Auth-Token")
    if not api_key:
        return {"status": "SKIP", "message": "X-Auth-Token not configured"}

    url = "https://api.football-data.org/v4/competitions"
    headers = {"X-Auth-Token": api_key}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                competitions = data.get("competitions", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(competitions)} competitions",
                    "sample": [c["name"] for c in competitions[:5]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_api_sports():
    """Test API-Sports.io connection (basketball/tennis)."""
    api_key = os.getenv("x-apisports-key")
    if not api_key:
        return {"status": "SKIP", "message": "x-apisports-key not configured"}

    # Test basketball endpoint
    url = "https://v1.basketball.api-sports.io/leagues"
    headers = {"x-apisports-key": api_key}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                leagues = data.get("response", [])
                remaining = response.headers.get("x-ratelimit-requests-remaining", "?")
                return {
                    "status": "OK",
                    "message": f"Found {len(leagues)} basketball leagues",
                    "remaining_requests": remaining,
                    "sample": [l["name"] for l in leagues[:5]] if leagues else []
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_rapidapi_tennis():
    """Test RapidAPI Tennis connection."""
    api_key = os.getenv("X-RapidAPI-Key")
    if not api_key:
        return {"status": "SKIP", "message": "X-RapidAPI-Key not configured"}

    # Tennis Live Data API
    url = "https://tennis-live-data.p.rapidapi.com/tournaments/ATP"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "tennis-live-data.p.rapidapi.com"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(results)} ATP tournaments",
                    "sample": [t.get("name", "?") for t in results[:5]] if results else []
                }
            elif response.status_code == 403:
                return {"status": "ERROR", "message": "API key invalid or not subscribed to this API"}
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_pandascore():
    """Test PandaScore (eSports) API connection."""
    api_key = os.getenv("PANDASCORE_API_KEY")
    if not api_key:
        return {"status": "SKIP", "message": "PANDASCORE_API_KEY not configured"}

    url = "https://api.pandascore.co/lol/matches/upcoming"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"per_page": 5}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "OK",
                    "message": f"Found {len(data)} upcoming LoL matches",
                    "sample": [f"{m.get('name', '?')}" for m in data[:3]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_newsapi():
    """Test NewsAPI connection."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return {"status": "SKIP", "message": "NEWSAPI_KEY not configured"}

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": api_key,
        "category": "sports",
        "country": "us",
        "pageSize": 5
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                return {
                    "status": "OK",
                    "message": f"Found {data.get('totalResults', 0)} sports articles",
                    "sample": [a.get("title", "?")[:50] for a in articles[:3]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_brave_search():
    """Test Brave Search API connection."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return {"status": "SKIP", "message": "BRAVE_API_KEY not configured"}

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {"q": "tennis ATP rankings", "count": 5}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("web", {}).get("results", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(results)} search results",
                    "sample": [r.get("title", "?")[:40] for r in results[:3]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_serper():
    """Test Serper (Google Search) API connection."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return {"status": "SKIP", "message": "SERPER_API_KEY not configured"}

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": "NBA basketball scores today"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                organic = data.get("organic", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(organic)} search results",
                    "sample": [r.get("title", "?")[:40] for r in organic[:3]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_openf1():
    """Test OpenF1 API (free, no key needed)."""
    url = "https://api.openf1.org/v1/sessions"
    params = {"year": 2024, "session_type": "Race"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "OK",
                    "message": f"Found {len(data)} F1 race sessions",
                    "sample": [f"{s.get('location', '?')} GP" for s in data[:5]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_ergast_f1():
    """Test Ergast F1 API (free, no key needed)."""
    url = "http://ergast.com/api/f1/2024/drivers.json"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                drivers = data.get("MRData", {}).get("DriverTable", {}).get("Drivers", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(drivers)} F1 drivers",
                    "sample": [f"{d.get('givenName', '')} {d.get('familyName', '')}" for d in drivers[:5]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_mlb_stats():
    """Test MLB Stats API (free, no key needed)."""
    url = "https://statsapi.mlb.com/api/v1/teams"
    params = {"sportId": 1}  # MLB

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                teams = data.get("teams", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(teams)} MLB teams",
                    "sample": [t.get("name", "?") for t in teams[:5]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def test_nhl_api():
    """Test NHL API (free, no key needed)."""
    url = "https://api-web.nhl.com/v1/standings/now"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                standings = data.get("standings", [])
                return {
                    "status": "OK",
                    "message": f"Found {len(standings)} NHL team standings",
                    "sample": [t.get("teamName", {}).get("default", "?") for t in standings[:5]]
                }
            else:
                return {"status": "ERROR", "message": f"HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


async def main():
    """Run all API tests."""
    print("=" * 60)
    print("NEXUS API Connection Tests")
    print("=" * 60)
    print()

    tests = [
        ("The Odds API", test_odds_api),
        ("Football-Data.org", test_football_data_org),
        ("API-Sports.io", test_api_sports),
        ("RapidAPI Tennis", test_rapidapi_tennis),
        ("PandaScore (eSports)", test_pandascore),
        ("NewsAPI", test_newsapi),
        ("Brave Search", test_brave_search),
        ("Serper (Google)", test_serper),
        ("OpenF1 (free)", test_openf1),
        ("Ergast F1 (free)", test_ergast_f1),
        ("MLB Stats (free)", test_mlb_stats),
        ("NHL API (free)", test_nhl_api),
    ]

    results = {}
    for name, test_func in tests:
        print(f"Testing {name}...", end=" ", flush=True)
        result = await test_func()
        results[name] = result

        status = result["status"]
        if status == "OK":
            print(f"[OK] {result['message']}")
            if result.get("sample"):
                print(f"   Sample: {result['sample']}")
        elif status == "SKIP":
            print(f"[SKIP] {result['message']}")
        else:
            print(f"[ERROR] {result['message']}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok_count = sum(1 for r in results.values() if r["status"] == "OK")
    skip_count = sum(1 for r in results.values() if r["status"] == "SKIP")
    error_count = sum(1 for r in results.values() if r["status"] == "ERROR")

    print(f"[OK] Working: {ok_count}")
    print(f"[SKIP] Skipped: {skip_count}")
    print(f"[ERROR] Errors: {error_count}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
