# Changelog

All notable changes to NEXUS AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] - 2026-01-21

### Added
- **Statistical Model Agent** (`agents/statistical_model_agent.py`)
  - New agent for statistical predictions
  - Supports all sports: Tennis, Basketball, Greyhound, Handball, Table Tennis
  - Can be used as primary, fallback, or ensemble with LLM
  - Automatic model initialization and routing

- **Frontend Integration Tests** (`frontend/app/src/lib/api.test.ts`)
  - TypeScript tests for API client
  - Tests for all API functions
  - Tests for helper utilities (formatEdge, formatOdds, etc.)
  - WebSocket connection tests

- **API Endpoint Tests** (`tests/test_api_endpoints.py`)
  - 30 comprehensive tests for all endpoints
  - Status, sports, analysis, predictions, value-bets, matches, stats
  - WebSocket progress tracking tests

- **WebSocket Tests** (`tests/test_websocket.py`)
  - Connection manager tests
  - Message format validation
  - Reconnection handling tests

- **New Sports Model Tests** (`tests/test_new_sports_models.py`)
  - 35 tests for Greyhound, Handball, Table Tennis models
  - Model initialization tests
  - Prediction and probability tests
  - Feature weight validation
  - Integration tests

- **Main Entry Point** (`main.py`)
  - CLI interface with argparse
  - Server mode (`python main.py`)
  - Analysis mode (`python main.py --analyze tennis`)
  - Development mode (`python main.py --dev`)
  - System check mode (`python main.py --check`)

- **Startup Scripts**
  - `scripts/start_dev.sh` - Development environment setup
  - `scripts/start_prod.sh` - Production environment setup
  - `scripts/run_tests.sh` - Test runner with coverage

- **Enhanced pyproject.toml**
  - Complete project metadata
  - Code quality tool configuration (black, isort, flake8, mypy)
  - Pytest configuration
  - Optional dependency groups

### Changed
- **Updated README.md**
  - Complete feature documentation
  - Architecture overview with diagrams
  - Sports supported table
  - Project structure documentation
  - Docker deployment instructions

- **API Main Module** (`api/main.py`)
  - Fixed parameter shadowing bug in `get_matches`
  - Improved error handling

- **Frontend API Client** (`frontend/app/src/lib/api.ts`)
  - Enhanced TypeScript interfaces
  - Added handicap prediction support
  - Improved WebSocket reconnection logic

### Fixed
- **Test Suite**
  - All 30 API endpoint tests passing
  - All 35 sports model tests passing
  - Fixed async test configuration issues

### Removed
- ~~`backend_draft/`~~ - Consolidated into main codebase

---

## [2.1.0] - 2026-01-20

### Added
- **Greyhound Racing Model** (`core/models/greyhound_model.py`)
  - SVR/SVM ensemble for position prediction
  - Trap bias factors and early pace rating
  - Trainer form and weight analysis
  - Forecast and tricast combinations

- **Handball Model** (`core/models/handball_model.py`)
  - SEL (Statistically Enhanced Learning) approach
  - CMP-like distribution for goal modeling
  - First half/second half analysis
  - Handicap and total goals markets

- **Table Tennis Model** (`core/models/table_tennis_model.py`)
  - XGBoost/RandomForest/GradientBoosting ensemble
  - Rating, ranking, and form factors
  - Style matchup matrix (offensive/defensive/chopper)
  - Set handicap and total points markets

- **Configuration Updates** (`config/leagues.py`)
  - Added leagues for greyhound, handball, table_tennis
  - `get_supported_sports()` helper function
  - `get_leagues_for_sport()` helper function

### Changed
- **Enhanced Ranker** (`agents/ranker.py`)
  - Now supports all sports
  - Tournament diversification constraint
  - Composite scoring formula: edge^0.5 * quality^0.3 * confidence^0.2

---

## [2.0.0] - 2026-01-19

### Added
- **LangGraph Multi-Agent System**
  - Supervisor Agent orchestration
  - News Analyst Agent (news aggregation, injury extraction)
  - Data Evaluator Agent (quality scoring)
  - Analyst Agent (AI predictions)
  - Ranker Agent (Top 3 selection)
  - Risk Manager Agent (Kelly Criterion)
  - Decision Maker Agent (final recommendations)
  - Bettor Agent (bet placement simulation)

- **Statistical Prediction Models**
  - TennisModel (ELO-based with surface/fatigue factors)
  - BasketballModel (offensive/defensive ratings, rest days)
  - HandicapModel (spread and totals predictions)

- **Data Collection System**
  - Fixture Collector (multi-source aggregation)
  - Flashscore Scraper (Playwright-based)
  - TheSportsDB Client (free API)
  - Odds Merger (multiple bookmakers)

- **Quality Evaluation System**
  - Source Agreement Checker
  - Freshness Checker
  - Web Data Evaluator

- **React Frontend**
  - Vite + TypeScript + shadcn/ui
  - Real-time WebSocket updates
  - Sport selector with beta badges
  - Value bets visualization with quality scores

- **FastAPI Backend**
  - REST API endpoints
  - WebSocket for live updates
  - Prometheus metrics
  - CORS configuration

---

## [1.0.0] - 2026-01-15

### Added
- Initial implementation
- Basic tennis and basketball predictions
- Claude AI integration
- Basic odds comparison

---

## [Unreleased]

### Planned
- Real-time betting integration
- Additional sports (football, cricket, esports)
- Mobile app
- Cloud deployment templates
- Backtesting framework enhancements
- Performance optimizations
