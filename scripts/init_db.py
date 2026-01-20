# scripts/init_db.py
"""
Database initialization script for NEXUS AI.
Sets up PostgreSQL schema, creates tables, and seeds initial data.

Usage:
    python scripts/init_db.py [--reset] [--seed]
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SQL Schema definitions
SCHEMA_SQL = """
-- NEXUS AI Database Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================
-- Core Tables
-- ==========================

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE,
    sport VARCHAR(50) NOT NULL,
    league VARCHAR(255),
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'scheduled',
    home_score INTEGER,
    away_score INTEGER,
    source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matches_sport ON matches(sport);
CREATE INDEX IF NOT EXISTS idx_matches_start_time ON matches(start_time);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);

-- Players/Teams table
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'player' or 'team'
    sport VARCHAR(50) NOT NULL,
    country VARCHAR(100),
    ranking INTEGER,
    rating FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_sport ON entities(sport);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

-- ==========================
-- Predictions Tables
-- ==========================

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    home_probability FLOAT NOT NULL,
    away_probability FLOAT NOT NULL,
    draw_probability FLOAT,
    confidence FLOAT NOT NULL,
    features JSONB DEFAULT '{}',
    explanation TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);

-- Value bets table
CREATE TABLE IF NOT EXISTS value_bets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    bet_on VARCHAR(50) NOT NULL, -- 'home', 'away', 'over', 'under'
    odds FLOAT NOT NULL,
    probability FLOAT NOT NULL,
    edge FLOAT NOT NULL,
    kelly_fraction FLOAT NOT NULL,
    kelly_stake FLOAT NOT NULL,
    quality_multiplier FLOAT DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'identified',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_value_bets_match ON value_bets(match_id);
CREATE INDEX IF NOT EXISTS idx_value_bets_status ON value_bets(status);
CREATE INDEX IF NOT EXISTS idx_value_bets_edge ON value_bets(edge DESC);

-- ==========================
-- Betting Tables
-- ==========================

-- Bets table
CREATE TABLE IF NOT EXISTS bets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id),
    value_bet_id UUID REFERENCES value_bets(id),
    external_bet_id VARCHAR(255),
    bookmaker VARCHAR(100) NOT NULL,
    selection VARCHAR(100) NOT NULL,
    bet_type VARCHAR(50) NOT NULL,
    stake FLOAT NOT NULL,
    odds FLOAT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    profit_loss FLOAT DEFAULT 0,
    placed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settled_at TIMESTAMP WITH TIME ZONE,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_bets_match ON bets(match_id);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
CREATE INDEX IF NOT EXISTS idx_bets_placed_at ON bets(placed_at);

-- Betting sessions table
CREATE TABLE IF NOT EXISTS betting_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    sport VARCHAR(50),
    starting_bankroll FLOAT NOT NULL,
    ending_bankroll FLOAT,
    total_staked FLOAT DEFAULT 0,
    profit_loss FLOAT DEFAULT 0,
    bet_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_sessions_date ON betting_sessions(date);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON betting_sessions(status);

-- ==========================
-- Data Quality Tables
-- ==========================

-- Data quality evaluations
CREATE TABLE IF NOT EXISTS data_quality (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    overall_score FLOAT NOT NULL,
    completeness FLOAT,
    freshness FLOAT,
    source_agreement FLOAT,
    source_count INTEGER,
    issues TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_match ON data_quality(match_id);
CREATE INDEX IF NOT EXISTS idx_quality_score ON data_quality(overall_score DESC);

-- ==========================
-- Historical Data Tables
-- ==========================

-- Historical matches (for backtesting)
CREATE TABLE IF NOT EXISTS historical_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sport VARCHAR(50) NOT NULL,
    league VARCHAR(255),
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    match_date DATE NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    home_odds FLOAT,
    away_odds FLOAT,
    draw_odds FLOAT,
    surface VARCHAR(50), -- for tennis
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_historical_sport ON historical_matches(sport);
CREATE INDEX IF NOT EXISTS idx_historical_date ON historical_matches(match_date);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    sport VARCHAR(50),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_bankroll FLOAT NOT NULL,
    final_bankroll FLOAT NOT NULL,
    total_bets INTEGER NOT NULL,
    win_count INTEGER NOT NULL,
    roi FLOAT NOT NULL,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_model ON backtest_results(model_name);
CREATE INDEX IF NOT EXISTS idx_backtest_roi ON backtest_results(roi DESC);

-- ==========================
-- System Tables
-- ==========================

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    old_value JSONB,
    new_value JSONB,
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

-- System settings
CREATE TABLE IF NOT EXISTS system_settings (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================
-- Views
-- ==========================

-- Active value bets view
CREATE OR REPLACE VIEW active_value_bets AS
SELECT
    vb.*,
    m.home_team,
    m.away_team,
    m.start_time,
    m.league,
    p.confidence as prediction_confidence
FROM value_bets vb
JOIN matches m ON vb.match_id = m.id
LEFT JOIN predictions p ON vb.prediction_id = p.id
WHERE vb.status = 'identified'
AND m.start_time > NOW()
ORDER BY vb.edge DESC;

-- Recent performance view
CREATE OR REPLACE VIEW recent_performance AS
SELECT
    DATE(placed_at) as bet_date,
    COUNT(*) as total_bets,
    SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as losses,
    SUM(stake) as total_staked,
    SUM(profit_loss) as profit_loss,
    AVG(odds) as avg_odds
FROM bets
WHERE placed_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(placed_at)
ORDER BY bet_date DESC;

-- ==========================
-- Functions
-- ==========================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to tables
DO $$
DECLARE
    t text;
BEGIN
    FOR t IN
        SELECT table_name
        FROM information_schema.columns
        WHERE column_name = 'updated_at'
        AND table_schema = 'public'
    LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS update_%I_updated_at ON %I;
            CREATE TRIGGER update_%I_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at();
        ', t, t, t, t);
    END LOOP;
END;
$$ LANGUAGE plpgsql;
"""


SEED_DATA_SQL = """
-- Seed initial data

-- System settings
INSERT INTO system_settings (key, value) VALUES
('app_version', '"1.0.0"'),
('default_bankroll', '1000'),
('kelly_fraction', '0.25'),
('min_edge_popular', '0.03'),
('min_edge_medium', '0.05'),
('min_edge_unpopular', '0.07'),
('max_stake_percent', '0.05'),
('supported_sports', '["tennis", "basketball"]')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

-- Log initialization
INSERT INTO audit_log (action, entity_type, new_value)
VALUES ('database_initialized', 'system', '{"version": "1.0.0"}');
"""


async def get_db_connection():
    """Get database connection."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        return None

    try:
        conn = await asyncpg.connect(settings.database_url)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


async def reset_database(conn):
    """Drop all tables and reset database."""
    logger.warning("Resetting database - all data will be lost!")

    drop_sql = """
    DROP VIEW IF EXISTS active_value_bets CASCADE;
    DROP VIEW IF EXISTS recent_performance CASCADE;
    DROP TABLE IF EXISTS audit_log CASCADE;
    DROP TABLE IF EXISTS system_settings CASCADE;
    DROP TABLE IF EXISTS backtest_results CASCADE;
    DROP TABLE IF EXISTS historical_matches CASCADE;
    DROP TABLE IF EXISTS data_quality CASCADE;
    DROP TABLE IF EXISTS bets CASCADE;
    DROP TABLE IF EXISTS betting_sessions CASCADE;
    DROP TABLE IF EXISTS value_bets CASCADE;
    DROP TABLE IF EXISTS predictions CASCADE;
    DROP TABLE IF EXISTS entities CASCADE;
    DROP TABLE IF EXISTS matches CASCADE;
    """

    await conn.execute(drop_sql)
    logger.info("Database reset complete")


async def init_schema(conn):
    """Initialize database schema."""
    logger.info("Creating database schema...")
    await conn.execute(SCHEMA_SQL)
    logger.info("Schema created successfully")


async def seed_data(conn):
    """Seed initial data."""
    logger.info("Seeding initial data...")
    await conn.execute(SEED_DATA_SQL)
    logger.info("Data seeded successfully")


async def verify_schema(conn) -> bool:
    """Verify database schema is correct."""
    required_tables = [
        "matches", "entities", "predictions", "value_bets",
        "bets", "betting_sessions", "data_quality",
        "historical_matches", "backtest_results",
        "audit_log", "system_settings"
    ]

    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """

    rows = await conn.fetch(query)
    existing_tables = {row["table_name"] for row in rows}

    missing = set(required_tables) - existing_tables
    if missing:
        logger.error(f"Missing tables: {missing}")
        return False

    logger.info(f"Schema verification passed ({len(existing_tables)} tables)")
    return True


async def get_db_stats(conn) -> dict:
    """Get database statistics."""
    stats = {}

    tables = ["matches", "predictions", "value_bets", "bets", "historical_matches"]

    for table in tables:
        try:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            stats[table] = count
        except:
            stats[table] = 0

    return stats


async def main_async(args):
    """Main async function."""
    conn = await get_db_connection()

    if not conn:
        logger.error("Could not connect to database")
        logger.info("Make sure PostgreSQL is running and DATABASE_URL is set")
        logger.info(f"Current DATABASE_URL: {settings.database_url[:50]}...")
        return False

    try:
        if args.reset:
            confirm = input("Are you sure you want to reset the database? (yes/no): ")
            if confirm.lower() != "yes":
                logger.info("Reset cancelled")
                return False
            await reset_database(conn)

        await init_schema(conn)

        if args.seed:
            await seed_data(conn)

        # Verify
        success = await verify_schema(conn)

        if success:
            stats = await get_db_stats(conn)
            print("\n" + "="*50)
            print("Database Statistics:")
            print("="*50)
            for table, count in stats.items():
                print(f"  {table}: {count} rows")
            print("="*50 + "\n")

        return success

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS AI Database Initialization"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database (drops all tables)"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed initial data"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify schema, don't modify"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("NEXUS AI - Database Initialization")
    print("="*60 + "\n")

    success = asyncio.run(main_async(args))

    if success:
        print("Database initialization complete!")
    else:
        print("Database initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
