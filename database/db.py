# database/db.py
"""
Database connection and session management.
Supports both SQLite (development) and PostgreSQL (production).
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from config.settings import settings
from database.models import Base

logger = logging.getLogger(__name__)


# === ENGINE CONFIGURATION ===

def get_engine_config():
    """Get SQLAlchemy engine configuration based on database URL"""
    config = {
        "echo": settings.DEBUG,
        "future": True,
    }

    # SQLite specific configuration
    if settings.DATABASE_URL.startswith("sqlite"):
        config.update({
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        })
    # PostgreSQL specific configuration
    elif settings.DATABASE_URL.startswith("postgresql"):
        config.update({
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,  # Verify connections before using
            "pool_recycle": 3600,  # Recycle connections after 1 hour
        })

    return config


# Create engine
engine = create_engine(settings.DATABASE_URL, **get_engine_config())


# === SQLite OPTIMIZATIONS ===

if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """Enable SQLite optimizations"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()


# === SESSION FACTORY ===

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


# === DATABASE UTILITIES ===

def init_db():
    """
    Initialize database - create all tables.
    Call this on application startup.
    """
    logger.info(f"Initializing database: {settings.DATABASE_URL}")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_db():
    """
    Drop all database tables.
    WARNING: This deletes all data!
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped")


def reset_db():
    """
    Reset database - drop and recreate all tables.
    WARNING: This deletes all data!
    """
    logger.warning("Resetting database...")
    drop_db()
    init_db()
    logger.info("Database reset complete")


# === SESSION MANAGEMENT ===

def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI/other frameworks.
    Yields a database session and ensures it's closed after use.

    Usage:
        def my_function(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as db:
            db.query(Match).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# === HEALTH CHECK ===

def check_db_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with get_db_session() as db:
            db.execute("SELECT 1")
        logger.info("Database connection: OK")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# === UTILITY FUNCTIONS ===

def get_table_counts() -> dict:
    """
    Get row counts for all tables.

    Returns:
        dict: Table names and their row counts
    """
    from database.models import (
        Match, Odds, Prediction, Bet, News, MatchStats,
        BettingSession, SystemMetrics
    )

    counts = {}
    with get_db_session() as db:
        counts["matches"] = db.query(Match).count()
        counts["odds"] = db.query(Odds).count()
        counts["predictions"] = db.query(Prediction).count()
        counts["bets"] = db.query(Bet).count()
        counts["news"] = db.query(News).count()
        counts["match_stats"] = db.query(MatchStats).count()
        counts["betting_sessions"] = db.query(BettingSession).count()
        counts["system_metrics"] = db.query(SystemMetrics).count()

    return counts


def vacuum_db():
    """
    Vacuum database to reclaim space (SQLite only).
    For PostgreSQL, use VACUUM command directly.
    """
    if settings.DATABASE_URL.startswith("sqlite"):
        logger.info("Running VACUUM on SQLite database...")
        with engine.connect() as conn:
            conn.execute("VACUUM")
        logger.info("VACUUM complete")
    else:
        logger.warning("VACUUM is only supported for SQLite. For PostgreSQL, run 'VACUUM;' directly.")


# === MIGRATION HELPERS ===

def backup_db(backup_path: str = "nexus_backup.db"):
    """
    Backup SQLite database.

    Args:
        backup_path: Path where to save the backup
    """
    if not settings.DATABASE_URL.startswith("sqlite"):
        logger.error("Backup is only implemented for SQLite")
        return

    import shutil
    import os

    # Extract path from DATABASE_URL (sqlite:///./nexus.db -> ./nexus.db)
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")

    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
    else:
        logger.error(f"Database file not found: {db_path}")


# === EXPORTS ===

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "init_db",
    "drop_db",
    "reset_db",
    "check_db_connection",
    "get_table_counts",
    "vacuum_db",
    "backup_db",
]
