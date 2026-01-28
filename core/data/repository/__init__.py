"""
NEXUS ML Data Repository.

Checkpoint: 0.10
Exports unified data repository and interfaces.
"""

from core.data.repository.interface import (
    IDataRepository,
    IMatchDataProvider,
    IOddsProvider,
)
from core.data.repository.unified import (
    UnifiedDataRepository,
    get_repository,
    reset_repository,
)

__all__ = [
    # Interfaces
    "IDataRepository",
    "IMatchDataProvider",
    "IOddsProvider",
    # Implementation
    "UnifiedDataRepository",
    # Helpers
    "get_repository",
    "reset_repository",
]
