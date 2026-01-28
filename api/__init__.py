# api/__init__.py
"""
FastAPI backend for NEXUS AI.
Replaces Gradio app.py with REST API for React frontend.

Checkpoint: 4.5 - Added ML router
"""

from api.main import app
from api.routers import (
    ml_router,
    ensemble_router,
    monitoring_router,
    admin_router,
    register_routers,
)

__all__ = [
    "app",
    "ml_router",
    "ensemble_router",
    "monitoring_router",
    "admin_router",
    "register_routers",
]
