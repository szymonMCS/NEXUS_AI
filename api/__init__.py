# api/__init__.py
"""
FastAPI backend for NEXUS AI.
Replaces Gradio app.py with REST API for React frontend.
"""

from api.main import app
from api.routers import (
    ensemble_router,
    monitoring_router,
    admin_router,
    register_routers,
)

__all__ = [
    "app",
    "ensemble_router",
    "monitoring_router",
    "admin_router",
    "register_routers",
]
