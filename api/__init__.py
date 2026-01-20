# api/__init__.py
"""
FastAPI backend for NEXUS AI.
Replaces Gradio app.py with REST API for React frontend.
"""

from api.main import app

__all__ = ["app"]
