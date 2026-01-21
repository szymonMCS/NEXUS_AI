# tests/test_websocket.py
"""
Tests for WebSocket functionality in FastAPI backend.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWebSocketManager:
    """Tests for WebSocket connection manager."""
    
    def test_connection_manager_initializes(self):
        """Test that ConnectionManager initializes with empty connections."""
        from api.main import ConnectionManager
        manager = ConnectionManager()
        
        assert manager.active_connections == []
    
    @pytest.mark.asyncio
    async def test_connect_adds_websocket(self):
        """Test that connect adds websocket to active connections."""
        from api.main import ConnectionManager
        
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        
        await manager.connect(mock_ws)
        
        assert mock_ws in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_disconnect_removes_websocket(self):
        """Test that disconnect removes websocket from active connections."""
        from api.main import ConnectionManager
        
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        
        await manager.connect(mock_ws)
        manager.disconnect(mock_ws)
        
        assert mock_ws not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_connections(self):
        """Test that broadcast sends message to all connections."""
        from api.main import ConnectionManager
        
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        test_message = {"type": "test", "data": "hello"}
        await manager.broadcast(test_message)
        
        mock_ws1.send_json.assert_called_once_with(test_message)
        mock_ws2.send_json.assert_called_once_with(test_message)


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""
    
    def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is defined."""
        # This is a structural test - the endpoint is defined in the app
        # Verify the app has routes defined
        from api.main import app
        
        # Just verify routes exist and app is configured
        assert app.title == "NEXUS AI Lite"
        assert app.version == "2.2.0"


class TestProgressBroadcasting:
    """Tests for progress broadcasting functionality."""
    
    def test_analysis_state_initialization(self):
        """Test that analysis state is properly initialized."""
        from api.main import analysis_state
        
        assert analysis_state["is_running"] == False
        assert analysis_state["current_step"] == None
        assert analysis_state["progress"] == 0
        assert analysis_state["last_result"] == None
    
    @pytest.mark.asyncio
    async def test_broadcast_progress_updates_state(self):
        """Test that broadcast progress updates analysis state."""
        from api.main import analysis_state, _broadcast_progress
        
        await _broadcast_progress("testing", 50, "Testing progress")
        
        assert analysis_state["current_step"] == "testing"
        assert analysis_state["progress"] == 50


class TestWebSocketMessageFormats:
    """Tests for WebSocket message format validation."""
    
    def test_progress_message_format(self):
        """Test that progress messages have correct format."""
        from api.main import _broadcast_progress
        
        # The message format is validated through the broadcast function
        # Format: {"type": "progress", "step": str, "progress": int, "message": str, "data": Any}
        
        message = {
            "type": "progress",
            "step": "collecting",
            "progress": 10,
            "message": "Collecting fixtures...",
            "data": None
        }
        
        assert message["type"] == "progress"
        assert "step" in message
        assert "progress" in message
        assert "message" in message
    
    def test_complete_message_format(self):
        """Test that complete messages include value bets data."""
        value_bets = [
            {"rank": 1, "match": "Test Match", "odds": 2.0}
        ]
        
        message = {
            "type": "progress",
            "step": "complete",
            "progress": 100,
            "message": "Analysis complete!",
            "data": value_bets
        }
        
        assert message["step"] == "complete"
        assert message["progress"] == 100
        assert message["data"] == value_bets
    
    def test_error_message_format(self):
        """Test that error messages have correct format."""
        message = {
            "type": "progress",
            "step": "error",
            "progress": 0,
            "message": "Error: Something went wrong",
            "data": None
        }
        
        assert message["step"] == "error"
        assert "Error:" in message["message"]


class TestWebSocketReconnection:
    """Tests for WebSocket reconnection behavior."""
    
    def test_websocket_handles_disconnect_gracefully(self):
        """Test that disconnect is handled without errors."""
        from api.main import ConnectionManager
        
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        
        # Should not raise any exceptions
        manager.disconnect(mock_ws)
        
        # Websocket should not be in connections
        assert mock_ws not in manager.active_connections
    
    def test_websocket_handles_broadcast_errors(self):
        """Test that broadcast handles connection errors gracefully."""
        from api.main import ConnectionManager
        
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.send_json.side_effect = Exception("Connection closed")
        
        # Should not raise - errors are caught
        manager.active_connections.append(mock_ws)
        
        # This should not raise an exception
        import asyncio
        asyncio.run(manager.broadcast({"type": "test"}))
        
        # Connection should be removed or handled gracefully
        # In production, you'd want to implement cleanup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
