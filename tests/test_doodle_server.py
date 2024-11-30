import sys
import pytest
from mcp.server.doodle_server import DoodleServer

@pytest.mark.asyncio
async def test_doodle_server():
    server = DoodleServer()
    assert isinstance(server, DoodleServer)
    
    # Test bark functionality
    result = await server.bark()
    assert "sound" in result
    assert "timestamp" in result
    
    # Test different intensities
    quiet_bark = await server.bark(intensity="quiet")
    assert quiet_bark["sound"] == "woof..."
    
    loud_bark = await server.bark(intensity="loud")
    assert loud_bark["sound"] == "WOOF!!!"
    
    # Test last bark tracking
    last_bark = await server.get_last_bark()
    assert "last_bark" in last_bark 