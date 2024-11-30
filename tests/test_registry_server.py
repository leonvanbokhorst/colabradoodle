import pytest
from datetime import datetime, timedelta
import asyncio
from mcp_registry.server import RegistryServer

@pytest.mark.asyncio
async def test_registry_server_custom_timeout():
    # Initialize server with 2 second timeout
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)
    
    # Register a test server
    result = await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server for timeout",
        capabilities=["test"],
        endpoint="test://endpoint"
    )
    assert result["status"] == "registered"
    
    # Verify server is discoverable
    servers = await server.discover_servers()
    assert len(servers) == 1
    assert servers[0]["id"] == "test_server"
    
    # Wait for timeout period
    await asyncio.sleep(3)  # Wait longer than timeout
    
    # Start cleanup
    cleanup_task = asyncio.create_task(server._cleanup_loop())
    await asyncio.sleep(0.1)  # Allow cleanup to run
    
    # Verify server was removed
    servers = await server.discover_servers()
    assert len(servers) == 0
    
    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_registry_server_heartbeat_prevents_timeout():
    # Initialize server with 2 second timeout
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)
    
    # Register a test server
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server for timeout",
        capabilities=["test"],
        endpoint="test://endpoint"
    )
    
    # Store initial heartbeat timestamp
    initial_heartbeat = server.servers["test_server"].last_heartbeat
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(server._cleanup_loop())
    
    # Wait just before timeout and send heartbeat
    await asyncio.sleep(1.9)  # Just before 2s timeout
    await server.heartbeat("test_server")
    
    # Verify heartbeat timestamp was updated
    assert server.servers["test_server"].last_heartbeat > initial_heartbeat
    
    # Wait another period to ensure server stays registered
    await asyncio.sleep(1.5)
    servers = await server.discover_servers()
    assert len(servers) == 1
    assert servers[0]["id"] == "test_server"
    
    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_registry_server_default_timeout():
    # Initialize server with default timeout
    server = RegistryServer(name="test_registry")
    
    # Verify default timeout is 60 seconds
    assert server.server_timeout_seconds == 60
    
    # Register a test server
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server for timeout",
        capabilities=["test"],
        endpoint="test://endpoint"
    )
    
    # Verify server is registered
    servers = await server.discover_servers()
    assert len(servers) == 1 