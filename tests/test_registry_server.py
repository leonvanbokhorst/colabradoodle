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
        endpoint="test://endpoint",
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
        endpoint="test://endpoint",
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
        endpoint="test://endpoint",
    )

    # Verify server is registered
    servers = await server.discover_servers()
    assert len(servers) == 1

@pytest.mark.asyncio
async def test_cleanup_stale_servers_with_concurrent_heartbeats():
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)
    
    # Register servers
    servers_config = [
        ("server1", "Server 1"),
        ("server2", "Server 2"),
        ("server3", "Server 3")
    ]
    
    for server_id, name in servers_config:
        await server.register_server(
            server_id=server_id,
            name=name,
            description=f"Test server {name}",
            capabilities=["test"],
            endpoint="test://endpoint"
        )
    
    # Start cleanup task before sleep
    cleanup_task = asyncio.create_task(server._cleanup_loop())
    
    # Use shorter sleep to prevent cleanup from occurring too soon
    await asyncio.sleep(0.5)
    
    try:
        await asyncio.gather(
            server.heartbeat("server1"),
            server.heartbeat("server2")
        )
    finally:
        cleanup_task.cancel()


@pytest.mark.asyncio
async def test_heartbeat_updates_timestamp():
    """Test that heartbeat calls update the last_heartbeat timestamp."""
    server = RegistryServer(name="test_registry", server_timeout_seconds=5)

    # Register a test server
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server for timestamp updates",
        capabilities=["test"],
        endpoint="test://endpoint",
    )

    # Get initial timestamp
    initial_timestamp = server.servers["test_server"].last_heartbeat

    # Wait briefly and send heartbeat
    await asyncio.sleep(0.1)
    await server.heartbeat("test_server")

    # Verify timestamp was updated
    updated_timestamp = server.servers["test_server"].last_heartbeat
    assert updated_timestamp > initial_timestamp


@pytest.mark.asyncio
async def test_edge_case_heartbeat_before_timeout():
    """Test server remains registered when heartbeat is sent just before timeout."""
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)

    # Register a test server
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server for edge case",
        capabilities=["test"],
        endpoint="test://endpoint",
    )

    # Start cleanup task
    cleanup_task = asyncio.create_task(server._cleanup_loop())

    # Wait until just before timeout and send heartbeat
    await asyncio.sleep(1.9)  # Wait for 1.9 seconds of 2 second timeout
    await server.heartbeat("test_server")

    # Wait briefly to allow cleanup to run
    await asyncio.sleep(0.2)

    # Verify server is still registered
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
async def test_multiple_heartbeats_maintain_registration():
    """Test that multiple heartbeats maintain server registration."""
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)

    # Register a test server
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server for multiple heartbeats",
        capabilities=["test"],
        endpoint="test://endpoint",
    )

    # Start cleanup task
    cleanup_task = asyncio.create_task(server._cleanup_loop())

    # Send multiple heartbeats over time
    timestamps = []
    for _ in range(3):
        await asyncio.sleep(0.5)
        await server.heartbeat("test_server")
        timestamps.append(server.servers["test_server"].last_heartbeat)

    # Verify timestamps are monotonically increasing
    assert all(t1 < t2 for t1, t2 in zip(timestamps, timestamps[1:]))

    # Verify server remains registered
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
async def test_edge_case_heartbeat_timeouts():
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)
    
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server",
        capabilities=["test"],
        endpoint="test://endpoint",
    )
    
    # Manually set the last_heartbeat to ensure timeout
    server.servers["test_server"].last_heartbeat = datetime.utcnow() - timedelta(seconds=3)
    
    cleanup_task = asyncio.create_task(server._cleanup_loop())
    
    try:
        # Wait for cleanup to run
        await asyncio.sleep(0.5)  # Shorter wait time since we pre-dated the heartbeat
        
        servers = await server.discover_servers()
        assert len(servers) == 0
    finally:
        cleanup_task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, 0, -0.5])
async def test_registry_server_invalid_timeout(timeout):
    """Test that RegistryServer raises ValueError for invalid timeout values."""
    with pytest.raises(ValueError, match="Server timeout must be positive"):
        RegistryServer(name="test_registry", server_timeout_seconds=timeout)

