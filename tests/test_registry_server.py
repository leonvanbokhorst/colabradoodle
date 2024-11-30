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
    # Initialize server with 2 second timeout
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)
    
    # Register multiple test servers
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
    
    # Manipulate last_heartbeat timestamps
    now = datetime.utcnow()
    server.servers["server1"].last_heartbeat = now - timedelta(seconds=3)  # Definitely stale
    server.servers["server2"].last_heartbeat = now - timedelta(seconds=1.5)  # Still fresh
    server.servers["server3"].last_heartbeat = now - timedelta(seconds=1.5)  # Still fresh
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(server._cleanup_loop())
    
    # Send heartbeats concurrently
    await asyncio.sleep(1)  # Wait a bit before sending heartbeats
    await asyncio.gather(
        server.heartbeat("server1"),
        server.heartbeat("server2")
    )
    
    # Allow cleanup to run
    await asyncio.sleep(1)
    
    # Verify results
    remaining_servers = await server.discover_servers()
    remaining_ids = {s["id"] for s in remaining_servers}
    
    assert "server1" in remaining_ids, "Server 1 should remain due to heartbeat"
    assert "server2" in remaining_ids, "Server 2 should remain due to heartbeat"
    assert "server3" in remaining_ids, "Server 3 should remain"
    assert len(remaining_servers) == 3, "Should have exactly 3 servers remaining"
    

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
    """Test server registration behavior at exact timeout and just after timeout."""
    server = RegistryServer(name="test_registry", server_timeout_seconds=2)

    # Test exact timeout
    await server.register_server(
        server_id="test_server",
        name="Test Server",
        description="Test server",
        capabilities=["test"],
        endpoint="test://endpoint",
    )

    # Verify initial registration
    assert await server.is_registered("test_server")

    # Start cleanup task
    cleanup_task = asyncio.create_task(server._cleanup_loop())

    # Wait exactly until timeout plus a small delay for cleanup
    await asyncio.sleep(2)  # Wait for timeout
    await asyncio.sleep(0.1)  # Allow cleanup to run
    servers = await server.discover_servers()
    assert len(servers) == 0
    assert not await server.is_registered("test_server")

    # Test just after timeout
    await server.register_server(
        server_id="test_server2",
        name="Test Server",
        description="Test server",
        capabilities=["test"],
        endpoint="test://endpoint",
    )

    # Verify second registration
    assert await server.is_registered("test_server2")

    # Wait just past timeout plus cleanup delay
    await asyncio.sleep(2.1)  # Wait past timeout
    await asyncio.sleep(0.1)  # Allow cleanup to run
    servers = await server.discover_servers()
    assert len(servers) == 0
    assert not await server.is_registered("test_server2")

    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

