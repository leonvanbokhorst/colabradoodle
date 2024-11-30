import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import numpy as np
from typing import Dict, Any

from router import (
    SemanticRouter,
    EmbeddingService,
    Route,
    RouteError,
    ExecutionError,
    SessionProtocol,
)


# Test fixtures
@pytest.fixture
def embedding_service():
    service = Mock(spec=EmbeddingService)
    # Mock the get_embedding method to return normalized vectors
    service.get_embedding.return_value = [0.5, 0.5, 0.5]
    return service


@pytest.fixture
async def router(embedding_service):
    return SemanticRouter(
        embedding_service=embedding_service,
        similarity_threshold=0.5,
        cleanup_interval=1.0,
    )


@pytest.fixture
async def router_with_defaults(embedding_service):
    return SemanticRouter(
        embedding_service=embedding_service,
        similarity_threshold=0.5,
        cleanup_interval=1.0,
    )


# Basic initialization tests
@pytest.mark.asyncio
async def test_router_initialization(router_with_defaults):
    router = await router_with_defaults
    assert router.similarity_threshold == 0.5
    assert router.cleanup_interval == 1.0
    assert len(router.routes) == 0


@pytest.mark.asyncio
async def test_add_route(router_with_defaults):
    router = await router_with_defaults
    handler = AsyncMock()
    route = Route(
        pattern="test", handler=handler, route_description="Test route", priority=1
    )

    await router.add_route(route)
    assert len(router.routes) == 1
    assert router.routes[0].pattern == "test"


@pytest.mark.asyncio
async def test_duplicate_route(router_with_defaults):
    router = await router_with_defaults
    handler = AsyncMock()
    route = Route(pattern="test", handler=handler, route_description="Test route")

    await router.add_route(route)
    with pytest.raises(ValueError):
        await router.add_route(route)


@pytest.mark.asyncio
async def test_route_matching(router_with_defaults):
    router = await router_with_defaults
    handler = AsyncMock(return_value="Success")
    route = Route(pattern="test", handler=handler, route_description="Test route")
    await router.add_route(route)

    result = await router.handle_request("Test message")
    assert result == "Success"
    handler.assert_called_once_with("Test message")


@pytest.mark.asyncio
async def test_no_matching_route(router_with_defaults):
    router = await router_with_defaults
    with pytest.raises(RouteError):
        await router.handle_request("Unmatched message")


@pytest.mark.asyncio
async def test_default_handler(embedding_service):
    default_handler = AsyncMock(return_value="Default response")
    router = SemanticRouter(
        embedding_service=embedding_service,
        similarity_threshold=0.5,
        default_handler=default_handler,
    )

    result = await router.handle_request("Unmatched message")
    assert result == "Default response"
    default_handler.assert_called_once()


@pytest.mark.asyncio
async def test_route_priority(router_with_defaults):
    router = await router_with_defaults
    handler1 = AsyncMock(return_value="High priority")
    handler2 = AsyncMock(return_value="Low priority")

    route1 = Route(
        pattern="test_high",
        handler=handler1,
        route_description="High priority route",
        priority=2,
    )
    route2 = Route(
        pattern="test_low",
        handler=handler2,
        route_description="Low priority route",
        priority=1,
    )

    await router.add_route(route2)
    await router.add_route(route1)

    assert router.routes[0].priority == 2
    assert router.routes[1].priority == 1


@pytest.mark.asyncio
async def test_timeout_handling(embedding_service):
    async def slow_handler(content: str):
        await asyncio.sleep(2)
        return "Done"

    router = SemanticRouter(
        embedding_service=embedding_service,
        request_timeout=0.1,
        similarity_threshold=0.0,
    )

    embedding_service.get_embedding.return_value = [1.0, 0.0, 0.0]

    route = Route(pattern="test", handler=slow_handler, route_description="Slow route")
    await router.add_route(route)

    try:
        await router.handle_request("Test message")
        pytest.fail("Should have raised ExecutionError")
    except ExecutionError as e:
        assert "timed out" in str(e).lower()


@pytest.mark.asyncio
async def test_session_protocol():
    class MockSession(SessionProtocol):
        async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
            return f"Called {tool_name}"

    session = MockSession()
    router = SemanticRouter(
        embedding_service=Mock(spec=EmbeddingService), similarity_threshold=0.5
    )

    await router.set_session(session)
    assert router._session is not None


# Error handling tests
@pytest.mark.asyncio
async def test_empty_request(router_with_defaults):
    router = await router_with_defaults
    with pytest.raises(ValueError):
        await router.handle_request("")


@pytest.mark.asyncio
async def test_handler_error(router_with_defaults):
    router = await router_with_defaults

    async def failing_handler(content: str):
        raise ValueError("Handler error")

    route = Route(
        pattern="test", handler=failing_handler, route_description="Failing route"
    )
    await router.add_route(route)

    with pytest.raises(ValueError):
        await router.handle_request("Test message")
