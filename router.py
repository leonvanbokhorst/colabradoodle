from typing import (
    Dict,
    List,
    Optional,
    Any,
    Tuple,
    Protocol,
    TypeAlias,
    Final,
    Callable,
    Awaitable,
)
from dataclasses import dataclass, field
import asyncio
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from agents.doodle import DoodleAgent
from utils.log_config import setup_logger
from collections import OrderedDict
import time

# Type aliases
EmbeddingVector: TypeAlias = List[float]
HandlerFunction: TypeAlias = Callable[[str], Awaitable[Any]]

# Constants
EMBEDDING_MODEL: Final[str] = "BAAI/bge-large-en-v1.5"
SIMILARITY_THRESHOLD: Final[float] = 0.625
DEFAULT_CACHE_SIZE: Final[int] = 1000
REQUEST_TIMEOUT: Final[float] = 30.0
MAX_TEXT_LENGTH: Final[int] = 1000

logger = setup_logger("Router")


@dataclass
class Route:
    """Represents a routing configuration for handling specific types of requests.

    Args:
        pattern: The pattern to match against
        handler: The handler function to process matching requests
        route_description: Description used for semantic matching
        priority: Priority level for route selection (higher = more priority)
        metadata: Additional route configuration
        embedding: Cached embedding vector
    """

    pattern: str
    handler: HandlerFunction
    route_description: str
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[EmbeddingVector] = None


class SessionProtocol(Protocol):
    """Protocol defining the interface for session management."""

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any: ...


class EmbeddingCache:
    """LRU cache for embedding vectors with size limit."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[EmbeddingVector]:
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: EmbeddingVector) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value


class EmbeddingService:
    """Service for generating and managing text embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info("Initializing embedding service...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded: {model_name} with dimension {self.dimension}"
        )

    def get_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding vector for input text."""
        if not text.startswith("Represent this sentence: "):
            text = f"Represent this sentence: {text}"
        return self.model.encode(text, normalize_embeddings=True).tolist()


class RouterError(Exception):
    pass


class RouteNotFoundError(RouterError):
    pass


class EmbeddingError(RouterError):
    pass


class HandlerExecutionError(RouterError):
    pass


class RateLimitExceededError(RouterError):
    pass


class RequestTimeoutError(RouterError):
    pass


class SemanticRouter:
    """Semantic router for handling and routing requests based on embeddings."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        default_handler: Optional[HandlerFunction] = None,
    ) -> None:
        """Initialize the semantic router.

        Args:
            embedding_service: Service for generating embeddings
            similarity_threshold: Minimum similarity score (0-1)
            default_handler: Handler for unmatched requests

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        if embedding_service is None:
            raise ValueError("Embedding service cannot be None")

        logger.info(
            f"Initializing SemanticRouter with threshold {similarity_threshold}"
        )
        self.routes: List[Route] = []
        self._session: Optional[SessionProtocol] = None
        self._embedding_cache = EmbeddingCache(DEFAULT_CACHE_SIZE)
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.default_handler = default_handler
        self._request_times: List[float] = []

    async def add_route(self, route: Route) -> None:
        """Add a new route to the router.

        Args:
            route: Route configuration to add

        Raises:
            ValueError: If route configuration is invalid
            EmbeddingError: If embedding generation fails
        """
        if not route.pattern or not route.handler or not route.route_description:
            raise ValueError("Route must have pattern, handler and description")

        if any(r.pattern == route.pattern for r in self.routes):
            raise ValueError(f"Route with pattern '{route.pattern}' already exists")

        try:
            route.embedding = await self._get_embedding(route.route_description)
        except Exception as e:
            logger.error(f"Route addition failed: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e

        self.routes.append(route)
        self.routes.sort(key=lambda x: x.priority, reverse=True)
        logger.debug(f"Total routes after addition: {len(self.routes)}")

    def _check_rate_limit(self) -> None:
        """Check if request rate limit is exceeded."""
        current_time = time.time()
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        if len(self._request_times) > 100:  # 100 requests per minute
            raise RateLimitExceededError("Rate limit exceeded")
        self._request_times.append(current_time)

    def _compute_similarity(
        self, embedding1: EmbeddingVector, embedding2: EmbeddingVector
    ) -> float:
        """Compute cosine similarity between two embedding vectors."""
        return float(np.dot(embedding1, embedding2))

    async def _get_embedding(self, text: str) -> EmbeddingVector:
        """Generate or retrieve cached embedding for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If text is empty
            EmbeddingError: If embedding generation fails
        """
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            logger.warning(f"Text truncated to {MAX_TEXT_LENGTH} characters")

        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        try:
            result = self.embedding_service.get_embedding(text)
            if not result:
                raise EmbeddingError("Embedding service returned empty result")
            self._embedding_cache.put(text, result)
            return result
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e

    async def handle_request(self, content: str) -> Any:
        """Handle an incoming request.

        Args:
            content: Request content

        Returns:
            Handler response

        Raises:
            ValueError: If content is empty
            RouteNotFoundError: If no matching route is found
            RequestTimeoutError: If request times out
            Exception: Original exception from handler execution
        """
        if not content.strip():
            raise ValueError("Request content cannot be empty")

        self._check_rate_limit()

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                logger.info("Processing new request")
                route_result = await self.route(content)

                if not route_result:
                    if self.default_handler:
                        logger.info("Using default handler for unmatched request")
                        return await self._execute_handler(self.default_handler, content)
                    logger.error("No matching route found for request")
                    raise RouteNotFoundError("No matching route found")

                handler, similarity = route_result
                logger.info(f"Executing handler '{handler.__name__}' with similarity {similarity:.3f}")
                return await self._execute_handler(handler, content)

        except asyncio.TimeoutError:
            logger.error("Request timed out")
            raise RequestTimeoutError(f"Request timed out after {REQUEST_TIMEOUT} seconds")

    async def route(self, content: str) -> Optional[Tuple[HandlerFunction, float]]:
        logger.debug(f"Routing request: {content[:100]}...")
        content_embedding = await self._get_embedding(content)

        best_match = None
        highest_similarity = 0.0

        for route in self.routes:
            if route.embedding:
                similarity = self._compute_similarity(
                    content_embedding, route.embedding
                )
                logger.debug(f"Route '{route.handler}' similarity: {similarity:.3f}")
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = route.handler

        if best_match and highest_similarity > self.similarity_threshold:
            logger.info(
                f"Found matching route: {best_match.__name__} (similarity: {highest_similarity:.3f})"
            )
            return best_match, highest_similarity

        logger.warning(
            f"No matching route found above threshold {self.similarity_threshold}"
        )
        return None

    async def _execute_handler(self, handler: HandlerFunction, content: str) -> Any:
        """Execute the handler function while preserving original exceptions.

        Args:
            handler: Function to handle the request
            content: Request content to pass to handler

        Returns:
            Handler response

        Raises:
            Exception: Original exception from handler execution
        """
        try:
            return await handler(content)
        except Exception as e:
            logger.error(f"Handler '{handler.__name__}' execution failed: {e}", exc_info=True)
            raise  # Re-raise the original exception with full context

    async def set_session(self, session: SessionProtocol) -> None:
        self._session = session


async def default_response(content: str) -> str:
    return f"I'm not sure how to handle that request. {content[:100]}..."


async def main():
    try:
        embedding_service = EmbeddingService()
        router = SemanticRouter(
            embedding_service,
            similarity_threshold=0.625,
            default_handler=default_response,
        )

        doodle = DoodleAgent()
        route_description = await doodle.get_description()

        await router.add_route(
            Route(
                pattern="bark",
                handler=doodle.bark,
                route_description=route_description,
            )
        )

        result = await router.handle_request("Who let the dogs out?")
        print(f"Result: {result}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
