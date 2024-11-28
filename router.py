"""
Semantic Router based on MCP Protocol.

This module implements a semantic routing system that directs requests to appropriate 
handlers based on semantic understanding of the content. It leverages the MCP protocol
for standardized communication between components.

The router uses semantic embeddings to match incoming requests with the most appropriate
handler based on their semantic descriptions. This allows for flexible and intelligent
routing that goes beyond simple pattern matching.

Key Components:
    - Route: Dataclass representing a semantic routing rule
    - SemanticRouter: Main router class handling semantic matching and request routing

Semantic Matching Process:
    1. Each route is registered with a semantic description of its capabilities
    2. Descriptions are converted to embeddings using an embedding service
    3. Incoming requests are matched against route descriptions using cosine similarity
    4. Requests are routed to the handler with highest semantic similarity above threshold

Example:
    ```python
    router = SemanticRouter()
    
    # Register a route for image generation
    await router.add_route(Route(
        pattern="generate",
        handler="image_gen_handler",
        route_description="Creates and generates images from text descriptions. 
                         Handles drawing, illustration, and visual content creation."
    ))
    
    # Route will match requests semantically related to image generation
    await router.handle_request("Can you draw me a picture of a sunset?")
    ```

Dependencies:
    - MCP Protocol: For standardized communication between components
    - Embedding Service: Must be registered with MCP for semantic embedding generation
"""

from typing import Dict, List, Optional, Any, Tuple, Protocol
from dataclasses import dataclass, field
import asyncio
import sys
from sentence_transformers import SentenceTransformer
from agents.doodle import DoodleAgent
import logging

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """Represents a semantic routing rule with agent description.

    This class defines how requests should be routed to specific handlers based on
    semantic understanding of their capabilities. Each route maintains its own
    semantic embedding for efficient matching.

    Attributes:
        pattern: Semantic pattern to match (used for legacy/fallback matching)
        handler: Identifier for the handler that will process matched requests
        route_description: Detailed description of handler's capabilities for semantic matching
        priority: Priority level for this route (higher numbers matched first)
        metadata: Additional route-specific configuration data
        embedding: Cached semantic embedding vector for efficient matching
    """

    pattern: str  # Semantic pattern to match
    handler: str  # Handler identifier
    route_description: str  # Semantic description of agent capabilities
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Cache for semantic embedding


class SessionProtocol(Protocol):
    """Protocol defining the interface for MCP sessions."""

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any: ...


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """Initialize the embedding service with a pre-trained model.

        Args:
            model_name: Name of the pre-trained model to use
        """
        logger.info("\nInitializing embedding service...")

        self.model = SentenceTransformer(model_name)

        logger.info(
            f"✓ Embedding model {model_name} with dimension {self.model.get_sentence_embedding_dimension()} loaded successfully!"
        )

    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        # BGE models perform better with a query prefix for asymmetric tasks
        if not text.startswith("Represent this sentence: "):
            text = f"Represent this sentence: {text}"
        return self.model.encode(text, normalize_embeddings=True).tolist()


class SemanticRouter:
    """Routes requests based on semantic matching of content.

    This router implements intelligent request routing using semantic similarity
    between the request content and registered route descriptions. It maintains
    an embedding cache to optimize performance for repeated requests.

    The routing process uses cosine similarity between normalized embedding vectors
    to find the best matching handler for each request. A configurable similarity
    threshold (default 0.7) ensures that requests are only routed when there is
    sufficient semantic relevance.

    Key Features:
        - Semantic matching using embedding vectors
        - Embedding cache for performance optimization
        - Priority-based route ordering
        - Configurable similarity threshold
        - Async interface for non-blocking operations

    Dependencies:
        - Requires an active MCP session with registered embedding service
        - Embedding service must return normalized vectors for accurate matching
    """

    def __init__(
        self, embedding_service: EmbeddingService, similarity_threshold: float = 0.5
    ) -> None:
        """Initialize the semantic router with an embedding service.

        Args:
            embedding_service: Service for generating text embeddings
            similarity_threshold: Minimum similarity score required for matching (0-1)
        """
        self.routes: List[Route] = []
        self._session: Optional[SessionProtocol] = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    async def add_route(self, route: Route) -> None:
        """Add a new routing rule.

        Args:
            route: The Route object containing pattern, handler and description
        """
        # Generate and cache embedding for route description
        route.embedding = await self._get_embedding(route.route_description)
        self.routes.append(route)
        self.routes.sort(key=lambda x: x.priority, reverse=True)

    async def route(self, content: str) -> Optional[Tuple[str, float]]:
        """Route content to appropriate handler based on semantic matching."""
        content_embedding = await self._get_embedding(content)

        best_match = None
        highest_similarity = 0.0

        for route in self.routes:
            if route.embedding:
                similarity = self._compute_similarity(
                    content_embedding, route.embedding
                )
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = route.handler

        if best_match and highest_similarity > self.similarity_threshold:
            return best_match, highest_similarity
        return None

    async def _get_embedding(self, text: str) -> List[float]:
        """Get semantic embedding for text using the embedding service."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Use the embedding service to generate the embedding
        result = self.embedding_service.get_embedding(text)
        self._embedding_cache[text] = result
        return result

    def _compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between embeddings.

        Calculates semantic similarity between two embedding vectors using
        dot product. This works because vectors are expected to be normalized,
        making dot product equivalent to cosine similarity.

        Args:
            embedding1: First normalized embedding vector
            embedding2: Second normalized embedding vector

        Returns:
            float: Similarity score between 0 (unrelated) and 1 (identical)

        Note:
            Assumes input vectors are normalized to unit length. If vectors
            are not normalized, results will not represent true cosine similarity.
        """
        # Simple dot product for normalized embeddings
        return sum(a * b for a, b in zip(embedding1, embedding2))

    async def handle_request(self, content: str) -> Any:
        """Process incoming request and route to appropriate handler.

        Args:
            content: The request content to be processed

        Returns:
            Response from the matched handler

        Raises:
            ValueError: If no matching route is found
        """
        route_result = await self.route(content)
        if route_result:
            handler, similarity = route_result
            return await self._execute_handler(handler, content)
        raise ValueError("No matching route found")

    async def _execute_handler(self, handler: Any, content: str) -> Any:
        """Execute the matched handler.

        Args:
            handler: The actual handler function to execute
            content: Original request content

        Returns:
            Handler execution result
        """
        # Direct execution without MCP session check
        return await handler(content)

    async def set_session(self, session: SessionProtocol) -> None:
        """Set the MCP client session.

        Args:
            session: Active session implementing SessionProtocol
        """
        self._session = session


async def main():
    # Create router with embedding service
    embedding_service = EmbeddingService()
    router = SemanticRouter(embedding_service)

    # Create agent and get its description
    doodle = DoodleAgent()
    route_description = await doodle.get_description()

    # Add route that uses the agent's bark method
    await router.add_route(
        Route(
            pattern="bark",
            handler=doodle.bark,  # Use the instance method instead of the class
            route_description=route_description,
        )
    )

    # Test the router
    try:
        result = await router.handle_request("Who's a good boy?")
        print(f"Result: {result}")  # Added to see the output
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
