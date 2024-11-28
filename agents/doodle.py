from typing import Any, Dict, Optional
from mcp import ClientSession
from .base import BaseAgent


class DoodleAgent(BaseAgent):
    """A semantically-aware test agent that exhibits dog-like behaviors."""

    ROUTE_DESCRIPTION = (
        "Woof! Barking labradoodle dog with a wagging tail. Is a good boy."
    )

    def __init__(self, session: Optional[ClientSession] = None) -> None:
        """Initialize the DoodleAgent."""
        super().__init__(session)
        self._bark_count = 0

    async def get_description(self) -> str:
        self._logger.debug(f"Returning route description: {self.ROUTE_DESCRIPTION}")
        return self.ROUTE_DESCRIPTION

    async def bark(self, content: Dict[str, Any]) -> str:
        self._bark_count += 1
        self._logger.info(f"Bark request: {content}")

        # Simple response variation based on bark count
        if self._bark_count % 3 == 0:
            response = "Ruff ruff! *tail wagging*"
        elif self._bark_count % 2 == 0:
            response = "Arf! *excited*"
        else:
            response = "Woof!"

        self._logger.info(f"Bark response sent: {response}")
        return response
