from typing import Optional
from mcp import ClientSession
from .base import BaseAgent
from utils.log_config import setup_logger

logger = setup_logger("DoodleAgent")


class DoodleAgent(BaseAgent):
    """A semantically-aware dog-like agent that barks."""

    ROUTE_DESCRIPTION = "Woof! Labradoodle dog that barks and wags his tail. He's a good boy. Atta girl!"

    def __init__(self, session: Optional[ClientSession] = None) -> None:
        """Initialize the DoodleAgent."""
        super().__init__(session)

    async def get_description(self) -> str:
        logger.debug(f"Returning route description: {self.ROUTE_DESCRIPTION}")
        return self.ROUTE_DESCRIPTION

    async def bark(self, _content: str) -> str:
        """Handle bark-related requests."""
        return "Woof!"
