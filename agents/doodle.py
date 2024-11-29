from typing import Optional
from mcp import ClientSession
from .base import BaseAgent
from utils.log_config import setup_logger

logger = setup_logger("DoodleAgent")


class DoodleAgent(BaseAgent):
    """A semantically-aware dog-like agent that barks."""

    ROUTE_DESCRIPTION = """
    A friendly dog agent that responds to questions and commands about dogs, barking, 
    and general dog behavior. Handles queries like 'who let the dogs out', 'can you bark',
    'what does the dog say', etc.
    """

    def __init__(self, session: Optional[ClientSession] = None) -> None:
        """Initialize the DoodleAgent."""
        super().__init__(session)

    async def get_description(self) -> str:
        logger.debug(f"Returning route description: {self.ROUTE_DESCRIPTION}")
        return self.ROUTE_DESCRIPTION

    async def bark(self, _content: str) -> str:
        """Handle bark-related requests."""
        return "Woof!"
