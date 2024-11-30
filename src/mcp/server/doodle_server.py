from typing import Dict, Optional
from datetime import datetime
from datetime import timezone
from mcp.server import Server
from utils.log_config import setup_logger

logger = setup_logger("DoodleServer")


class DoodleServer(Server):
    """MCP Server that provides dog-related functionality"""

    ROUTE_DESCRIPTION = """
    A playful dog server that responds to dog-related queries and commands.
    Handles requests about dogs, barking, and common dog-related phrases like
    'Who let the dogs out?', 'Can you bark?', or 'What does the dog say?'
    The server can bark with different intensities and keep track of bark history.
    """

    def __init__(self, name: str = "doodle"):
        super().__init__(name=name)
        self.last_bark: Optional[datetime] = None
        self.tools = {
            "bark": self.bark,
            "get_last_bark": self.get_last_bark,
            "get_description": self.get_description,
        }
        logger.info("DoodleServer initialized")

    async def bark(self, intensity: str = "normal") -> Dict[str, str]:
        """
        Simulate a dog bark with different intensities.

        Args:
            intensity: Bark intensity ("quiet", "normal", or "loud")

        Returns:
            Dict containing the bark sound and timestamp
        """
        bark_sounds = {"quiet": "woof...", "normal": "Woof!", "loud": "WOOF!!!"}

        self.last_bark = datetime.now(timezone.utc)
        bark = bark_sounds.get(intensity.lower(), bark_sounds["normal"])

        logger.debug(f"Bark requested with intensity: {intensity}")
        return {"sound": bark, "timestamp": self.last_bark.isoformat()}

    async def get_last_bark(self) -> Dict[str, str]:
        """Get information about the last bark"""
        if not self.last_bark:
            return {"status": "no_barks_yet"}

        return {"last_bark": self.last_bark.isoformat()}

    async def get_description(self) -> Dict[str, str]:
        """Get the semantic routing description for this server"""
        return {
            "description": self.ROUTE_DESCRIPTION,
            "capabilities": ["bark", "dog_responses"],
        }
