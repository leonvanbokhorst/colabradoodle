from typing import Dict, Optional
from datetime import datetime, UTC
import logging
from utils.log_config import setup_logger
from mcp.server.server import Server

logger = setup_logger("DoodleServer")


class DoodleServer(Server):
    """MCP Server that provides dog-related functionality"""

    BARK_SOUNDS = {
        "quiet": "woof...",
        "normal": "WOOF",
        "loud": "WOOF!!!",
    }

    def __init__(self, name: str = "doodle"):
        """Initialize the doodle server.

        Args:
            name: The name of the server instance
        """
        super().__init__(name=name)

        self.last_bark: Optional[datetime] = None

    async def bark(self, intensity: str = "normal") -> Dict[str, str]:
        """Simulate a dog bark with different intensities."""
        self.last_bark = datetime.now(UTC)
        bark = self.BARK_SOUNDS.get(intensity.lower(), self.BARK_SOUNDS["normal"])

        return {"sound": bark, "timestamp": self.last_bark.isoformat()}

    async def get_last_bark(self) -> Dict[str, str]:
        """Get information about the last bark"""
        if not self.last_bark:
            return {"status": "no_barks_yet"}
        return {"last_bark": self.last_bark.isoformat()}
