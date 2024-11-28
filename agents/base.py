from typing import Optional
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from mcp import ClientSession


class BaseAgent(ABC):
    """Base interface for all agents in the system.

    Provides common functionality and required interface methods for agents.
    """

    def __init__(self, session: Optional[ClientSession] = None) -> None:
        """Initialize the base agent.

        Args:
            session: Optional MCP client session for communication
        """
        self._session = session
        self._logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging to both console and file.

        Returns:
            Logger: Configured logger instance
        """
        # Get logger for this class
        logger = logging.getLogger(self.__class__.__name__)
        
        # Set the base logging level
        logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(
            log_dir / f"{self.__class__.__name__.lower()}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        )
        
        # Add handlers if they haven't been added already
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger

    @property
    @abstractmethod
    def ROUTE_DESCRIPTION(self) -> str:
        """Route description for agent capabilities."""
        pass

    @abstractmethod
    async def get_description(self) -> str:
        """Get the agent's route description."""
        pass

    async def set_session(self, session: ClientSession) -> None:
        """Set the MCP client session.

        Args:
            session: Active MCP client session
        """
        self._logger.info("Setting new client session")
        self._session = session 