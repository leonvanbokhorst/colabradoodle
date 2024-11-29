from typing import Optional
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from mcp import ClientSession
from utils.log_config import setup_logger

logger = setup_logger("BaseAgent")



class BaseAgent(ABC):
    """Base interface for all agents in the system.

    Provides common functionality and required interface methods for agents.

    Attributes:
        _session: MCP client session for communication
        _logger: Configured logger instance for this agent
    """

    def __init__(self, session: Optional[ClientSession] = None) -> None:
        """Initialize the base agent.

        Args:
            session: Optional MCP client session for communication
        
        Note:
            Logger is automatically configured with agent class name
        """
        self._session: Optional[ClientSession] = session
        self._logger: logging.Logger = setup_logger(
            self.__class__.__name__,
            log_dir=Path("logs")
        )

    @property
    @abstractmethod
    def ROUTE_DESCRIPTION(self) -> str:
        """Required route description for agent capabilities.
        
        Returns:
            str: Description of the agent's API routes and capabilities
        """
        raise NotImplementedError

    @abstractmethod 
    async def get_description(self) -> str:
        """Get the agent's detailed description.
        
        Returns:
            str: Detailed description of the agent's functionality
            
        Note: 
            This may include current state and dynamic capabilities
        """
        raise NotImplementedError

    async def set_session(self, session: ClientSession) -> None:
        """Set the MCP client session.

        Args:
            session: Active MCP client session to use for communication
            
        Raises:
            ValueError: If provided session is None
        """
        if session is None:
            raise ValueError("Session cannot be None")
            
        self._logger.info("Setting new client session")
        self._session = session
