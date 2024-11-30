from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import logging
from mcp.server.server import Server


class RegistryError(Exception):
    """Base error class for MCP Registry"""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class RegisteredServer:
    """Represents a registered MCP server"""

    id: str
    name: str
    description: str
    capabilities: List[str]
    endpoint: str
    last_heartbeat: datetime
    metadata: Dict[str, str]


class RegistryServer(Server):
    """MCP Registry Server for publishing and discovering MCP servers"""

    def __init__(self, name: str = "registry", server_timeout_seconds: int = 60):
        super().__init__(name=name)
        self.servers: Dict[str, RegisteredServer] = {}
        self.logger = logging.getLogger("mcp.registry")
        self._cleanup_task: Optional[asyncio.Task] = None
        self.server_timeout_seconds = server_timeout_seconds
        self.tools = {
            "register_server": self.register_server,
            "heartbeat": self.heartbeat,
            "discover_servers": self.discover_servers,
        }

    async def register_server(
        self,
        server_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        endpoint: str,
        metadata: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """Register a new MCP server with the registry"""
        if server_id in self.servers:
            raise RegistryError(
                "server_exists", f"Server {server_id} already registered"
            )

        self.servers[server_id] = RegisteredServer(
            id=server_id,
            name=name,
            description=description,
            capabilities=capabilities,
            endpoint=endpoint,
            last_heartbeat=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.logger.info(f"Registered new server: {server_id}")
        return {"status": "registered", "server_id": server_id}

    async def heartbeat(self, server_id: str) -> Dict[str, str]:
        """Update server heartbeat timestamp"""
        if server_id not in self.servers:
            raise RegistryError(
                "server_not_found", f"Server {server_id} not registered"
            )

        self.servers[server_id].last_heartbeat = datetime.utcnow()
        return {"status": "ok"}

    async def discover_servers(
        self, capability: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Discover available MCP servers, optionally filtered by capability"""
        servers = []
        for server in self.servers.values():
            if capability and capability not in server.capabilities:
                continue
            servers.append(
                {
                    "id": server.id,
                    "name": server.name,
                    "description": server.description,
                    "capabilities": server.capabilities,
                    "endpoint": server.endpoint,
                }
            )
        return servers

    async def _cleanup_stale_servers(self) -> None:
        """Remove servers that haven't sent a heartbeat within the timeout period.

        This method checks all registered servers and removes any that have exceeded

        the configured timeout period since their last heartbeat.
        """
        now = datetime.utcnow()
        stale_servers = [
            server_id
            for server_id, server in self.servers.items()
            if (now - server.last_heartbeat).total_seconds()
            > self.server_timeout_seconds
        ]

        for server_id in stale_servers:
            del self.servers[server_id]
            self.logger.info(
                f"Removed stale server: {server_id} (timeout: {self.server_timeout_seconds}s)"
            )

    async def _cleanup_loop(self):
        """Periodically run the cleanup process to remove stale server registrations.

        This method controls the cleanup loop timing and error handling, delegating
        the actual cleanup work to _cleanup_stale_servers.

        """
        while True:
            try:
                await self._cleanup_stale_servers()

                # Sleep for half the timeout period, but between 5 and 30 seconds
                await asyncio.sleep(max(5, min(30, self.server_timeout_seconds / 2)))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)  # On error, wait 30s before retry
