import asyncio
import logging
import argparse
from mcp.server.stdio import stdio_server
from .server import RegistryServer

async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="MCP Registry Server")
    parser.add_argument("--name", default="registry", help="Name of the registry server")
    parser.add_argument("--timeout", type=int, default=60, 
                       help="Server timeout in seconds (default: 60)")
    args = parser.parse_args()
    
    server = RegistryServer(name=args.name, server_timeout_seconds=args.timeout)
    server._cleanup_task = asyncio.create_task(server._cleanup_loop())
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
    
    if server._cleanup_task:
        server._cleanup_task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 