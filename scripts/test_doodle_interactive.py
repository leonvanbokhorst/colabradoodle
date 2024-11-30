import sys
from pathlib import Path

# Get absolute path to project root
project_root = Path(__file__).parent.parent.absolute()
src_path = project_root / "src"

# Add to Python path
sys.path.insert(0, str(src_path))

from mcp.server.doodle_server import DoodleServer
import asyncio

async def main():
    server = DoodleServer()

    # Test basic bark
    print("\nTesting basic bark:")
    result = await server.bark()
    print(f"Basic bark result: {result}")

    # Test different intensities
    print("\nTesting different intensities:")
    for intensity in ["quiet", "normal", "loud"]:
        result = await server.bark(intensity=intensity)
        print(f"{intensity.capitalize()} bark: {result}")

    # Test last bark tracking
    print("\nTesting last bark tracking:")
    last_bark = await server.get_last_bark()
    print(f"Last bark info: {last_bark}")

    # Test semantic routing description
    print("\nTesting semantic routing description:")
    description = await server.get_description()
    print(f"Server description: {description}")

if __name__ == "__main__":
    asyncio.run(main())
