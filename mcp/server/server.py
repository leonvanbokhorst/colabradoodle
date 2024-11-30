class Server:
    """Base server class for all server implementations."""

    def __init__(self, name: str = "unnamed"):
        """Initialize the base server.

        Args:
            name: The name of the server instance
        """
        self.name = name
