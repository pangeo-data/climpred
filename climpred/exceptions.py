class Error(Exception):
    """Base class for custom exceptions in climpred."""
    pass


class DimensionError(Error):
    """Exception raised when the input xarray object doesn't have the
    appropriate dimensions."""
    def __init__(self, message):
        self.message = message


class CoordinateError(Error):
    """Exception raised when the input xarray object doesn't have the
    appropriate coordinates."""
    def __init__(self, message):
        self.message = message
