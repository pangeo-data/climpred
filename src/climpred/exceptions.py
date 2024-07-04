class Error(Exception):
    """Base class for custom exceptions in climpred."""

    pass


class CoordinateError(Error):
    """Exception raised when the input xarray object doesn't have the
    appropriate coordinates."""

    def __init__(self, message):
        self.message = message


class DatasetError(Error):
    """Exception raised when there is any issues related to a dataset."""

    def __init__(self, message):
        self.message = message


class DimensionError(Error):
    """Exception raised when the input xarray object doesn't have the
    appropriate dimensions."""

    def __init__(self, message):
        self.message = message


class KeywordError(Error):
    """Exception raised when the keyword used in the function is not appropriate or
    does not work in the given case."""

    def __init__(self, message):
        self.message = message


class VariableError(Error):
    """Exception raised when the input xarray object doesn't have the
    appropriate variables."""

    def __init__(self, message):
        self.message = message
