"""Project-wide exception hierarchy."""


class Track360Error(Exception):
    """Base class for expected tracking-system failures."""


class ConfigError(Track360Error):
    """Raised when configuration is missing, malformed, or inconsistent."""


class DecodeError(Track360Error):
    """Raised when an input frame cannot be decoded or validated."""


class GeometryError(Track360Error):
    """Raised when a spherical geometry operation cannot be completed."""


class ModelError(Track360Error):
    """Raised when a tracker model cannot load or infer."""


class ProtocolError(Track360Error):
    """Raised when frame ordering or a module protocol is violated."""


class OutputError(Track360Error):
    """Raised when results cannot be written or finalized."""
