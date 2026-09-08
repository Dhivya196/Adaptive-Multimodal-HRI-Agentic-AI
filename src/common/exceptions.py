"""Common exception hierarchy for the Adaptive Multimodal HRI Framework."""


class HRIException(Exception):
    """Base exception for all HRI multi-agent system errors."""
    pass


class PerceptionError(HRIException):
    """Raised when an error occurs during perception processing (vision, speech, gesture)."""
    pass


class CameraStreamError(PerceptionError):
    """Raised when a camera stream cannot be initialized, read, or closed properly."""
    pass


class ModelLoadError(PerceptionError):
    """Raised when an AI/ML perception model fails to load."""
    pass


class AgentExecutionError(HRIException):
    """Raised when an agent encounters an unrecoverable failure during execution."""
    pass


class ConfigurationError(HRIException):
    """Raised when configuration values are missing, invalid, or unreadable."""
    pass
