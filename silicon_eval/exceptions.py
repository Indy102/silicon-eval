"""Exception hierarchy for silicon-eval."""


class SiliconEvalError(Exception):
    """Base class for all silicon-eval errors."""


class RuntimeUnavailableError(SiliconEvalError):
    """A runtime's backing library is not installed or not usable on this machine."""


class ModelLoadError(SiliconEvalError):
    """A model could not be downloaded or loaded into the runtime."""


class InvalidStateError(SiliconEvalError):
    """An operation was called in the wrong order, e.g. generate() before load()."""
