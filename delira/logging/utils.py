import typing
from .base_logger import Logger

# Global dict handling all available loggers
_AVAILABLE_LOGGERS = {}


def log(msg: typing.Union[dict, list, str], name=None):
    """
    Function to provide global logging API

    Parameters
    ----------
    msg : any
        the function message to log; If not dict or list/tuple of length 2 this
        will be logged to python's logging module
    name : str
        the name specifying the logger to use

    Returns
    -------
    Any
        return type of the loggers ``log`` function

    """

    if name is None:
        _logger = list(_AVAILABLE_LOGGERS.values())[0]
    else:
        _logger = _AVAILABLE_LOGGERS[name]

    assert isinstance(_logger, Logger)

    return _logger.log(msg)


def logger_exists(name: str):
    """
    Checks for logger existence

    Parameters
    ----------
    name : str
        the name of the logger to check for

    Returns
    -------
    bool
        True if a logger with the given name already exists

    """
    return name in _AVAILABLE_LOGGERS


def register_logger(logger: Logger, name: str, overwrite=False):
    """
    registers a new logger

    Parameters
    ----------
    logger : :class:`Logger`
        the logger to register
    name : str
        the name to register it at
    overwrite : bool
        whether to overwrite already existing loggers (if necessary)

    Returns
    -------
    :class:`Logger`
        the registered logger

    """

    if name not in _AVAILABLE_LOGGERS or overwrite:
        _AVAILABLE_LOGGERS[name] = logger

    return _AVAILABLE_LOGGERS[name]


def unregister_logger(name: str):
    """
    Unregisters a logger

    Parameters
    ----------
    name : str
        the name specifying the logger to unregister

    Returns
    -------
    :class:`Logger`
        The unregistered logger

    """
    return _AVAILABLE_LOGGERS.pop(name)


def get_logger(name):
    """
    Returns an already registered logger

    Parameters
    ----------
    name : str
        the name specifying the logger to return

    Returns
    -------
    :class:`Logger`
        the logger to return

    """
    return _AVAILABLE_LOGGERS[name]
