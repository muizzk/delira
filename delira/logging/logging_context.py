import typing

from .utils import logger_exists, register_logger, unregister_logger, \
    log as _log
from .base_logger import make_logger


class LoggingContext(object):
    """
    Context Handler to forward the logging to a specific logger
    """

    def __init__(self, name, initialize_if_missing=False,
                 destroy_on_exit=False, **kwargs):
        """

        Parameters
        ----------
        name : str
            the logger name
        initialize_if_missing : bool
            whether to create a logger with the given name if not already
            existent
        destroy_on_exit : bool
            whether to destroy the specified logger during context exit
        **kwargs :
            additional keyword arguments (needed for logger initializing if
            ``initialize_if_missing`` is True

        """
        if logger_exists(name):
            self._name = name
        elif initialize_if_missing:
            register_logger(make_logger(**kwargs), name)
            self._name = name
            destroy_on_exit = True
        else:
            raise ValueError("No valid logger for name %s and "
                             "'initialize_if_missing' is False" % name)

        self._destroy_on_exit = destroy_on_exit

    def __enter__(self):
        """
        Enters the context and sets the global logging to the current logger

        """
        global log
        log = self.log
        return self

    def __exit__(self, *args, **kwargs):
        """
        Exits the context (deletes the logger if necessary) and resets the
        global logging function

        """
        if self._destroy_on_exit:
            _logger = unregister_logger(self._name)
            del _logger

        global log
        log = _log

    def log(self, msg: typing.Union[dict, list, str]):
        """
        Function to forward the actual logging to the current logger

        Parameters
        ----------
        msg : Any
            the logging message

        """
        _log(msg, self._name)