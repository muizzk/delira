from multiprocessing import Queue, Event
from .base_backend import BaseBackend, ThreadedBaseBackend
import logging


class Logger(object):
    """
    Basic Logger class
    """
    def __init__(self, backend: BaseBackend, max_queue_size: int = None,
                 level=logging.NOTSET):
        """

        Parameters
        ----------
        backend : :class:`BaseBackend`
            the actual logging backend
        max_queue_size : int
            the maximum limit of logging items to enqueue
            defaults to None which results in an unlimited queue
        level : int
            the logging level to fall back to

        """

        # 0 means unlimited size, but None is more readable
        if max_queue_size is None:
            max_queue_size = 0

        # set all attributes
        self._abort_event = Event()
        self._flush_queue = Queue(max_queue_size)
        self._backend = backend
        self._backend.set_queue(self._flush_queue)
        self._backend.set_event(self._abort_event)
        self._level = level

    def log(self, log_message: dict):
        """
        Logs one item (enqueues it to the logging pipe to be processed by the
        backend, if ``log_message`` is either dict or iterable of length 2;
        else logs it to logging module)

        Parameters
        ----------
        log_message : Any
            the message to log. Can be of any type, but everything else than a
            dict or a tuple/list of length 2 will be logged to the logging
            module not the specified backend

        Raises
        ------
        RuntimeError
            Abort event was set in backend

        """

        # if not self._backend.is_alive():
        #     self._backend.start()

        if self._abort_event.is_set():
            self.close()
            raise RuntimeError("Abort-Event in logging process was set: %s"
                               % self._backend.name)

        if isinstance(log_message, dict):
            self._flush_queue.put_nowait(log_message)
        elif isinstance(log_message, (tuple, list)) and len(log_message) == 2:
            self._flush_queue.put_nowait(log_message)
        else:
            logging.log(self._level, log_message)

    def close(self):
        """
        Closes the actual logger
        (Closes the queue and waits for the queue thread to finish); Sets the
        abort event

        """
        self._flush_queue.close()
        self._flush_queue.join_thread()

        self._abort_event.set()

    def __del__(self):
        self.close()


class SingleThreadedLogger(Logger):
    def __init__(self, backend: BaseBackend, max_queue_size: int = None,
                 level=logging.NOTSET):
        if isinstance(backend, ThreadedBaseBackend):
            raise ValueError("A threaded Backend lives in an own thread and "
                             "does not work with a SingleThreadedLogger")
        super().__init__(backend, max_queue_size, level)

    def log(self, log_message: dict):
        super().log(log_message)
        self._backend.run()


class MultiThreadedLogger(Logger):
    def __init__(self, backend: ThreadedBaseBackend, max_queue_size: int = None,
                 level=logging.NOTSET):

        if not isinstance(backend, ThreadedBaseBackend):
            raise ValueError("A non-threaded Backend lives in the main thread "
                             "and does not work with a MultiThreadedLogger")

        super().__init__(backend, max_queue_size, level)

        self._backend.start()

    def close(self):
        self._flush_queue.close()
        self._flush_queue.join_thread()

        self._abort_event.set()

        if self._backend.is_alive():
            self._backend.join()


def make_logger(backend: BaseBackend, max_queue_size: int = None,
                level=logging.NOTSET):
    if isinstance(backend, ThreadedBaseBackend):
        return MultiThreadedLogger(backend, max_queue_size, level)

    return SingleThreadedLogger(backend, max_queue_size, level)