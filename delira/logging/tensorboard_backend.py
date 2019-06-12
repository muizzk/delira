import tensorboardX
from threading import Event
from queue import Queue

from .base_backend import ThreadedBaseBackend
from .writer_backend import WriterLoggingBackend


class TensorboardBackend(WriterLoggingBackend):
    """
    The tensorboard logging backend
    """

    def __init__(self, writer_kwargs: dict = {},
                 abort_event: Event = None, queue: Queue = None):
        """

        Parameters
        ----------
        writer_kwargs : dict
            keyword arguments to initialize the actual writer
        abort_event : :class:`threading.Event`
            the abort-event for communication between logger and backend
        queue : :class:`queue.Queue`
            the piping queue

        """

        super().__init__(tensorboardX.SummaryWriter, writer_kwargs,
                         abort_event, queue)

    def _call_exec_fn(self, exec_fn, args):
        """
        Logs the actual function to log items and flushes results afterwards

        Parameters
        ----------
        exec_fn : function
            the function doing the acutal logging
        args : dict or list or tuple
            the arguments used for logging

        Returns
        -------
        Any
            return value of parent classes ``_call_exec_fn``

        """
        ret_val = super()._call_exec_fn(exec_fn, args)

        self._writer.file_writer.flush()

        return ret_val


class TensorboardThreadedBackend(ThreadedBaseBackend, TensorboardBackend):
    def __init__(self, writer_kwargs: dict = {}, abort_event: Event = None,
                 queue: Queue = None, name: str = None):
        ThreadedBaseBackend.__init__(
            self, abort_event=abort_event, queue=queue, name=name)
        TensorboardBackend.__init__(self, writer_kwargs=writer_kwargs,
                                    abort_event=abort_event, queue=queue)

    def run(self):
        return ThreadedBaseBackend.run(self)

    def _call_exec_fn(self, exec_fn, args):
        return TensorboardBackend._call_exec_fn(self, exec_fn, args)
