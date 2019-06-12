import tensorboardX
from threading import Event
from queue import Queue

from .writer_backend import WriterLoggingBackend
from .base_backend import ThreadedBaseBackend


class VisdomBackend(WriterLoggingBackend):
    """
    The Visdom Logging Backend
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
        super().__init__(tensorboardX.visdom_writer.VisdomWriter, writer_kwargs,
                         abort_event, queue)


class VisdomThreadedBackend(ThreadedBaseBackend, VisdomBackend):
    def __init__(self, writer_kwargs: dict = {}, abort_event: Event = None,
                 queue: Queue = None, name: str = None):
        ThreadedBaseBackend.__init__(self, abort_event=abort_event, queue=queue,
                                     name=name)
        VisdomBackend.__init__(self, writer_kwargs=writer_kwargs,
                               abort_event=abort_event, queue=queue)

    def run(self):
        return ThreadedBaseBackend.run(self)

    def _call_exec_fn(self, exec_fn, args):
        return VisdomBackend._call_exec_fn(self, exec_fn, args)
