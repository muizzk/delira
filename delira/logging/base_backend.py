from queue import Empty
from abc import abstractmethod, ABCMeta
from threading import Event, Thread
from queue import Queue
import traceback


class FigureManager:
    """Basic Context Manager to push all matplotlib figures with the specified
    ``push_fn``"""

    def __init__(self, push_fn, figure_kwargs: dict, push_kwargs: dict):
        """

        Parameters
        ----------
        push_fn : function
            a function which logs arbitrary matplotlib figures
        figure_kwargs : dict
            all keyword arguments to create a figure
        push_kwargs : dict
            all keyword arguments which are necessary to push the figure via
            ``push_fn``
        """
        self._push_fn = push_fn
        self._figure_kwargs = figure_kwargs
        self._push_kwargs = push_kwargs
        self._fig = None

    def __enter__(self):
        """
        Creates the figure while entering the context

        """
        from matplotlib.pyplot import figure
        self._fig = figure(**self._figure_kwargs)

    def __exit__(self, *args):
        """
        Pushes the figure via ``push_fn`` and closes it afterwards

        """
        from matplotlib.pyplot import close
        self._push_fn(figure=self._fig, **self._push_kwargs)

        close(self._fig)
        self._fig = None


_FUNCTIONS_WITHOUT_STEP = ["graph_pytorch", "graph_tf", "graph_onnx",
                           "embedding"]


class BaseBackend(object, metaclass=ABCMeta):
    """
    Basic Logging backend; Defines API for all future backends
    """

    def __init__(self, abort_event: Event = None, queue: Queue = None):
        """

        Parameters
        ----------
        abort_event : :class:`threading.Event`
            the event specifying when to abort the logging process
        queue : :class:`queue.Queue`
            a queue (used as asynchronous pipe) holding all elements to log
        """
        super().__init__()

        # initialize with empty dict
        self.KEYWORD_FN_MAPPING = {}

        self.daemon = True

        # set all attributes
        self._queue = queue
        self._abort_event = abort_event
        self._global_steps = {}

        # update keyword-function mapping
        self.KEYWORD_FN_MAPPING.update(**{
            "image": self._image,
            "img": self._image,
            "picture": self._image,
            "images": self._images,
            "imgs": self._images,
            "pictures": self._images,
            "image_with_boxes": self._image_with_boxes,
            "bounding_boxes": self._image_with_boxes,
            "bboxes": self._image_with_boxes,
            "scalar": self._scalar,
            "value": self._scalar,
            "scalars": self._scalars,
            "values": self._scalars,
            "histogram": self._histogram,
            "hist": self._histogram,
            "figure": self._figure,
            "fig": self._figure,
            "audio": self._audio,
            "sound": self._audio,
            "video": self._video,
            "text": self._text,
            "graph_pytorch": self._graph_pytorch,
            "graph_tf": self._graph_tf,
            "graph_onnx": self._graph_onnx,
            "embedding": self._embedding,
            "pr_curve": self._pr_curve,
            "pr": self._pr_curve,
            "scatter": self._scatter,
            "line": self._line,
            "curve": self._line,
            "stem": self._stem,
            "heatmap": self._heatmap,
            "hm": self._heatmap,
            "bar": self._bar,
            "boxplot": self._boxplot,
            "surface": self._surface,
            "contour": self._contour,
            "quiver": self._quiver,
            # "mesh": self._mesh
        })

    def _log_item(self):
        """
        Helper Function to log a single item (from ``self.queue``

        Raises
        ------
        ValueError
            item to log is not a dict

        """
        process_item = self._queue.get_nowait()
        if isinstance(process_item, dict):
            for key, val in process_item.items():
                execute_fn = self.KEYWORD_FN_MAPPING[str(key).lower()]
                val = self.resolve_global_step(str(key).lower(), **val)

                self._call_exec_fn(execute_fn, val)

        else:
            raise ValueError("Invalid Value passed for logging: %s"
                             % str(process_item))

    def resolve_global_step(self, key, **val):
        """
        Checks if function needs a global step or not

        Parameters
        ----------
        key : str
            function key (must be in ``KEYWORD_FN_MAPPING``)
        **val :
            keyword arguments for logging

        Returns
        -------
        dict
            dictionary containing keyword arguments with resolved global step

        """
        # check if function should be processed statically
        # (no time update possible)
        if str(key).lower() not in _FUNCTIONS_WITHOUT_STEP:

            if "tag" in val:
                tag = "tag"
            elif "main_tag" in val:
                tag = "main_tag"
            else:
                raise ValueError("No valid tag found to extract global step")

            # check if global step is given
            if "global_step" not in val or val["global_step"] is None:

                # check if tag is already part of internal global steps
                if val[tag] in self._global_steps:
                    # if already existent: increment step for given tag
                    step = self._global_steps[val[tag]]
                    self._global_steps[val[tag]] += 1

                else:
                    # if not existent_ set step for given tag to zero
                    step = 0
                    self._global_steps[val[tag]] = step

                val.update({"global_step": step})

            elif "global_step" in val:
                self._global_steps[tag] = val["global_step"]

        return val

    def run(self):
        """
        Tries to log a single item

        """
        try:
            self._log_item()

        except Empty:
            pass

        except Exception as e:
            self._abort_event.set()
            raise e

    def set_queue(self, queue: Queue):
        """
        Setter for the pipe queue

        Parameters
        ----------
        queue : :class:`queue.Queue`

        """
        self._queue = queue

    def set_event(self, event: Event):
        """
        Setter for the abortion event

        Parameters
        ----------
        event : :class:`threading.Event`

        """
        self._abort_event = event

    def _call_exec_fn(self, exec_fn, args):
        """
        Helper function to call the actual execution function (obtained from
        ``KEYWORD_FN_MAPPING``) and correctly feeding the arguments to it

        Parameters
        ----------
        exec_fn : function
            the function to execute for logging
        args : dict, tuple or list
            the logging arguments

        Returns
        -------
        Any
            return Value from logging function

        Raises
        ------
        TypeError
            if ``args`` is not of type dict, tuple or list

        """

        if isinstance(args, dict):
            ret_val = exec_fn(**args)
        elif isinstance(args, (tuple, list)):
            ret_val = exec_fn(*args)

        else:
            raise TypeError("Invalid type for args. Must be either dict, "
                            "tuple or list, but got %s."
                            % args.__class__.__name__)

        return ret_val

    @abstractmethod
    def _image(self, *args, **kwargs):
        """
        Abstract function to log a single image
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _images(self, *args, **kwargs):
        """
        Abstract function to log multiple images
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _image_with_boxes(self, *args, **kwargs):
        """
        Abstract function to log a single image with multiple bounding boxes
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _scalar(self, *args, **kwargs):
        """
        Abstract function to log a single scalar value
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _scalars(self, *args, **kwargs):
        """
        Abstract function to log multiple scalar values
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _histogram(self, *args, **kwargs):
        """
        Abstract function to log multiple values to a histogram
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _figure(self, *args, **kwargs):
        """
        Abstract function to log a single matplotlib figure
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _audio(self, *args, **kwargs):
        """
        Abstract function to log a single audio signal
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _video(self, *args, **kwargs):
        """
        Abstract function to log multiple frames as video
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _text(self, *args, **kwargs):
        """
        Abstract function to log a single string
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _graph_pytorch(self, *args, **kwargs):
        """
        Abstract function to log a pytorch graph
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _graph_tf(self, *args, **kwargs):
        """
        Abstract function to log a tensorflow graph
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _graph_onnx(self, *args, **kwargs):
        """
        Abstract function to log a ONNX graph
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _embedding(self, *args, **kwargs):
        """
        Abstract function to log multiple values as embedding
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @abstractmethod
    def _pr_curve(self, *args, **kwargs):
        """
        Abstract function to log a basic pr-curve
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    def _scatter(self, plot_kwargs: dict, figure_kwargs={}, **kwargs):
        """
        Abstract function to log create a scatter plot from multiple points
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """

        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import scatter

            scatter(self, **plot_kwargs)

    def _line(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic line-plot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """

        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import plot
            plot(**plot_kwargs)

    def _stem(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic stem plot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import stem
            stem(**plot_kwargs)

    def _heatmap(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic heatmap
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from seaborn import heatmap
            heatmap(**plot_kwargs)

    def _bar(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic barplot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import bar
            bar(**plot_kwargs)

    def _boxplot(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic boxplot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import boxplot
            boxplot(**plot_kwargs)

    def _surface(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic surface plot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from seaborn import kdeplot

            kdeplot(**plot_kwargs)

    def _contour(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic contour plot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import contour

            contour(**plot_kwargs)

    def _quiver(self, plot_kwargs={}, figure_kwargs={}, **kwargs):
        """
        Abstract function to log a basic quiver plot
        (should be overwritten by subclasses)

        Parameters
        ----------
        *args :
            arbitrary positional arguments (used for logging)
        **kwargs :
            arbitrary keyword arguments (used for logging)

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        with FigureManager(self._figure, figure_kwargs, kwargs):
            from matplotlib.pyplot import quiver
            quiver(**plot_kwargs)


class ThreadedBaseBackend(BaseBackend, Thread):
    def __init__(self, abort_event: Event = None, queue: Queue = None,
                 name: str = None):
        Thread.__init__(self, name=name)
        BaseBackend.__init__(self, abort_event, queue)

    def run(self):
        while True:
            try:
                super().run()

            except Exception as e:
                tb = traceback.format_exc()
                logging.exception(tb)
                logging.exception(e)
                raise e

            finally:
                if self._abort_event.is_set():
                    break

    def set_event(self, event: Event):
        assert not self.is_alive()
        return BaseBackend.set_event(self, event)

    def set_queue(self, queue: Queue):
        assert not self.is_alive()
        return BaseBackend.set_queue(self, queue)