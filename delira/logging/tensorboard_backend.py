from threading import Event
from queue import Queue
from typing import Optional, Callable, Union, Iterable, Mapping, Dict, Any
import numpy as np

from delira.logging.writer_backend import WriterLoggingBackend

# use torch SummaryWriter if possible, since this one has latest pytorch
# capabilities
try:
    from torch.utils.tensorboard import SummaryWriter
    from delira.models.backends.torch import AbstractPyTorchNetwork
    LOGDIR_KWARG = "log_dir"
except ImportError:
    from tensorboardX import SummaryWriter
    LOGDIR_KWARG = "logdir"
    # Mock torch Tensor for Annotations if necessary
    torch = type("torch", (), {'Tensor': None})

try:
    from tensorflow import Graph
except ImportError:
    # Mock Tensorflow if necessary for Annotations
    tensorflow = type("tensorflow", (), {"Graph": None})


class TensorboardBackend(WriterLoggingBackend):
    """
    A Tensorboard logging backend
    """

    def __init__(self, writer_kwargs: Optional[Dict[str, Any]] = None,
                 abort_event: Optional[Event] = None,
                 queue: Optional[Queue] = None) -> None:
        """

        Parameters
        ----------
        writer_kwargs : dict
            arguments to initialize a writer
        abort_event : :class:`threading.Event`
            the abortion event
        queue : :class:`queue.Queue`
            the queue holding all logging tasks
        """

        if writer_kwargs is None:
            writer_kwargs = {}

        if "logdir" in writer_kwargs:
            writer_kwargs[LOGDIR_KWARG] = writer_kwargs.pop("logdir")
        elif "log_dir" in writer_kwargs:
            writer_kwargs[LOGDIR_KWARG] = writer_kwargs.pop("log_dir")

        super().__init__(SummaryWriter, writer_kwargs,
                         abort_event, queue)

    def _call_exec_fn(self, exec_fn: Callable,
                      args: Union[Union[list, tuple, Iterable],
                                  Union[dict, Mapping]]) -> Any:
        """
        Helper Function calling the actual mapped function and flushing
        results to the writer afterwards

        Parameters
        ----------
        exec_fn : function
            the function which will execute the actual logging
        args : iterable (listlike) or mapping (dictlike)
            the arguments passed to the ``exec_fn``

        Returns
        -------
        Any
            the return value obtained by the ``exec_fn``

        """
        ret_val = super()._call_exec_fn(exec_fn, args)

        self._writer.file_writer.flush()

        return ret_val

    def __del__(self) -> None:
        """
        Function to be executed at deletion;
        Flushes all unsaved changes

        """
        self._writer.file_writer.flush()

    def _graph_pytorch(self, model: AbstractPyTorchNetwork,
                       input_to_model: Optional[torch.Tensor] = None,
                       verbose: bool = False,
                       **kwargs) -> None:
        """
        Function to log a PyTorch graph

        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            the model, whose graph shall be logged
        input_to_model : :class:`torch.Tensor`
            the input to the model; necessary for graph traversal
        verbose : bool
            verbosity option
        **kwargs :
            additional keyword arguments

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            model=model, input_to_model=input_to_model,
            verbose=verbose, **kwargs)

        self._writer.add_graph(*converted_args, **converted_kwargs)

    def _graph_tf(self, graph: tensorflow.Graph,
                  run_metadata: Optional = None) -> None:
        """
        Function to log a TensorFlow Graph

        Parameters
        ----------
        graph : :class:`tensorflow.Graph` or :class:`tensorflow.GraphDef`
        run_metadata :
            the run metadata

        Raises
        ------
        TypeError
            if given graph cannot be converted to graphdef

        """
        import tensorflow as tf
        from tensorboardX.proto.event_pb2 import Event, TaggedRunMetadata

        # convert to graphdef
        if isinstance(graph, tf.Graph):
            graphdef = graph.as_graph_def()
        elif isinstance(graph, tf.GraphDef):
            graphdef = graph
        elif hasattr(graph, "SerializeToString"):
            graphdef = graph
        else:
            raise TypeError("Invalid type given for graph: %s" %
                            graph.__class__.__name__)

        if run_metadata:
            run_metadata = TaggedRunMetadata(
                tag='step1', run_metadata=run_metadata.SerializeToString())

        self._writer._get_file_writer().add_event(
            Event(
                graph_def=graphdef.SerializeToString(),
                tagged_run_metadata=run_metadata))

    def _graph_onnx(self, prototxt: str) -> None:
        """
        Function to log a ONNX graph to file

        Parameters
        ----------
        prototxt : str
            filepath to a given prototxt file containing an ONNX graph

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            prototxt=prototxt)
        self._writer.add_onnx_graph(*converted_args, **converted_kwargs)

    def _embedding(self, mat: Union[np.ndarray, Iterable],
                   metadata: Optional = None,
                   label_img: Optional[Union[np.array, Iterable]] = None,
                   global_step: Optional[int] = None,
                   tag: str = 'default', metadata_header: Optional = None
                   ) -> None:
        """
        Function to create an embedding of given data

        Parameters
        ----------
        mat : array-like
            an arraylike object, which can be converted to a numpy array;
            holds the actual embedding value
        metadata :
            the embeddings metadata
        label_img : array-like
            an arraylike object, which can be converted to a numpy array;
            holds the label image
        global_step : int
            the global step
        tag : str
            the tag to store the embedding at
        metadata_header :
            the metadata header

        """
        converted_args, converted_kwargs = self.convert_to_npy(
            mat=mat, metadata=metadata, label_img=label_img,
            global_step=global_step
        )
        self._writer.add_embedding(*converted_args, **converted_kwargs)

    def _scalars(self, main_tag: str, tag_scalar_dict: dict,
                 global_step: Optional[int] = None,
                 walltime: Optional = None, sep: int = "/") -> None:
        """
        Function to log multiple scalars at once. Opposing to the base
        function, this is done sequentially rather then parallel to avoid
        creating new event files

        Parameters
        ----------
        main_tag : str
            the main tag, will be combined with the subtags inside the
            ``tag_scalar_dict``
        tag_scalar_dict : dict
            dictionary of (key, scalar) pairs
        global_step : int
            the global step
        walltime :
            the overall time
        sep : str
            the character separating maintag and subtag in the final tag

        """

        # log scalars sequentially
        for key, val in tag_scalar_dict.items():
            # combine tags
            new_tag = main_tag + sep + key
            self._scalar(new_tag, val, global_step=global_step,
                         walltime=walltime)

    @property
    def name(self) -> str:
        return "TensorFlow Backend"
