from delira.training.backends.tf_eager.utils import create_optims_default
from delira.training.backends.tf_eager.utils import convert_to_numpy
from delira.training.base_trainer import BaseNetworkTrainer
from delira.io.tf import save_checkpoint_eager, load_checkpoint_eager
from delira.models.backends.tf_eager import AbstractTfEagerNetwork, \
    DataParallelTfEagerNetwork
from delira.data_loading import BaseDataManager
from delira.training.callbacks import AbstractCallback
import logging
import os
from functools import partial

import tensorflow as tf
from typing import Optional, Union, Callable, Iterable, Type

logger = logging.getLogger(__name__)


class TfEagerNetworkTrainer(BaseNetworkTrainer):
    def __init__(self,
                 network: AbstractTfEagerNetwork,
                 save_path: str,
                 key_mapping: dict,
                 losses: dict,
                 optimizer_cls: Type[tf.train.Optimizer],
                 optimizer_params: Optional[dict] = None,
                 train_metrics: Optional[dict] = None,
                 val_metrics: Optional[dict] = None,
                 lr_scheduler_cls: Optional[Type[AbstractCallback]] = None,
                 lr_scheduler_params: Optional[dict] = None,
                 gpu_ids: Optional[Union[list, Iterable, tuple]] = None,
                 save_freq: int = 1,
                 optim_fn: Callable = create_optims_default,
                 logging_type: str = "tensorboardx",
                 logging_kwargs: Optional[dict] = None,
                 fold: int = 0,
                 callbacks: Optional[Union[list, Iterable, tuple]] = None,
                 start_epoch: int = 1,
                 metric_keys: Optional[dict] = None,
                 convert_batch_to_npy_fn: Callable = convert_to_numpy,
                 val_freq: int = 1,
                 **kwargs):
        """

        Parameters
        ----------
        network : :class:`AbstractTfEagerNetwork`
            the network to train
        save_path : str
            path to save networks to
        key_mapping : dict
            a dictionary containing the mapping from the ``data_dict`` to
            the actual model's inputs.
            E.g. if a model accepts one input named 'x' and the data_dict
            contains one entry named 'data' this argument would have to
            be ``{'x': 'data'}``
        losses : dict
            dictionary containing the training losses
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of choice
        optimizer_params : dict
            keyword arguments passed to optimizer during construction
        train_metrics : dict, optional
            metrics, which will be evaluated during train phase
            (should work on numpy arrays)
        val_metrics : dict, optional
            metrics, which will be evaluated during test phase
            (should work on numpy arrays)
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        save_freq : int
            integer specifying how often to save the current model's state.
            State is saved every state_freq epochs
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        logging_type : str or callable
            the type of logging. If string: it must be one of
            ["visdom", "tensorboardx"]
            If callable: it must be a logging handler class
        logging_kwargs : dict
            dictionary containing all logging keyword arguments
        fold : int
            current cross validation fold (0 per default)
        callbacks : list
            initial callbacks to register
        start_epoch : int
            epoch to start training at
        metric_keys : dict
            dict specifying which batch_dict entry to use for which metric as
            target; default: None, which will result in key "label" for all
            metrics
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, per default this is
            the identity function
        val_freq : int
            validation frequency specifying how often to validate the trained
            model (a value of 1 denotes validating every epoch,
            a value of 2 denotes validating every second epoch etc.);
            defaults to 1
        **kwargs :
            Additional keyword arguments

        """

        # prevent mutable default arguments
        if logging_kwargs is None:
            logging_kwargs = {}
        if callbacks is None:
            callbacks = []
        if gpu_ids is None:
            gpu_ids = []
        if lr_scheduler_params is None:
            lr_scheduler_params = {}
        if val_metrics is None:
            val_metrics = {}
        if train_metrics is None:
            train_metrics = {}
        if optimizer_params is None:
            optimizer_params = {}

        # check if eager execution is enabled
        assert tf.executing_eagerly()

        super().__init__(network=network,
                         save_path=save_path,
                         losses=losses,
                         optimizer_cls=optimizer_cls,
                         optimizer_params=optimizer_params,
                         train_metrics=train_metrics,
                         val_metrics=val_metrics,
                         lr_scheduler_cls=lr_scheduler_cls,
                         lr_scheduler_params=lr_scheduler_params,
                         gpu_ids=gpu_ids,
                         save_freq=save_freq,
                         optim_fn=optim_fn,
                         key_mapping=key_mapping,
                         logging_type=logging_type,
                         logging_kwargs=logging_kwargs,
                         fold=fold,
                         callbacks=callbacks,
                         start_epoch=start_epoch,
                         metric_keys=metric_keys,
                         convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                         val_freq=val_freq,
                         **kwargs
                         )

        self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                    lr_scheduler_cls, lr_scheduler_params,
                    key_mapping, convert_batch_to_npy_fn, gpu_ids)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup(self, network: AbstractTfEagerNetwork, optim_fn: Callable,
               optimizer_cls: Type[tf.train.Optimizer],
               optimizer_params: dict,
               lr_scheduler_cls: Type[AbstractCallback],
               lr_scheduler_params: dict, key_mapping: dict,
               convert_batch_to_npy_fn: Callable,
               gpu_ids: Union[list, tuple, Iterable]):
        """
        Defines the Trainers Setup

        Parameters
        ----------
        network : instance of :class: `AbstractTfNetwork`
            the network to train
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of choice
        optimizer_params : dict
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, per default this is
            the identity function
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead

        Raises
        ------
        RuntimeError
            if multiple GPU ids passed
        """

        if gpu_ids and tf.test.is_gpu_available():
            self.use_gpu = True
            if len(gpu_ids) > 1:
                raise RuntimeError("Multiple GPUs not yet supported")
                # logger.warning(
                #     "multi-GPU training not yet tested!")

                # network = DataParallelTfEagerNetwork(network, gpu_ids)
                #
                # self.input_device = "/cpu:0"
                # self.output_device = "/cpu:0"
            else:
                self.input_device = "/gpu:%d" % gpu_ids[0]
                self.output_device = "/gpu:%d" % gpu_ids[0]
        else:
            self.use_gpu = False
            self.input_device = "/cpu:0"
            self.output_device = "/cpu:0"

        self.optimizers = optim_fn(optimizer_cls, **optimizer_params)

        super()._setup(network, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                       key_mapping, convert_batch_to_npy_fn,
                       network.prepare_batch)
        self._prepare_batch = partial(self._prepare_batch,
                                      input_device=self.input_device,
                                      output_device=self.output_device)

        # Load latest epoch file if available
        if os.path.isdir(self.save_path):
            # check all files in directory starting with "checkpoint" and
            # not ending with "_best.meta"
            latest_state_path, latest_epoch = self._search_for_prev_state(
                self.save_path
            )

            if latest_state_path is not None:
                logger.info("Attempting to load state from previous \
                                training from %s" % latest_state_path)

                self.update_state(latest_state_path)
                self.start_epoch = latest_epoch

    def _at_training_end(self):
        """
        Defines Behaviour at end of training: Loads best model if available

        Returns
        -------
        :class:`AbstractTfNetwork`
            best network

        """
        if os.path.isfile(os.path.join(self.save_path,
                                       'checkpoint_best.meta')):

            # load best model and return it.
            self.update_state(os.path.join(self.save_path,
                                           'checkpoint_best')
                              )

        return self.module

    def _train_single_epoch(self, batchgen: BaseDataManager, epoch: int,
                            verbose: bool = False):
        """
        Trains the network a single epoch

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the training batches
        epoch : int
            current epoch

        """
        self.module.trainable = True

        return super()._train_single_epoch(batchgen, epoch, verbose=verbose)

    def predict_data_mgr(self, datamgr: BasseDataManager,
                         batchsize: Optional[int] = None,
                         metrics: Optional[dict] = None,
                         metric_keys: Optional[dict] = None,
                         verbose: bool = False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        **kwargs :
            additional keyword arguments

        """
        if metrics is None:
            metrics = {}
        self.module.trainable = False

        return super().predict_data_mgr(datamgr, batchsize, metrics,
                                        metric_keys, verbose=verbose, **kwargs)

    def save_state(self, file_name: str, *args, **kwargs):
        """
        saves the current state via :func:`delira.io.tf.save_checkpoint_eager`

        Parameters
        ----------
        file_name : str
            filename to save the state to
        """
        save_checkpoint_eager(file_name, self.module, self.optimizers,
                              *args, **kwargs)

    def load_state(self, file_name: str, *args, **kwargs):
        """
        Loads the new state from file via
        :func:`delira.io.tf.load_checkpoint_eager`

        Parameters
        ----------
        file_name : str
            the file to load the state from
        Returns
        -------

        """
        return load_checkpoint_eager(
            file_name, self.module, self.optimizers)

    @staticmethod
    def _search_for_prev_state(
            path: str,
            extensions: Optional[Union[list, tuple, Iterable]] = None):
        """
        Helper function to search in a given path for previous epoch states
        (indicated by extensions)

        Parameters
        ----------
        path : str
            the path to search in
        extensions : list
            list of strings containing valid file extensions for checkpoint
            files

        Returns
        -------
        str
            the file containing the latest checkpoint (if available)
        None
            if no latst checkpoint was found
        int
            the latest epoch (1 if no checkpoint was found)

        """
        if extensions is None:
            extensions = [".meta"]
        return BaseNetworkTrainer._search_for_prev_state(path, extensions)
