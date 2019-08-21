import logging
import pickle
import os
from datetime import datetime

import copy

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, \
    StratifiedShuffleSplit, ShuffleSplit

from delira.data_loading import BaseDataManager
from delira.models import AbstractNetwork

from delira.training.parameters import Parameters
from delira.training.base_trainer import BaseNetworkTrainer
from delira.training.predictor import Predictor
from delira.training.utils import convert_to_numpy_identity
logger = logging.getLogger(__name__)


class BaseExperiment(object):
    """
    Baseclass for Experiments.
    Implements:
    * Setup-Behavior for Models, Trainers and Predictors (depending on train
        and test case)
    * The K-Fold logic (including stratified and random splitting)
    * Argument Handling
    """

    def __init__(self,
                 model_cls: AbstractNetwork,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 verbose=True,
                 logging_type="tensorboard",
                 logging_kwargs=None,
                 convert_to_npy=convert_to_numpy_identity
                 ):
        """

        Parameters
        ----------
        model_cls : Subclass of :class:`AbstractNetwork`
            the class implementing the model to train
        name : str or None
            the Experiment's name
        save_path : str or None
            the path to save the results and checkpoints to.
            if None: Current working directory will be used
        key_mapping : dict
            mapping between data_dict and model inputs (necessary for
            prediction with :class:`Predictor`-API)
        verbose : bool
            verbosity argument
        logging_type : str
            which type of logging to use; must be one of
            'visdom' | 'tensorboad'
            Defaults to 'tensorboard'
        logging_kwargs : dict
            a dictionary containing all necessary keyword arguments to
            properly initialize the logging
        convert_to_npy : function
            function to convert all outputs and metrics to numpy types

        """

        if name is None:
            name = "UnnamedExperiment"
        self.name = name

        if save_path is None:
            save_path = os.path.abspath(".")

        # append name and date-time-stamp to save_path
        self.save_path = os.path.join(save_path, name,
                                      str(datetime.now().strftime(
                                          "%y-%m-%d_%H-%M-%S")))

        if os.path.isdir(self.save_path):
            logger.warning("Save Path %s already exists")

        os.makedirs(self.save_path, exist_ok=True)

        assert key_mapping is not None
        self.key_mapping = key_mapping

        self.model_cls = model_cls
        self._run = 0
        self.verbose = verbose
        self._logging_type = logging_type
        self._logging_kwargs = logging_kwargs
        self._convert_to_npy = convert_to_npy

    def run(self, params, train_data: BaseDataManager,
            val_data: BaseDataManager, optim_builder, gpu_ids=None,
            checkpoint_freq=1, reduce_mode='mean', val_score_key=None,
            val_score_mode="lowest", val_freq=1, callbacks=None,
            trainer_cls=BaseNetworkTrainer, **kwargs):
        """
        Function to run the actual training

        Parameters
        ----------
        params : :class:`Parameters`
            the parameters containing the model and training kwargs
        train_data : :class:`BaseDataManager`
            the datamanager containing the training data
        val_data : :class:`BaseDataManager` or None
            the datamanager containing the validation data (may also be None
            for no validation)
        optim_builder : function
            the function creating suitable optimizers and returns them as dict
        gpu_ids : list
            a list of integers representing the GPUs to use;
            if empty or None: No gpus will be used at all
        checkpoint_freq : int
            determines how often to checkpoint the training
        reduce_mode : str
            determines how to reduce metrics; must be one of
            'mean' | 'sum' | 'first_only'
        val_score_key : str
            specifies which metric to use for best model selection.
        val_score_mode : str
            determines whether a high or a low val_score is best. Must be one
            of 'highest' | 'lowest'
        val_freq : int
            specifies how often to run a validation step
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        trainer_cls : type
            the class implementing the actual training routine
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            the trained model

        """

        params.permute_training_on_top()
        training_params = params.training
        trainer = self._setup_training(params, gpu_ids=gpu_ids,
                                       optim_builder=optim_builder,
                                       callbacks=callbacks,
                                       checkpoint_freq=checkpoint_freq,
                                       val_freq=val_freq,
                                       trainer_cls=trainer_cls,
                                       **kwargs)

        self._run += 1

        num_epochs = training_params.nested_get("num_epochs")

        return trainer.train(num_epochs, train_data, val_data,
                             val_score_key, val_score_mode,
                             reduce_mode=reduce_mode)

    def resume(self, save_path, params, train_data: BaseDataManager,
               val_data: BaseDataManager, optim_builder, gpu_ids=None,
               checkpoint_freq=1, reduce_mode='mean', val_score_key=None,
               val_score_mode="lowest", val_freq=1, callbacks=None,
               **kwargs):
        """
        Function to resume the training from an earlier state

        Parameters
        ----------
        save_path : str
            the path containing the earlier training state
        params : :class:`Parameters`
            the parameters containing the model and training kwargs
        train_data : :class:`BaseDataManager`
            the datamanager containing the training data
        val_data : :class:`BaseDataManager` or None
            the datamanager containing the validation data (may also be None
            for no validation)
        optim_builder : function
            the function creating suitable optimizers and returns them as dict
        gpu_ids : list
            a list of integers representing the GPUs to use;
            if empty or None: No gpus will be used at all
        checkpoint_freq : int
            determines how often to checkpoint the training
        reduce_mode : str
            determines how to reduce metrics; must be one of
            'mean' | 'sum' | 'first_only'
        val_score_key : str
            specifies which metric to use for best model selection.
        val_score_mode : str
            determines whether a high or a low val_score is best. Must be one
            of 'highest' | 'lowest'
        val_freq : int
            specifies how often to run a validation step
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            the trained model

        """

        return self.run(params=params, train_data=train_data,
                        val_data=val_data, optim_builder=optim_builder,
                        gpu_ids=gpu_ids, checkpoint_freq=checkpoint_freq,
                        reduce_mode=reduce_mode,
                        val_score_key=val_score_key,
                        val_score_mode=val_score_mode, val_freq=val_freq,
                        callbacks=callbacks,
                        save_path=save_path, **kwargs)

    def test(self, model, test_data, prepare_batch, callbacks=None,
             predictor_cls=Predictor, metrics=None, metric_keys=None,
             **kwargs):
        """
        Setup and run testing on a given network
        Parameters
        ----------
        model : :class:`AbstractNetwork`
            the (trained) network to test
        test_data : :class:`BaseDataManager`
            the data to use for testing
        metrics : dict
            the metrics to calculate
        metric_keys : dict of tuples
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label`` will be used for metric calculation
        prepare_batch : function
            function to convert a batch-dict to a format accepted by the model.
            This conversion typically includes dtype-conversion, reshaping,
            wrapping to backend-specific tensors and pushing to correct devices
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        predictor_cls : type
            the class implementing the actual prediction routine
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            all predictions obtained by feeding the ``test_data`` through the
            ``network``
        dict
            all metrics calculated upon the ``test_data`` and the obtained
            predictions
        """

        if metrics is None:
            metrics = {}

        if metric_keys is None:
            metric_keys = {}

        if callbacks is None:
            callbacks = []

        predictor = self._setup_test(model=model,
                                     prepare_batch_fn=prepare_batch,
                                     callbacks=callbacks,
                                     predictor_cls=predictor_cls, **kwargs)

        # return first item of generator
        return next(predictor.predict_data_mgr_cache_all(test_data, 1, metrics,
                                                         metric_keys,
                                                         self.verbose))

    def kfold(self,
              data: BaseDataManager,
              params,
              optim_builder,
              gpu_ids=None,
              checkpoint_freq=1,
              reduce_mode='mean',
              val_score_key=None,
              val_score_mode="lowest",
              val_freq=1,
              callbacks=None,
              num_splits=None,
              shuffle=False,
              random_seed=None,
              split_type="random",
              val_split=0.2,
              label_key="label",
              train_kwargs: dict = None,
              test_kwargs: dict = None,
              prepare_batch=lambda x: x,
              ):
        """
        Performs a k-fold cross validation

        Parameters
        ----------
        data : :class:`BaseDataManager`
            the datamanager containing all the data for training, testing
            and (optional) validation
        params : :class:`Parameters`
            the parameters containing the model and training kwargs
        optim_builder : function
            the function creating suitable optimizers and returns them as dict
        gpu_ids : list
            a list of integers representing the GPUs to use;
            if empty or None: No gpus will be used at all
        checkpoint_freq : int
            determines how often to checkpoint the training
        reduce_mode : str
            determines how to reduce metrics; must be one of
            'mean' | 'sum' | 'first_only'
        val_score_key : str
            specifies which metric to use for best model selection.
        val_score_mode : str
            determines whether a high or a low val_score is best. Must be one
            of 'highest' | 'lowest'
        val_freq : int
            specifies how often to run a validation step
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        num_splits : int or None
            the number of splits to extract from ``data``.
            If None: uses a default of 10
        shuffle : bool
            whether to shuffle the data before splitting or not (implemented by
            index-shuffling rather than actual data-shuffling to retain
            potentially lazy-behavior of datasets)
        random_seed : None
            seed to seed numpy, the splitting functions and the used
            backend-framework
        split_type : str
            must be one of ['random', 'stratified']
            if 'random': uses random data splitting
            if 'stratified': uses stratified data splitting. Stratification
            will be based on ``label_key``
        val_split : float or None
            the fraction of the train data to use as validation set. If None:
            No validation will be done during training; only testing for each
            fold after the training is complete
        label_key : str
            the label to use for stratification. Will be ignored unless
            ``split_type`` is 'stratified'. Default: 'label'
        train_kwargs : dict or None
            kwargs to update the behavior of the :class:`BaseDataManager`
            containing the train data. If None: empty dict will be passed
        test_kwargs : dict or None
            kwargs to update the behavior of the :class:`BaseDataManager`
            containing the test and validation data.
            If None: empty dict will be passed
        prepare_batch : function
            function to convert a batch-dict to a format accepted by the model.
            This conversion typically includes dtype-conversion, reshaping,
            wrapping to backend-specific tensors and pushing to correct devices
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            all predictions from all folds
        dict
            all metric values from all folds

        Raises
        ------
        ValueError
            if ``split_type`` is neither 'random', nor 'stratified'

        See Also
        --------

        * :class:`sklearn.model_selection.KFold`
        and :class:`sklearn.model_selection.ShuffleSplit`
        for random data-splitting

        * :class:`sklearn.model_selection.StratifiedKFold`
        and :class:`sklearn.model_selection.StratifiedShuffleSplit`
        for stratified data-splitting

        * :meth:`BaseDataManager.update_from_state_dict` for updating the
        data managers by kwargs

        * :meth:`BaseExperiment.run` for the training

        * :meth:`BaseExperiment.test` for the testing

        Notes
        -----
        using stratified splits may be slow during split-calculation, since
        each item must be loaded once to obtain the labels necessary for
        stratification.

        """

        # set number of splits if not specified
        if num_splits is None:
            num_splits = 10
            logger.warning("num_splits not defined, using default value of \
                                    10 splits instead ")

        metrics = params.nested_get("val_metrics", {})
        metric_keys = params.nested_get("metric_keys", {})

        metrics_test, outputs = {}, {}
        split_idxs = list(range(len(data.dataset)))

        if train_kwargs is None:
            train_kwargs = {}
        if test_kwargs is None:
            test_kwargs = {}

        # switch between differnt kfold types
        if split_type == "random":
            split_cls = KFold
            val_split_cls = ShuffleSplit
            # split_labels are ignored for random splitting, set them to
            # split_idxs just ensures same length
            split_labels = split_idxs
        elif split_type == "stratified":
            split_cls = StratifiedKFold
            val_split_cls = StratifiedShuffleSplit
            # iterate over dataset to get labels for stratified splitting
            split_labels = [data.dataset[_idx][label_key]
                            for _idx in split_idxs]
        else:
            raise ValueError("split_type must be one of "
                             "['random', 'stratified'], but got: %s"
                             % str(split_type))

        fold = split_cls(n_splits=num_splits, shuffle=shuffle,
                         random_state=random_seed)

        if random_seed is not None:
            np.random.seed(random_seed)

        # iterate over folds
        for idx, (train_idxs, test_idxs) in enumerate(
                fold.split(split_idxs, split_labels)):

            # extract data from single manager
            train_data = data.get_subset(train_idxs)
            test_data = data.get_subset(test_idxs)

            train_data.update_state_from_dict(copy.deepcopy(train_kwargs))
            test_data.update_state_from_dict(copy.deepcopy(test_kwargs))

            val_data = None
            if val_split is not None:
                if split_type == "random":
                    # split_labels are ignored for random splitting, set them
                    # to split_idxs just ensures same length
                    train_labels = train_idxs
                elif split_type == "stratified":
                    # iterate over dataset to get labels for stratified
                    # splitting
                    train_labels = [train_data.dataset[_idx][label_key]
                                    for _idx in train_idxs]
                else:
                    raise ValueError("split_type must be one of "
                                     "['random', 'stratified'], but got: %s"
                                     % str(split_type))

                _val_split = val_split_cls(n_splits=1, test_size=val_split,
                                           random_state=random_seed)

                for _train_idxs, _val_idxs in _val_split.split(train_idxs,
                                                               train_labels):
                    val_data = train_data.get_subset(_val_idxs)
                    val_data.update_state_from_dict(copy.deepcopy(test_kwargs))

                    train_data = train_data.get_subset(_train_idxs)

            model = self.run(params=params, train_data=train_data,
                             val_data=val_data, optim_builder=optim_builder,
                             gpu_ids=gpu_ids, checkpoint_freq=checkpoint_freq,
                             reduce_mode=reduce_mode,
                             val_score_key=val_score_key,
                             val_score_mode=val_score_mode, val_freq=val_freq,
                             callbacks=callbacks, trainer_cls=trainer_cls,
                             fold=idx, **kwargs)

            _outputs, _metrics_test = self.test(model=model, test_data=test_data,
                                                prepare_batch=prepare_batch,
                                                callbacks=callbacks,
                                                predictor_cls=predictor_cls,
                                                metrics=metrics,
                                                metric_keys=metric_keys,
                                                **kwargs)

            outputs[str(idx)] = _outputs
            metrics_test[str(idx)] = _metrics_test

        return outputs, metrics_test

    def _setup_training(self, params, gpu_ids, optim_builder, callbacks,
                        checkpoint_freq, val_freq, trainer_cls,
                        save_path=None, **kwargs):
        """
        Function to prepare the actual training

        Parameters
        ----------
        params : :class:`Parameters`
            the parameters containing the model and training kwargs
        gpu_ids : list
            a list of integers representing the GPUs to use;
            if empty or None: No gpus will be used at all
        optim_builder : function
            the function creating suitable optimizers and returns them as dict
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        checkpoint_freq : int
            determines how often to checkpoint the training
        val_freq : int
            specifies how often to run a validation step
        trainer_cls : type
            the class implementing the actual training routine
        save_path : str
            the path where to save the results. If None: the experiments
            attribute :attr:`self.save_path` will be used
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            the trained model

        """
        model_params = params.permute_training_on_top().model

        model_kwargs = {**model_params.fixed, **model_params.variable}

        model = self.model_cls(**model_kwargs)

        training_params = params.permute_training_on_top().training
        losses = training_params.nested_get("losses")
        optimizer_cls = training_params.nested_get("optimizer_cls")
        optimizer_params = training_params.nested_get("optimizer_params")
        train_metrics = training_params.nested_get("train_metrics", {})
        lr_scheduler_cls = training_params.nested_get("lr_sched_cls", None)
        lr_scheduler_params = training_params.nested_get("lr_sched_params",
                                                         {})
        val_metrics = training_params.nested_get("val_metrics", {})

        # necessary for resuming training from a given path
        if save_path is None:
            save_path = os.path.join(
                self.save_path,
                "checkpoints",
                "run_%02d" % self._run)

        if callbacks is None:
            callbacks = []

        if gpu_ids is None:
            gpu_ids = []

        fold = kwargs.get("fold", self._run)

        return trainer_cls(
            network=model,
            save_path=save_path,
            losses=losses,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_params=lr_scheduler_params,
            gpu_ids=gpu_ids,
            save_freq=checkpoint_freq,
            optim_fn=optim_builder,
            key_mapping=self.key_mapping,
            logging_type=self._logging_type,
            logging_kwargs=self._logging_kwargs,
            fold=fold,
            callbacks=callbacks,
            start_epoch=1,
            metric_keys=None,
            convert_batch_to_npy_fn=self._convert_to_npy,
            val_freq=val_freq,
            **kwargs
        )

    def _setup_test(self, model, prepare_batch, callbacks, predictor_cls,
                    metrics, metric_keys, **kwargs):
        """

        Parameters
        ----------
        model : :class:`AbstractNetwork`
            the model to test
        prepare_batch_fn : function
            function to convert a batch-dict to a format accepted by the model.
            This conversion typically includes dtype-conversion, reshaping,
            wrapping to backend-specific tensors and pushing to correct devices
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        predictor_cls : type
            class implementing the actual prediction routine
        metrics: dict
            dictionary containing all metrics to evaluate
        metric_keys : dict
            dictionary containing all keys for calculating each metric
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`Predictor`
            the created predictor

        """
        predictor = predictor_cls(
            model=model, key_mapping=self.key_mapping,
            convert_batch_to_npy_fn=self._convert_to_npy,
            prepare_batch_fn=prepare_batch,
            verbose=self.verbose, **kwargs

        )

        return predictor

    def __str__(self):
        """
        Converts :class:`BaseExperiment` to string representation

        Returns
        -------
        str
            representation of class

        """
        s = "Experiment:\n"
        for k, v in vars(self).items():
            s += "\t{} = {}\n".format(k, v)
        return s

    def __call__(self, *args, **kwargs):
        """
        Call :meth:`BaseExperiment.run`

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Returns
        -------
        :class:`BaseNetworkTrainer`
            trainer of trained network

        """
        return self.run(*args, **kwargs)

    def save(self):
        """
        Saves the Whole experiments

        """
        with open(os.path.join(self.save_path, "experiment.delira.pkl"),
                  "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        """
        Loads whole experiment

        Parameters
        ----------
        file_name : str
            file_name to load the experiment from

        """
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
