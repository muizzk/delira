import typing
from functools import partial

import tensorflow as tf

from delira.data_loading import BaseDataManager
from delira.models.backends.tf_eager import AbstractTfEagerNetwork

from delira.training.base_experiment import BaseExperiment
from delira.training.parameters import Parameters

from delira.training.backends.tf_eager.trainer import TfEagerNetworkTrainer
from delira.training.backends.tf_eager.utils import create_optims_default
from delira.training.backends.tf_eager.utils import convert_to_numpy


class TfEagerExperiment(BaseExperiment):
    def __init__(self,
                 params: typing.Union[str, Parameters],
                 model_cls: AbstractTfEagerNetwork,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 optim_builder=create_optims_default,
                 checkpoint_freq=1,
                 trainer_cls=TfEagerNetworkTrainer,
                 **kwargs):
        """

        Parameters
        ----------
        params : :class:`Parameters` or str
            the training parameters, if string is passed,
            it is treated as a path to a pickle file, where the
            parameters are loaded from
        model_cls : Subclass of :class:`AbstractTfEagerNetwork`
            the class implementing the model to train
        n_epochs : int or None
            the number of epochs to train, if None: can be specified later
            during actual training
        name : str or None
            the Experiment's name
        save_path : str or None
            the path to save the results and checkpoints to.
            if None: Current working directory will be used
        key_mapping : dict
            mapping between data_dict and model inputs (necessary for
            prediction with :class:`Predictor`-API), if no keymapping is
            given, a default key_mapping of {"x": "data"} will be used
            here
        val_score_key : str or None
            key defining which metric to use for validation (determining
            best model and scheduling lr); if None: No validation-based
            operations will be done (model might still get validated,
            but validation metrics can only be logged and not used further)
        optim_builder : function
            Function returning a dict of backend-specific optimizers.
            defaults to :func:`create_optims_default`
        checkpoint_freq : int
            frequency of saving checkpoints (1 denotes saving every epoch,
            2 denotes saving every second epoch etc.); default: 1
        trainer_cls : subclass of :class:`TfEagerNetworkTrainer`
            the trainer class to use for training the model, defaults to
            :class:`TfNetworkTrainer`
        **kwargs :
            additional keyword arguments

        """

        if key_mapping is None:
            key_mapping = {"x": "data"}
        super().__init__(params=params, model_cls=model_cls,
                         n_epochs=n_epochs, name=name, save_path=save_path,
                         key_mapping=key_mapping,
                         val_score_key=val_score_key,
                         optim_builder=optim_builder,
                         checkpoint_freq=checkpoint_freq,
                         trainer_cls=trainer_cls,
                         **kwargs)

    def kfold(self, data: BaseDataManager, metrics: dict, num_epochs=None,
              num_splits=None, shuffle=False, random_seed=None,
              split_type="random", val_split=0.2, label_key="label",
              train_kwargs: dict = None, test_kwargs: dict = None,
              metric_keys: dict = None, params=None, verbose=False,
              **kwargs):
        """
        Performs a k-Fold cross-validation

        Parameters
        ----------
        data : :class:`BaseDataManager`
            the data to use for training(, validation) and testing. Will be
            split based on ``split_type`` and ``val_split``
        metrics : dict
            dictionary containing the metrics to evaluate during k-fold
        num_epochs : int or None
            number of epochs to train (if not given, will either be
            extracted from ``params``, ``self.parms`` or ``self.n_epochs``)
        num_splits : int or None
            the number of splits to extract from ``data``.
            If None: uses a default of 10
        shuffle : bool
            whether to shuffle the data before splitting or not
            (implemented by index-shuffling rather than actual
            data-shuffling to retain potentially lazy-behavior of datasets)
        random_seed : None
            seed to seed numpy, the splitting functions and the used
            backend-framework
        split_type : str
            must be one of ['random', 'stratified']
            if 'random': uses random data splitting
            if 'stratified': uses stratified data splitting. Stratification
            will be based on ``label_key``
        val_split : float or None
            the fraction of the train data to use as validation set.
            If None: No validation will be done during training; only
            testing for each fold after the training is complete
        label_key : str
            the label to use for stratification. Will be ignored unless
            ``split_type`` is 'stratified'. Default: 'label'
        train_kwargs : dict or None
            kwargs to update the behavior of the :class:`BaseDataManager`
            containing the train data. If None: empty dict will be passed
        metric_keys : dict of tuples
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label`` will be used for metric calculation
        test_kwargs : dict or None
            kwargs to update the behavior of the :class:`BaseDataManager`
            containing the test and validation data.
            If None: empty dict will be passed
        params : :class:`Parameters`or None
            the training and model parameters
            (will be merged with ``self.params``)
        verbose : bool
            verbosity
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

        # seed tf backend
        if random_seed is not None:
            tf.set_random_seed(random_seed)

        return super().kfold(
            data=data,
            metrics=metrics,
            num_epochs=num_epochs,
            num_splits=num_splits,
            shuffle=shuffle,
            random_seed=random_seed,
            split_type=split_type,
            val_split=val_split,
            label_key=label_key,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
            metric_keys=metric_keys,
            params=params,
            verbose=verbose,
            **kwargs)

    def test(self, network, test_data: BaseDataManager,
             metrics: dict, metric_keys=None,
             verbose=False, prepare_batch=lambda x: x,
             convert_fn=None, **kwargs):
        """
        Setup and run testing on a given network

        Parameters
        ----------
        network : :class:`AbstractNetwork`
            the (trained) network to test
        test_data : :class:`BaseDataManager`
            the data to use for testing
        metrics : dict
            the metrics to calculate
        metric_keys : dict of tuples
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label``
             will be used for metric calculation
        verbose : bool
            verbosity of the test process
        prepare_batch : function
            function to convert a batch-dict to a format accepted by the
            model. This conversion typically includes dtype-conversion,
            reshaping, wrapping to backend-specific tensors and
            pushing to correct devices. If not further specified uses the
            ``network``'s ``prepare_batch`` with CPU devices
        convert_fn : function
            function to convert a batch of tensors to numpy
            if not specified defaults to
            :func:`convert_torch_tensor_to_npy`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            all predictions obtained by feeding the ``test_data`` through
            the ``network``
        dict
            all metrics calculated upon the ``test_data`` and the obtained
            predictions

        """
        # specify convert_fn to correct backend function
        if convert_fn is None:
            convert_fn = convert_to_numpy

        if prepare_batch is None:
            prepare_batch = partial(
                network.prepare_batch,
                input_device="/cpu:0",
                output_device="/cpu:0")

        return super().test(network=network, test_data=test_data,
                            metrics=metrics, metric_keys=metric_keys,
                            verbose=verbose, prepare_batch=prepare_batch,
                            convert_fn=convert_fn, **kwargs)

    def setup(self, params, training=True, **kwargs):
        """
        Defines the setup behavior (model, trainer etc.) for training and
        testing case

        Parameters
        ----------
        params : :class:`Parameters`
            the parameters to use for setup
        training : bool
            whether to setup for training case or for testing case
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`BaseNetworkTrainer`
            the created trainer (if ``training=True``)
        :class:`Predictor`
            the created predictor (if ``training=False``)

        See Also
        --------

        * :meth:`BaseExperiment._setup_training` for training setup

        * :meth:`BaseExperiment._setup_test` for test setup

        """
        tf.reset_default_graph()
        return super().setup(params=params, training=training,
                             **kwargs)