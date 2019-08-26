from functools import partial

import tensorflow as tf

from delira.data_loading import BaseDataManager
from delira.models.backends.tf_eager import AbstractTfEagerNetwork

from delira.training.base_experiment import BaseExperiment
from delira.training.parameters import Parameters

from delira.training.backends.tf_eager.trainer import TfEagerNetworkTrainer
from delira.training.backends.tf_eager.utils import create_optims_default
from delira.training.backends.tf_eager.utils import convert_to_numpy

from delira.training.predictor import Predictor


class TfEagerExperiment(BaseExperiment):
    """
    Class for running TF Eager Experiments.

    See Also
    --------
    :class:`BaseExperiment`
    """

    def __init__(self,
                 model_cls: AbstractTfEagerNetwork,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 verbose=True,
                 logging_type="tensorboard",
                 logging_kwargs=None,
                 convert_to_npy=convert_to_numpy
                 ):
        """

        Parameters
        ----------
        model_cls : Subclass of :class:`AbstractTfEagerNetwork`
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

        if key_mapping is None:
            key_mapping = {"x": "data"}

        super().__init__(
            model_cls=model_cls,
            name=name,
            save_path=save_path,
            key_mapping=key_mapping,
            verbose=verbose,
            logging_type=logging_type,
            logging_kwargs=logging_kwargs,
            convert_to_npy=convert_to_npy
        )

    def run(self, params, train_data: BaseDataManager,
            val_data: BaseDataManager, optim_builder=create_optims_default,
            gpu_ids=None, checkpoint_freq=1, reduce_mode='mean',
            val_score_key=None, val_score_mode="lowest", val_freq=1,
            callbacks=None, trainer_cls=TfEagerNetworkTrainer, **kwargs):
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
            additional keyword arguments given to
            :param:`trainer_cls` during initialization

        Returns
        -------
        :class:`AbstractTfEagerNetwork`
            the trained model

        """
        tf.reset_default_graph()

        return super().run(
            params=params, train_data=train_data, val_data=val_data,
            optim_builder=optim_builder, gpu_ids=gpu_ids,
            checkpoint_freq=checkpoint_freq, reduce_mode=reduce_mode,
            val_score_key=val_score_key, val_score_mode=val_score_mode,
            val_freq=val_freq, callbacks=callbacks,
            trainer_cls=trainer_cls, **kwargs)

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
        prepare_batch : function
            function to convert a batch-dict to a format accepted by the model.
            This conversion typically includes dtype-conversion, reshaping,
            wrapping to backend-specific tensors and pushing to correct devices
        callbacks : list
            list of callbacks to use during training. Each callback should be
            derived from :class:`delira.training.callbacks.AbstractCallback`
        predictor_cls : type
            the class implementing the actual prediction routine
        metrics : dict
            the metrics to calculate
        metric_keys : dict of tuples
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label`` will be used for metric calculation
        **kwargs :
            additional keyword arguments, which are given to the
            :param:`predictor_cls` during initialization

        Returns
        -------
        dict
            all predictions obtained by feeding the ``test_data`` through the
            ``network``
        dict
            all metrics calculated upon the ``test_data`` and the obtained
            predictions
        """

        if prepare_batch is None:
            prepare_batch = partial(
                model.prepare_batch,
                input_device="/cpu:0",
                output_device="/cpu:0")

        return super().test(model=model, test_data=test_data,
                            prepare_batch=prepare_batch, callbacks=callbacks,
                            predictor_cls=predictor_cls, metrics=metrics,
                            metric_keys=metric_keys, **kwargs)

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

        return super().kfold(data=data, params=params,
                             optim_builder=optim_builder,
                             gpu_ids=gpu_ids,
                             checkpoint_freq=checkpoint_freq,
                             reduce_mode=reduce_mode,
                             val_score_key=val_score_key,
                             val_score_mode=val_score_mode,
                             val_freq=val_freq,
                             callbacks=callbacks,
                             num_splits=num_splits,
                             shuffle=shuffle,
                             random_seed=random_seed,
                             split_type=split_type,
                             val_split=val_split,
                             label_key=label_key,
                             train_kwargs=train_kwargs,
                             test_kwargs=test_kwargs,
                             prepare_batch=prepare_batch)
