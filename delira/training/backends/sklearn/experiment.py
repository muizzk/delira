from functools import partial

from sklearn.base import BaseEstimator

from delira.data_loading import BaseDataManager

from delira.models.backends.sklearn import SklearnEstimator

from delira.training.base_experiment import BaseExperiment
from delira.training.parameters import Parameters
from delira.training.predictor import Predictor
from delira.training.utils import convert_to_numpy_identity
from delira.training.backends.sklearn.utils import create_optims_default

from delira.training.backends.sklearn.trainer import SklearnEstimatorTrainer


class SklearnExperiment(BaseExperiment):
    """
    Class for running Sklearn Experiments.

    See Also
    --------
    :class:`BaseExperiment`
    """

    def __init__(self,
                 model_cls: BaseEstimator,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 verbose=True,
                 logging_type="tensorboard",
                 logging_kwargs=None,
                 convert_to_npy=convert_to_numpy_identity,
                 model_wrapper_cls=SklearnEstimator,
                 ):
        """

        Parameters
        ----------
        model_cls : Subclass of :class:`BaseEstimator`
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
            key_mapping = {"X": "X"}
            model_cls = partial(model_wrapper_cls, module_cls=model_cls)

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
            callbacks=None, trainer_cls=SklearnEstimatorTrainer, **kwargs):
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
        :class:`SklearnEstimator`
            the trained model

        """

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
            additional keyword arguments, which are going to the :param`predictor_cls`
            during initialization

        Returns
        -------
        dict
            all predictions obtained by feeding the ``test_data`` through the
            ``network``
        dict
            all metrics calculated upon the ``test_data`` and the obtained
            predictions
        """

        if not isinstance(model, SklearnEstimator):
            model = self.model_cls(model)

        if prepare_batch is None:
            prepare_batch = partial(model.prepare_batch,
                                    input_device="cpu",
                                    output_device="cpu")

        return super().test(model=model, test_data=test_data,
                            prepare_batch=prepare_batch, callbacks=callbacks,
                            predictor_cls=predictor_cls, metrics=metrics,
                            metric_keys=metric_keys, **kwargs)