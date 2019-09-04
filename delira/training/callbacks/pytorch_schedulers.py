from delira import get_backends
from delira.training.callbacks.abstract_callback import AbstractCallback
from delira.training.base_trainer import BaseNetworkTrainer
from typing import Union, Iterable, Callable

if 'TORCH' in get_backends():
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau, \
        CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, StepLR

    class DefaultPyTorchSchedulerCallback(AbstractCallback):
        """
        Implements a Callback, which `at_epoch_end` function is suitable for
        most schedulers

        """

        def __init__(self, *args, **kwargs) -> None:
            """

            Parameters
            ----------
            *args :
                Arbitrary Positional Arguments
            **kwargs :
                Arbitrary Keyword Arguments

            """
            super().__init__()

            self.scheduler = None

        def at_epoch_end(self, trainer: BaseNetworkTrainer, **kwargs) -> dict:
            """
            Executes a single scheduling step

            Parameters
            ----------
            trainer : :class:`PyTorchNetworkTrainer`
                the trainer class, which can be changed
            **kwargs :
                additional keyword arguments

            Returns
            -------
            :class:`PyTorchNetworkTrainer`
                modified trainer

            """
            self.scheduler.step(epoch=kwargs.get("curr_epoch", None))
            return {}

    class ReduceLROnPlateauCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `ReduceLROnPlateau` Scheduler as Callback

        """

        def __init__(self, optimizer: Optimizer,
                     mode: str = 'min', factor: float = 0.1,
                     patience: int = 10, verbose: bool = False,
                     threshold: float = 1e-4, threshold_mode: str = 'rel',
                     cooldown: int = 0,
                     min_lr: Union[float, list, Iterable] = 0,
                     eps: float = 1e-8) -> None:
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            mode : str
                One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor : float
                Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience : int
                Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            verbose : bool
                If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            threshold : float
                Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode : string
                One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown : int
                Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr : float or list
                A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps : float
                Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8

            """
            super().__init__()
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode,
                factor,
                patience,
                verbose,
                threshold,
                threshold_mode,
                cooldown,
                min_lr,
                eps)

        def at_epoch_end(self, trainer: BaseNetworkTrainer,
                         **kwargs) -> dict:
            """
            Executes a single scheduling step

            Parameters
            ----------
            trainer : :class:`PyTorchNetworkTrainer`
                the trainer class, which can be changed
            kwargs :
                additional keyword arguments

            Returns
            -------
            :class:`PyTorchNetworkTrainer`
                modified trainer

            """
            val_metrics = kwargs.get("val_metrics", {})

            val_score_key = kwargs.get("val_score_key", None)

            metrics = val_metrics.get(val_score_key)

            self.scheduler.step(metrics=metrics)

            return {}

    class CosineAnnealingLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `CosineAnnealingLR` Scheduler as callback

        """

        def __init__(self, optimizer: Optimizer, T_max: int,
                     eta_min: float = 0, last_epoch: int = -1) -> None:
            """

            Parameters
            ----------
            optimizer : optimizer
                Wrapped optimizer.
            T_max : int
                Maximum number of iterations.
            eta_min : float
                Minimum learning rate. Default: 0.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = CosineAnnealingLR(optimizer, T_max, eta_min,
                                               last_epoch)

    class ExponentialLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `ExponentialLR` Scheduler as callback

        """

        def __init__(self, optimizer: Optimizer, gamma: float,
                     last_epoch: int = -1) -> None:
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            gamma : float
                Multiplicative factor of learning rate decay.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = ExponentialLR(optimizer, gamma, last_epoch)

    class LambdaLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `LambdaLR` Scheduler as callback

        """

        def __init__(self, optimizer: Optimizer,
                     lr_lambda: Union[Callable, list, tuple, Iterable],
                     last_epoch: int = -1) -> None:
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            lr_lambda : function or list
                A function which computes a multiplicative
                factor given an integer parameter epoch, or a list of such
                functions, one for each group in optimizer.param_groups.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)

    class MultiStepLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `MultiStepLR` Scheduler as callback

        """

        def __init__(self, optimizer: Optimizer,
                     milestones: Union[list, tuple, Iterable],
                     gamma: float = 0.1, last_epoch: int = -1) -> None:
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            milestones : list
                List of epoch indices. Must be increasing.
            gamma : float
                Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = MultiStepLR(
                optimizer, milestones, gamma, last_epoch)

    class StepLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `StepLR` Scheduler as callback

        """

        def __init__(self, optimizer: Optimizer, step_size: int,
                     gamma: float = 0.1, last_epoch: int = -1) -> None:
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            step_size : int
                Period of learning rate decay.
            gamma :float
                Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch : int
                The index of last epoch. Default: -1

            """
            super().__init__()

            self.scheduler = StepLR(optimizer, step_size, gamma, last_epoch)
