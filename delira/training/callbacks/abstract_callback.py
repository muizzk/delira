from delira.training.base_trainer import BaseNetworkTrainer


class AbstractCallback(object):
    """
    Implements abstract callback interface.
    All callbacks should be derived from this class

    See Also
    --------
    :class:`AbstractNetworkTrainer`

    """

    def __init__(self, *args, **kwargs) -> None:
        """

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        """
        pass

    def at_epoch_begin(self, trainer: BaseNetworkTrainer, **kwargs) -> dict:
        """
        Function which will be executed at begin of each epoch

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        """
        return {}

    def at_epoch_end(self, trainer: BaseNetworkTrainer, **kwargs) -> dict:
        """
        Function which will be executed at end of each epoch

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        """
        return {}
