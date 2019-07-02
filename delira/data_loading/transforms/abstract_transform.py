from numba import jit
from delira import get_current_debug_mode
import numpy as np
from abc import abstractmethod
from delira.data_loading.transforms.compose import Compose


class AbstractTransform(object):
    @abstractmethod
    def __call__(self, **kwargs) -> dict:
        raise NotImplementedError

    def __add__(self, other):
        """
        Overloads the addition operator to combine transformations

        Parameters
        ----------
        other
            the other affine to apply

        Returns
        -------
        :class:`delira.data_loading.transforms.Compose`
            the combined transform

        """

        return Compose(self, other)

    def __radd__(self, other):
        return self.__add__(other)


class BaseTransform(AbstractTransform):
    def __init__(self, source_keys, destination_keys, concatenate=True):

        if not isinstance(source_keys, (tuple, list)):
            source_keys = (source_keys,)

        if not isinstance(destination_keys, (tuple, list)):
            destination_keys = (destination_keys, )

        assert len(source_keys) == len(destination_keys)

        self._concatenate = concatenate
        self._source_keys = source_keys
        self._destination_keys = destination_keys

    @jit(nopython=get_current_debug_mode(),
         parallel=get_current_debug_mode())
    def __call__(self, **data_dict) -> dict:
        for src_key, dst_key in zip(self._source_keys,
                                    self._destination_keys):
            data_dict[dst_key] = self._apply_batch_trafo(data_dict[src_key])

        return data_dict

    def _apply_batch_trafo(self, batch: np.ndarray):
        transformed_batch = []
        for idx, _sample in enumerate(batch):
            transformed_batch.append(self._apply_sample_trafo(_sample))

        if self._concatenate:
            transformed_batch = np.concatenate(transformed_batch)

        return transformed_batch

    def _apply_sample_trafo(self, sample: np.ndarray):
        raise NotImplementedError

