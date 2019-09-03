from batchgenerators.transforms import AbstractTransform, Compose

import logging
from delira import get_current_debug_mode
import numba
from typing import ClassVar, List, Iterable, Union

logger = logging.getLogger(__name__)


class NumbaTransformWrapper(AbstractTransform):
    def __init__(self, transform: AbstractTransform, nopython: bool = True,
                 target: str = "cpu", parallel: bool = False, **options):

        if get_current_debug_mode():
            # set options for debug mode
            logging.debug("Debug mode detected. Overwriting numba options "
                          "nopython to False and target to cpu")
            nopython = False
            target = "cpu"

        transform.__call__ = numba.jit(transform.__call__, nopython=nopython,
                                       target=target,
                                       parallel=parallel, **options)
        self._transform = transform

    def __call__(self, **kwargs):
        return self._transform(**kwargs)


class NumbaTransform(NumbaTransformWrapper):
    def __init__(self, transform_cls: ClassVar[AbstractTransform],
                 nopython: bool = True, target: str = "cpu",
                 parallel: bool = False, **kwargs):
        trafo = transform_cls(**kwargs)

        super().__init__(trafo, nopython=nopython, target=target,
                         parallel=parallel)


class NumbaCompose(Compose):
    def __init__(self, transforms: Union[List, Iterable]):
        super().__init__(transforms=[NumbaTransformWrapper(trafo)
                                     for trafo in transforms])
