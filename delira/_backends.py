import os
import json
from delira._version import get_versions as _get_versions

# to register new possible backends, they have to be added to this list.
# each backend should consist of a tuple of length 2 with the first entry
# being the package import name and the second being the backend abbreviation.
# E.g. TensorFlow's package is named 'tensorflow' but if the package is found,
# it will be considered as 'tf' later on

__POSSIBLE_BACKENDS = []

__BACKENDS = ()
__BACKEND_CLASSES = {}
__DEFAULT_BACKEND = None


class NoDefaultWrapper(object):
    def __init__(*args, **kwargs):
        raise RuntimeError("No default backend specified so far")


_TRAINER_CLS = NoDefaultWrapper
_MODEL_CLS = NoDefaultWrapper
_EXPERIMENT_CLS = NoDefaultWrapper


def _check_import(backend_name):
    import importlib
    bcknd = importlib.util.find_spec(backend_name)
    return bcknd is not None


def _check_torch_backend():
    return _check_import("torch")


def _check_torchscript_backend():
    importable = _check_import("torch")
    if importable:
        try:
            import torch
            from packaging import version
            return version.parse(torch.__version__) >= version.parse("1.2.0")
        finally:
            del torch
            del version

    return False


def _check_chainer_backend():
    return _check_import("chainer")


def _check_tf_eager_backend():
    importable = _check_import("tensorflow")
    if importable:
        try:
            import tensorflow
            return hasattr(tensorflow, "enable_eager_execution")

        finally:
            del tensorflow
    return False


def _check_tf_graph_backend():
    importable = _check_import("tensorflow")
    if importable:
        try:
            import tensorflow
            return hasattr(tensorflow, "disable_eager_execution")

        finally:
            del tensorflow
    return False


def _check_tf_backend():
    return _check_import("tensorflow")


def _check_sklearn_backend():
    return _check_import("sklearn")


def _register_possible_backend(backend_name, check_fn, classes):
    __POSSIBLE_BACKENDS.append((backend_name, check_fn))
    __BACKEND_CLASSES[backend_name.upper()] = classes


_register_possible_backend(
    "torch", _check_torch_backend,
    (
        "delira.training.backends.PyTorchNetworkTrainer",
        "delira.training.backends.PyTorchExperiment",
        "delira.models.backends.AbstractPyTorchNetwork"
    )
)
_register_possible_backend(
    "torchscript", _check_torchscript_backend,
    ("delira.training.backends.TorchScriptNetworkTrainer",
     "delira.training.backends.TorchScriptExperiment",
     "delira.models.backends.AbstractPyTorchNetwork"
     )
)
_register_possible_backend(
    "tfeager", _check_tf_eager_backend,
    (
        "delira.training.backends.TfEagerNetworkTrainer",
        "delira.training.backends.TfEagerExperiment",
        "delira.models.backends.AbstractTfEagerNetwork"
    )
)
_register_possible_backend(
    "tf_graph", _check_tf_graph_backend,
    (
        "delira.training.backends.TfGraphNetworkTrainer",
        "delira.training.backends.TfGraphExperiment",
        "delira.models.backends.AbstractTfGraphNetwork"
    )
)
_register_possible_backend(
    "chainer", _check_chainer_backend,
    (
        "delira.training.backends.ChainerNetworkTrainer",
        "delira.training.backends.ChainerExperiment",
        "delira.models.backends.AbstractChainerNetwork"
    )
)
_register_possible_backend(
    "sklearn", _check_sklearn_backend,
    (
        "delira.training.backends.SklearnEstimatorTrainer",
        "delira.training.backends.SklearnExperiment",
        "delira.models.backends.SklearnEstimator"
    )
)


def _determine_backends():
    """
    Internal Helper Function to determine the currently valid backends by
    trying to import them. The valid backends are not returned, but appended
    to the global ``__BACKENDS`` variable

    """

    _config_file = __file__.replace("_backends.py", ".delira")
    # look for config file to determine backend
    # if file exists: load config into environment variables

    if not os.path.isfile(_config_file):
        _backends = {}
        # try to import all possible backends to determine valid backends

        for curr_backend in __POSSIBLE_BACKENDS:
            try:
                assert len(curr_backend) == 2

                _backend_name, _check_fn = curr_backend

                # check for current backend
                if _check_fn():
                    _backends[curr_backend[1]] = True
                else:
                    _backends[curr_backend[1]] = False

            except ValueError:
                _backends[curr_backend[1]] = False

        with open(_config_file, "w") as f:
            json.dump({"version": _get_versions()['version'],
                       "backend": _backends},
                      f, sort_keys=True, indent=4)

        del _backends

    # set values from config file to variable and empty Backend-List before
    global __BACKENDS
    __BACKENDS = []
    with open(_config_file) as f:
        _config_dict = json.load(f)
    for key, val in _config_dict.pop("backend").items():
        if val:
            __BACKENDS.append(key.upper())
    del _config_dict

    del _config_file

    # make __BACKENDS non mutable
    __BACKENDS = tuple(__BACKENDS)


def get_backends():
    """
    Return List of currently available backends

    Returns
    -------
    list
        list of strings containing the currently installed backends
    """
    global __BACKENDS

    if not __BACKENDS:
        _determine_backends()
    return __BACKENDS


def seed_all(seed):
    """
    Helper Function to seed all available backends

    Parameters
    ----------
    seed : int
        the new random seed

    """
    import sys

    import numpy as np
    np.random.seed(seed)

    import random
    random.seed = seed

    if "torch" in sys.modules and ("TORCH" in get_backends()
                                   or "TORCHSCRIPT" in get_backends()):
        import torch
        torch.random.manual_seed(seed)

    elif "tensorflow" in sys.modules and ("TFEAGER" in get_backends()
                                          or "TFGRAPH" in get_backends()):
        import tensorflow as tf
        tf.random.set_random_seed(seed)

    elif "chainer" in sys.modules and "CHAINER" in get_backends():
        try:
            import cupy
            cupy.random.seed(seed)
        except ImportError:
            pass


def set_default_backend(backend: str):
    backend = backend.upper()
    assert backend in get_backends()

    global __DEFAULT_BACKEND
    __DEFAULT_BACKEND = backend

    global _TRAINER_CLS
    global _MODEL_CLS
    global _EXPERIMENT_CLS

    import importlib

    _TRAINER_CLS, _MODEL_CLS, _EXPERIMENT_CLS = (
        importlib.import_module(_cls) for _cls in __BACKEND_CLASSES[backend])


def get_default_backend():
    return __DEFAULT_BACKEND

