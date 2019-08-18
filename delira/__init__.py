from delira._version import get_versions as _get_versions
from delira._backends import _EXPERIMENT_CLS as Experiment
from delira._backends import _MODEL_CLS as BaseNetwork
from delira._backends import _TRAINER_CLS as NetworkTrainer
from delira._backends import get_backends, seed_all, set_default_backend, \
    get_default_backend
from delira._debug_mode import get_current_debug_mode, switch_debug_mode, \
    set_debug_mode
import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)


__version__ = _get_versions()['version']
del _get_versions
