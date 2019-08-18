from delira import get_backends as _get_backends


if "CHAINER" in _get_backends():
    from delira.training.backends.chainer import *

if "SKLEARN" in _get_backends():
    from delira.training.backends.sklearn import *

if "TFEAGER" in _get_backends():
    from delira.training.backends.tf_graph import *

if "TFGRAPH" in _get_backends():
    from delira.training.backends.tf_eager import *

if "TORCH" in _get_backends():
    from delira.training.backends.torch import *

if "TORCHSCRIPT" in _get_backends():
    from delira.training.backends.torchscript import *
