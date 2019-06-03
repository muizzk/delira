from delira import get_backends

if "TORCH" in get_backends():
    from .torch import save_checkpoint as torch_save_checkpoint
    from .torch import load_checkpoint as torch_load_checkpoint

if "TF" in get_backends():
    from .tf import save_checkpoint as tf_save_checkpoint
    from .tf import load_checkpoint as tf_load_checkpoint

if "MX" in get_backends():
    from .mxnet import save_checkpoint as mx_save_checkpoint
    from .mxnet import load_checkpoint as mx_load_checkpoint
