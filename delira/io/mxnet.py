import mxnet
from ..models import AbstractMXNetwork
from ..utils.context_managers import TemporaryUnzip
import json
import os
import zipfile
import logging


def save_checkpoint(file: str, model: AbstractMXNetwork = None,
                    optimizers: dict = None, epoch=None):
    """
    Saves a checkpoint into a single zip-Archive (and creates temporary
    intermediate files)

    Parameters
    ----------
    file : str
        the file to save the final zip-archive to
    model : :class:`AbstractMXNetwork`
        the model to save; will only be saved if not None; default: None
    optimizers : dict
        a dictionary containing all optimizers; will only be saved if not None;
        default: None
    epoch : int
        the current epoch to save; will only be saved if not None;

    """

    file = os.path.abspath(file)

    # config_dict to store all files with the corresponding keys
    config_dict = {}
    # all files to save into zipfile and delete afterwards
    files = []

    # save model to file, add this file to config and to used files
    if model is not None:
        model_file = os.path.join(os.path.dirname(file), "model.mx")
        model.save_parameters(model_file)
        config_dict["model"] = os.path.basename(model_file)
        files.append((model_file, os.path.basename(model_file)))

    # save all optimizers to files, add all files to config and used files
    if optimizers is not None:
        config_dict["optimizers"] = {}
        for key, val in optimizers.items():
            optim_file = os.path.join(os.path.dirname(file),
                                      "optim_%s_state.mx" % key)
            if isinstance(val, mxnet.gluon.Trainer):
                val.save_states(optim_file)
                config_dict["optimizers"][key] = os.path.basename(optim_file)
                files.append((optim_file,
                              os.path.basename(optim_file))
                             )

    if epoch is not None:
        config_dict["epoch"] = epoch

    # create config file from config dict and append it to used files
    config_file = os.path.join(os.path.dirname(file), "config.mx")
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    files.append((config_file,
                  os.path.basename(config_file)))

    # create zipfile of used files
    with zipfile.ZipFile(file, mode="w") as f:
        for _file in files:
            f.write(*_file)

    # delete original files after zipping them
    for _file in files:
        os.remove(_file[0])


def load_checkpoint(file, model: AbstractMXNetwork = None,
                    optimizers: dict = None):
    """
    Loads a single checkpoint from a given zip-archive

    Parameters
    ----------
    file : str
        the path to the zip-archive containing the checkpoint
    model : :class:`AbstractMXNetwork`
        the model to load the state to
    optimizers : dict
        a dictionary of optimizers, the optimizer states should be loaded to

    Returns
    -------
    dict
        the loaded state

    """

    new_state = {}

    # open zipfile containing files for each state item
    with zipfile.ZipFile(file, mode="r") as f:
        config_dict = json.load(f.read("config.mx"))

        # load model (temporarily extract model file to cwd)
        if "model" in config_dict and model is not None:
            with TemporaryUnzip(f, config_dict["model"], os.getcwd()):
                model.load_parameters(os.path.join(os.getcwd(), "model.mx"))
            new_state["model"] = model
        elif "model" in config_dict and model is None:
            logging.warning("Cannot load model state saved in checkpoint, "
                            "because no model was passed")
        elif "model" not in config_dict and model is not None:
            logging.warning("Cannot load model state to passed model, because "
                            "checkpoint did not contain any model state")

        # load all optimizer states (temporarily extract each optimizer
        # file to cwd)
        if optimizers is not None:
            optim_config = config_dict.get("optimizers", {})
            for key, val in optimizers.items():
                if isinstance(val, mxnet.gluon.Trainer) and key in optim_config:
                    with TemporaryUnzip(f, optim_config[key],
                                        os.getcwd()):
                        val.load_states(os.path.join(os.getcwd(),
                                                     optim_config[key]))

                # key not in optim_config:
                elif isinstance(val, mxnet.gluon.Trainer):
                    logging.warning("could not load state for optimizer %s, "
                                    "because no key for this optimizer was "
                                    "saved in current checkpoint" % key)

                # val is no instance of mxnet.gluon.Trainer
                elif key in optim_config:
                    logging.warning("could not load state for optimizer %s, "
                                    "because the passed optimizer is no "
                                    "instance of mxnet.gluon.Trainer" % key)

            new_state["optimizers"] = optimizers

        # load epoch
        if "epoch" in config_dict:
            new_state["epoch"] = config_dict["epoch"]

    return new_state
