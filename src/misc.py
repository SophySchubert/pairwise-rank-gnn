import json
import os
import logging
from datetime import datetime
from shutil import copyfile

from models.prgnn import PRGNN
from models.general_gnn import GeneralGNN


def setup_logger(path="./", lvl=20, fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"):
    """
    Sets up a global logger accessible via logging.getLogger("root").
    The registered logger will stream its outputs to the console as well as
    to a file out.log in the specified directory.
    Parameters
    ----------
    path : string
        Path to the folder in which to save the logfile out.log.
    lvl : int
        One of CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0.
    fmt : string
        Format string representing the format of the logs.
    Returns
    -------
    root_logger : Logger
        The root logger.
    """
    log_path = os.path.join(path, "out.log")
    formatter = logging.Formatter(fmt=fmt)
    root_logger = logging.getLogger()
    root_logger.setLevel(lvl)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger

def setup_model(model_name, n_labels, config=None):
    if model_name == 'general_gnn':
        return GeneralGNN(n_labels, activation="softmax")
    elif model_name == 'prgnn':
        return PRGNN(config=config)
    else:
        raise ValueError(f"Model {model_name} unknown")

def now():
    return datetime.now()

def read_config(path: str):
    with open(path, "r") as config:
        return json.load(config)

def setup_experiment(path: str):
    config = read_config(path)
    experiment_path = os.path.join(".", config['folder_path'], now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(experiment_path)
    config['folder_path'] = experiment_path
    copyfile(path, experiment_path+"/config.json")

    return config