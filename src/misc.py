import os
import logging
from yaml import Loader, load
from datetime import datetime
from shutil import copyfile

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

def _read_config(path: str):
    with open(path, "r") as config:
        return load(config, Loader=Loader)

def setup_experiment(path: str):
    config = _read_config(path)
    experiment_path = os.path.join(".", config['folder_path'], datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(experiment_path)
    config['folder_path'] = experiment_path
    copyfile(path, experiment_path+"/config.yml")
    return config

def config_add_nagsl(config):
    # Add NAGSL specific configurations from the original code
    config['channel_ffn_size'] = 128
    config['n_channel_transformer_heads'] = 4 #most datasets, IMDBMulti was 8
    config['msa_bias'] = True
    config['dropout'] = 0.1
    config['n_heads'] = 8
    config['encoder_ffn_size'] = 128
    config['embedding_size'] = 128 # 32 for IMDB, 128 for AIDS dataset
    config['interaction_mask'] = False
    config['encoder_mask'] = False
    config['align_mask'] = False
    config['cnn_mask'] = True
    config['encoder_ffn_size'] = 128
    config['GNN'] = 'GCN'
    config['GT_res'] = True
    config['share_qk'] = True
    config['use_dist'] = False #changed by me because my graph datasets have no dist for nodes
    config['dist_decay'] = 0
    config['dist_start_decay'] = 0.5
    config['conv_channels_0'] = 32
    config['conv_channels_1'] = 64
    config['conv_channels_2'] = 128
    config['conv_channels_3'] = 256
    config['conv_l_relu_slope'] = 0.33
    config['conv_dropout'] = 0.1
    config['channel_align'] = True
    config['sim_mat_learning_ablation'] = False # set to True to use SimMatPooling
    return config