import os
import logging
from datetime import datetime

def setup_logger(path='./', lvl=20, fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"):
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

def now():
    return datetime.now()