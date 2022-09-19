import sys
import logging

depth = 0
INDENT = "  "
logger = logging.getLogger()

log_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def log_init(log_file, level=logging.DEBUG, tee_stdout=True):
    global logger
    if isinstance(level, str):
        level = log_level_map[level.lower()]
    logger = logging.getLogger("MTMC")
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter('%(levelname)-8s [%(asctime)s]: %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if tee_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


    
def inc_depth():
    global depth
    depth += 1


def dec_depth():
    global depth
    depth = max(0, depth - 1)


def debug(msg, *args):
    log_function(logger.debug, msg, *args)


def info(msg, *args):
    log_function(logger.info, msg, *args)


def warning(msg, *args):
    log_function(logger.warning, msg, *args)


def error(msg, *args):
    log_function(logger.error, msg, *args)


def log_function(log_func, msg, *args):
    msg = INDENT * depth + (msg % args)
    log_func(msg)
