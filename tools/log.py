import logging

depth = 0
log_level = logging.DEBUG
INDENT = "  "
inited = False
output_tee_stdout = False

log_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def log_init(log_file, level=logging.DEBUG, tee_stdout=True):
    global inited, output_tee_stdout, log_level
    if isinstance(level, str):
        level = log_level_map[level.lower()]
    if inited:
        return
    inited = True
    output_tee_stdout = tee_stdout
    log_level = level
    logging.basicConfig(filename=log_file, level=level)


def inc_depth():
    global depth
    depth += 1


def dec_depth():
    global depth
    depth = max(0, depth - 1)


def debug(msg, *args):
    log_function(logging.debug, msg, *args)


def info(msg, *args):
    log_function(logging.info, msg, *args)


def warning(msg, *args):
    log_function(logging.warning, msg, *args)


def error(msg, *args):
    log_function(logging.error, msg, *args)


_log_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
_log_funcs = [logging.debug, logging.info, logging.warning, logging.error]


def log_function(log_func, msg, *args):
    msg = INDENT * depth + (msg % args)
    if not inited:
        print(msg)
    else:
        log_func(msg)
        if output_tee_stdout and _log_levels.index(log_level) <= _log_funcs.index(log_func):
            print(msg)
