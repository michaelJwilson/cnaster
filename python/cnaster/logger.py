import logging
import sys
import time

start_time = time.time()

class RuntimeFormatter(logging.Formatter):
    def format(self, record):
        runtime_minutes = (time.time() - start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        return super().format(record)

def warning_once(self, msg, *args, **kwargs):
    if not hasattr(warning_once, "_seen"):
        warning_once._seen = set()

    if msg not in warning_once._seen:
        self.warning(msg, *args, **kwargs)
        warning_once._seen.add(msg)


def info_once(self, msg, *args, **kwargs):
    if not hasattr(info_once, "_seen"):
        info_once._seen = set()

    if msg not in info_once._seen:
        self.info(msg, *args, **kwargs)
        info_once._seen.add(msg)


def get_logger(name, level=logging.INFO):
    """
    Returns a configured logger with stream handler and formatter.

    Example usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = RuntimeFormatter(
        fmt="%(asctime)s - %(runtime)s - %(name)s - %(levelname)-7s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    logger.warning_once = warning_once.__get__(logger)
    logger.info_once = info_once.__get__(logger)

    return logger