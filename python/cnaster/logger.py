import logging
import sys
import time


class RuntimeFormatter(logging.Formatter):
    def __init__(self, *args, start_time=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = start_time if start_time is not None else time.time()

    def format(self, record):
        runtime_minutes = (time.time() - self.start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        
        # Look up the parent logger to see if a runtime_phase keyword is currently set
        parent_logger = logging.getLogger(record.name)
        runtime_phase = getattr(parent_logger, "runtime_phase", None)
        
        # Inject the keyword string or clear it if None
        record.runtime_phase_str = f" ({runtime_phase})" if runtime_phase else ""
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


def get_logger(name, start_time, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # Initial state: defaults to None (original behavior)
    logger.runtime_phase = None

    # Place %(runtime_phase_str)s right next to levelname
    formatter = RuntimeFormatter(
        fmt="%(asctime)s - %(runtime)s - %(levelname)-7s%(runtime_phase_str)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        start_time=start_time,
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.warning_once = warning_once.__get__(logger)
    logger.info_once = info_once.__get__(logger)

    return logger
