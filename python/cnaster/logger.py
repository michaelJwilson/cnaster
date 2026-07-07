import logging
import sys
import time
import types

# 1. Create a custom Logger class that shares state across all instances
class SharedStateLogger(logging.Logger):
    _shared_runtime_phase = None

    @property
    def runtime_phase(self):
        return SharedStateLogger._shared_runtime_phase

    @runtime_phase.setter
    def runtime_phase(self, value):
        SharedStateLogger._shared_runtime_phase = value

# Tell Python's logging registry to use this class for all new loggers
logging.setLoggerClass(SharedStateLogger)


def warning_once(self, msg, *args, **kwargs):
    if not hasattr(self, "_seen_warnings"):
        self._seen_warnings = set()
    if msg not in self._seen_warnings:
        kwargs.setdefault("stacklevel", 2)
        self.warning(msg, *args, **kwargs)
        self._seen_warnings.add(msg)


def info_once(self, msg, *args, **kwargs):
    if not hasattr(self, "_seen_infos"):
        self._seen_infos = set()
    if msg not in self._seen_infos:
        kwargs.setdefault("stacklevel", 2)
        self.info(msg, *args, **kwargs)
        self._seen_infos.add(msg)


# 2. Update the Filter to read from the global shared state
class RuntimePhaseFilter(logging.Filter):
    def filter(self, record):
        phase = SharedStateLogger._shared_runtime_phase
        record.runtime_phase_str = f" ({phase})" if phase else ""
        return True


class RuntimeFormatter(logging.Formatter):
    def __init__(self, *args, start_time=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = start_time if start_time is not None else time.time()

    def format(self, record):
        runtime_minutes = (time.time() - self.start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        
        if not hasattr(record, "runtime_phase_str"):
            record.runtime_phase_str = ""
            
        return super().format(record)


def get_logger(name, start_time, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.filters.clear()

    # The filter no longer needs the specific logger instance passed to it
    logger.addFilter(RuntimePhaseFilter())

    formatter = RuntimeFormatter(
        fmt="%(asctime)s - %(runtime)s - %(levelname)-4s%(runtime_phase_str)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        start_time=start_time,
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.warning_once = types.MethodType(warning_once, logger)
    logger.info_once = types.MethodType(info_once, logger)

    return logger