import logging
import colorlog
import absl.logging
import time
from pathlib import Path

# Track program start time
_start_time = time.time()


class ElapsedTimeFormatter(colorlog.ColoredFormatter):
    def formatTime(self, record, datefmt=None):
        elapsed_seconds = record.created - _start_time
        return f"{elapsed_seconds:.1f}s"


# Stream (console) handler
_stream_handler = colorlog.StreamHandler()
_stream_handler.setFormatter(
    ElapsedTimeFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        style="%",
    )
)

# Get the root logger
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

# Avoid duplicate handlers if reimported
if not any(isinstance(h, colorlog.StreamHandler) for h in LOGGER.handlers):
    LOGGER.addHandler(_stream_handler)

# Sync absl logging level
absl.logging.set_verbosity(absl.logging.INFO)


def update_file_handler(filename: str | Path, level=logging.INFO):
    """Adds or updates file logging."""
    filename = str(filename)
    file_handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
    file_handler.setLevel(level)

    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Remove existing FileHandlers before adding a new one
    for h in LOGGER.handlers[:]:
        if isinstance(h, logging.FileHandler):
            LOGGER.removeHandler(h)

    LOGGER.addHandler(_stream_handler)
    LOGGER.addHandler(file_handler)
