"""Coloured console logging for CADTopo.

Every module gets its logger via :func:`get_logger`, which adds a custom
``SUCCESS`` level (between INFO and WARNING) so a pass/fail result can be
highlighted without reaching for WARNING.
"""

import logging
import sys
from datetime import datetime

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

WHITE = "\033[37m"
CYAN = "\033[36m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

_LEVEL_STYLES = {
    logging.DEBUG: (DIM + WHITE, "DEBUG  "),
    logging.INFO: (CYAN, "INFO   "),
    SUCCESS: (BRIGHT_GREEN, "SUCCESS"),
    logging.WARNING: (BRIGHT_YELLOW, "WARNING"),
    logging.ERROR: (BRIGHT_RED, "ERROR  "),
    logging.CRITICAL: (BOLD + BRIGHT_RED, "CRITICAL"),
}


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level_color, level_label = _LEVEL_STYLES.get(
            record.levelno, (WHITE, record.levelname[:7].ljust(7))
        )

        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        name = record.name[:22].ljust(22)
        message = record.getMessage()

        line = (
            f"{DIM}{ts}{RESET} "
            f"{level_color}{BOLD}{level_label}{RESET} "
            f"{BRIGHT_BLUE}{name}{RESET}  "
            f"{level_color}{message}{RESET}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def get_logger(name: str) -> logging.Logger:
    """Get/create a CADTopo logger with a ``.success(...)`` convenience method.

    Default level is INFO; opt into DEBUG explicitly with
    ``logger.setLevel(logging.DEBUG)`` when you need the noisier detail.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)

    def success(msg: str, *args, **kwargs):
        logger.log(SUCCESS, msg, *args, **kwargs)

    logger.success = success  # type: ignore[attr-defined]
    return logger
