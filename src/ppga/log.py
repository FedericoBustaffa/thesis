import logging

from colorama import Back, Fore

FMT = "[{levelname}] {name}: {message}"


DEBUG = logging.DEBUG
SUCCESS = 15
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(15, "SUCCESS")

FORMATS = {
    logging.DEBUG: Fore.CYAN + FMT + Fore.RESET,
    SUCCESS: Fore.GREEN + FMT + Fore.RESET,
    logging.INFO: Fore.WHITE + FMT + Fore.RESET,
    logging.WARNING: Fore.YELLOW + FMT + Fore.RESET,
    logging.ERROR: Fore.RED + FMT + Fore.RESET,
    logging.CRITICAL: Back.RED + Fore.WHITE + FMT + Fore.RESET + Back.RESET,
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(FORMATS[record.levelno], style="{")
        return formatter.format(record)


class Logger(logging.Logger):
    """Wrapper around the Logger class of the `logging` module"""

    def __init__(self, name, level):
        super().__init__(name, level)

    def success(self, message) -> None:
        super().log(SUCCESS, message)


core_logger = Logger("CORE", WARNING)
formatter = ColorFormatter()
handler = logging.StreamHandler()
handler.setLevel(WARNING)
handler.setFormatter(formatter)
core_logger.addHandler(handler)


user_logger = Logger("USER", INFO)
formatter = ColorFormatter()
handler = logging.StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(formatter)
user_logger.addHandler(handler)


def set_core_level(level: int = WARNING):
    """Set the core logger level"""
    core_logger.setLevel(level)


def getLogger(level: int = INFO) -> Logger:
    """Provides a logger for the user with the given log level"""
    user_logger.setLevel(level)
    return user_logger
