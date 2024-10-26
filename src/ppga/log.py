import logging

from colorama import Fore

FMT = "[{levelname}] {name}: {message}"


DEBUG = logging.DEBUG
SUCCESS = 15
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

logging.addLevelName(15, "SUCCESS")

FORMATS = {
    logging.DEBUG: Fore.CYAN + FMT + Fore.RESET,
    SUCCESS: Fore.GREEN + FMT + Fore.RESET,
    logging.INFO: Fore.WHITE + FMT + Fore.RESET,
    logging.WARNING: Fore.YELLOW + FMT + Fore.RESET,
    logging.ERROR: Fore.RED + FMT + Fore.RESET,
    logging.CRITICAL: Fore.LIGHTRED_EX + FMT + Fore.RESET,
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(FORMATS[record.levelno], style="{")
        return formatter.format(record)


class Logger(logging.Logger):
    def __init__(self, name, level):
        super().__init__(name, level)

    def success(self, message) -> None:
        super().log(SUCCESS, message)


def getLogger(name: str, level: int | str = INFO) -> Logger:
    logger = Logger(name, level)
    formatter = ColorFormatter()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
