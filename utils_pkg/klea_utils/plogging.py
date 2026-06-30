#!/usr/bin/env python3
"""
Logging related utils

File: klea_utils/logging.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import sys


class LoggerNotInfoFilter(logging.Filter):
    """Allow only non INFO messages"""

    def filter(self, record):
        return record.levelno != logging.INFO


class LoggerInfoFilter(logging.Filter):
    """Allow only INFO messages"""

    def filter(self, record):
        return record.levelno == logging.INFO


logger_formatter_info = logging.Formatter(
    "%(asctime)s %(name)s (%(levelname)s) >>> %(message)s\n\n"
)
logger_formatter_other = logging.Formatter(
    "%(asctime)s %(name)s (%(levelname)s) in '%(funcName)s' >>> %(message)s\n\n"
)


def setup_logger(name: str, stderr_level: int = logging.DEBUG) -> logging.Logger:
    """Configure a dual-stream logger.

    INFO-level messages go to stdout with a simple format.  All other
    levels (DEBUG, WARNING, ERROR, CRITICAL) go to stderr with a
    format that includes the function name.

    :param name: Name of the logger (passed to ``logging.getLogger``)
    :param stderr_level: Level for the stderr handler (default ``DEBUG``)
    :returns: Configured logger with handlers attached
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(LoggerInfoFilter())
    stdout_handler.setFormatter(logger_formatter_info)
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(stderr_level)
    stderr_handler.addFilter(LoggerNotInfoFilter())
    stderr_handler.setFormatter(logger_formatter_other)
    logger.addHandler(stderr_handler)

    return logger
