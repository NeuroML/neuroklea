#!/usr/bin/env python3
"""
Logging related utils

File: neuroml_ai_utils/logging.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging


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
