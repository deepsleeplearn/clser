#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-12-12
# @Author : guojian
# @File : tools.py

import logging
import time
import json


COLORS = {
    "DEBUG": "\033[36m",    
    "INFO": "\033[32m",     
    "WARNING": "\033[33m",  
    "ERROR": "\033[31m",    
    "CRITICAL": "\033[35m", 
    "RESET": "\033[0m",     
}


class ColoredFormatter(logging.Formatter):

    default_msec_format = "%s.%03d"
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
            if self.default_msec_format:
                s = self.default_msec_format % (s, record.msecs)
        else:
            s = time.strftime(self.default_time_format, ct)
            if self.default_msec_format:
                s = self.default_msec_format % (s, record.msecs)
        return s
    
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, "")
        reset = COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        if getattr(record, "author", None) is None:
            author = "midea"
        else:
            author = record.author
        record.author = f"\033[31m{author}{reset}"
        return super().format(record)



def get_logger(name: str = __name__, to_file: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt_str = json.dumps({"asctime": "%(asctime)s", "levelname": "%(levelname)s", "module": "%(module)s",
        "funcName": "%(funcName)s", "message": "%(message)s", "author": "%(author)s"}, ensure_ascii=False)
    
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    colored_formatter = ColoredFormatter(fmt_str, date_fmt)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    if to_file:
        file_handler = logging.FileHandler("log.log", mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        plain_formatter = logging.Formatter(fmt_str, date_fmt)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    return logger
