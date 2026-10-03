"""
统一日志：所有 print 和 logger.info 同时写入 logs/<timestamp>.log 和控制台。
"""
import logging
import os
import sys
from datetime import datetime


def setup_logger():
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{ts}.log")

    logger = logging.getLogger("bot")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 防止重复添加
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.__stdout__)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 重定向 print -> logger
    class _PrintProxy:
        def __init__(self, lg):
            self.lg = lg
            self._buf = ""

        def write(self, msg):
            self._buf += msg
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self.lg.info(line)

        def flush(self):
            if self._buf.strip():
                self.lg.info(self._buf.strip())
            self._buf = ""

    sys.stdout = _PrintProxy(logger)
    sys.stderr = _PrintProxy(logger)

    return logger, log_path
