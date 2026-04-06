import logging


# Add SUCCESS level between INFO and WARNING
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)
logging.Logger.success = success

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;1m" 
    gray = "\x1b[38;5;240m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.SUCCESS: green + format + reset,
        logging.DEBUG: gray + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        if isinstance(record.msg, float):
            record.msg = f"{record.msg:.3f}"
        elif isinstance(record.msg, str):
            import re
            record.msg = re.sub(r'\d+\.\d+', lambda x: f"{float(x.group()):.3f}", record.msg)
            
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    
    if log_file:
        import os
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
            for h in logger.handlers
        )
        if not has_file_handler:
            file_exists = os.path.exists(log_file)
            fh = logging.FileHandler(log_file, mode='a' if file_exists else 'w')
            fh.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
        
    return logger


if __name__ == "__main__":
    logger = get_logger("test")
    logger.success("test success")
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    logger.critical("test critical")