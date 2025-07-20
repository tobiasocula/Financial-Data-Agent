import logging

logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("my_log_file.log")
formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(component)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

