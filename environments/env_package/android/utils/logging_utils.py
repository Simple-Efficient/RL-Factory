import logging
import os
import datetime

def init_logging(log_file="logs/main.log", level=logging.DEBUG):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.debug(" Logging initialized.")

def log_with_time(msg: str, level=logging.DEBUG):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] {msg}")

    logging.log(level, f"[{now}] {msg}")
