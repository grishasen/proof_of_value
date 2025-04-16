import logging
import logging.config
import os

import yaml


class LastPartFilter(logging.Filter):
    def filter(self, record):
        record.name_last = record.name.rsplit(".", 1)[-1]
        return True


def configure_logging(config_path="value_dashboard/config/logging_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name_last)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    if logging.getLogger().hasHandlers():
        for h in logging.getLogger().handlers:
            h.addFilter(LastPartFilter())


@staticmethod
def get_logger(name, level=None):
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level)
    return logger
