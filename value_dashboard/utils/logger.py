import logging


class LastPartFilter(logging.Filter):
    def filter(self, record):
        record.name_last = record.name.rsplit('.', 1)[-1]
        return True


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d:%(levelname)s:%(name_last)s:%(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
if logging.getLogger().hasHandlers():
    for h in logging.getLogger().handlers:
        h.addFilter(LastPartFilter())


@staticmethod
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
