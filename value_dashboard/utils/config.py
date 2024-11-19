import logging
import tomllib

import streamlit as st

from value_dashboard.utils.string_utils import strtobool
from value_dashboard.utils.logger import get_logger

logger = get_logger(__name__, logging.DEBUG)


@st.cache_resource()
def get_config():
    config_file = None
    if "args" in st.session_state.keys():
        config_file = st.session_state["args"].config
    if not config_file:
        config_file = "value_dashboard/config/config.toml"
    logger.debug("Config file: " + config_file)
    with open(config_file, mode="rb") as fp:
        config = tomllib.load(fp)
    return config


def clv_metrics_avail() -> bool:
    metrics = get_config()["metrics"]
    for metric in metrics:
        if metric.startswith("clv"):
            return True
    return False


def ih_metrics_avail() -> bool:
    metrics = get_config()["metrics"]
    for metric in metrics:
        is_dict = isinstance(metrics[metric], dict)
        if is_dict and (not metric.startswith("clv")):
            return True
    return False

def chat_with_data() -> bool:
    ux = get_config()["ux"]
    if "chat_with_data" in ux.keys():
        return strtobool(ux["chat_with_data"])
    return False
