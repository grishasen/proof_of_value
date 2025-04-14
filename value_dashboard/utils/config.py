import logging
import tomllib

import streamlit as st

from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.string_utils import strtobool

logger = get_logger(__name__, logging.DEBUG)


@st.cache_resource()
def get_config() -> dict:
    config_file = None
    if "args" in st.session_state:
        config_file = st.session_state["args"].config
    if not config_file:
        config_file = "value_dashboard/config/config.toml"

    logger.debug("Config file: " + config_file)

    try:
        with open(config_file, mode="rb") as fp:
            return tomllib.load(fp)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_file}")
        st.error(f"Configuration file not found: {config_file}")
        return {}
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Failed to parse config file: {e}")
        st.error("Configuration file is not valid TOML.")
        return {}


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


def is_demo_mode() -> bool:
    variants = get_config()["variants"]
    return strtobool(variants.get("demo_mode", False))


def chat_with_data() -> bool:
    ux = get_config()["ux"]
    return strtobool(ux.get("chat_with_data", False))
