import os
import tomllib
from traceback import print_stack

import streamlit as st

from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.string_utils import strtobool

logger = get_logger(__name__)


@st.cache_resource()
def get_config() -> dict:
    config_file = None
    if "app_config" in st.session_state.keys():
        config_file = st.session_state["app_config"]
    if not config_file:
        package_dir = os.path.dirname(__file__)
        config_file = os.path.join(package_dir, "../config", "config_template.toml")
        #config_file = "value_dashboard/config/config_template.toml"

    logger.debug("Config file: " + config_file)

    try:
        with open(config_file, mode="rb") as fp:
            return tomllib.load(fp)
    except FileNotFoundError:
        print_stack()
        logger.error(f"Config file not found: {config_file}")
        st.error(f"Configuration file not found: {config_file}")
        return {}
    except tomllib.TOMLDecodeError as e:
        print_stack()
        logger.error(f"Failed to parse config file: {e}")
        st.error(f"Configuration file is not valid TOML. {e}")
        return {}
    except KeyError as e:
        print_stack()
        logger.error(f"Failed to parse config file: {e}")
        st.error(f"Configuration file is not valid TOML. {e}")
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
