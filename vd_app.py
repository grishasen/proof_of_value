import argparse
import os
from traceback import print_stack

import streamlit as st

from value_dashboard.utils.config import clv_metrics_avail, ih_metrics_avail, chat_with_data, is_demo_mode
from value_dashboard.utils.logger import configure_logging
from value_dashboard.utils.st_utils import get_page_configs

st.set_option("client.showErrorDetails", 'full')
st.set_option("client.toolbarMode", 'minimal')
st.set_page_config(**get_page_configs())


def create_page(relative_path, name):
    current_dir = os.path.dirname(__file__)
    return st.Page(os.path.join(current_dir, relative_path), title=name)


def get_pages():
    pages = [
        create_page("value_dashboard/pages/home.py", "Home"),
        create_page("value_dashboard/pages/data_import.py", "Data Import"),
    ]
    if ih_metrics_avail():
        pages.append(create_page("value_dashboard/pages/dashboard.py", "Dashboard"))
    if ih_metrics_avail() and chat_with_data():
        pages.append(create_page("value_dashboard/pages/chat_with_data.py", "Chat with data"))
    if clv_metrics_avail():
        pages.append(create_page("value_dashboard/pages/clv_analysis.py", "CLV Analysis"))
    pages.append(create_page("value_dashboard/pages/toml_editor.py", "Configuration"))
    if not is_demo_mode():
        pages.append(create_page("value_dashboard/pages/config_gen.py", "GenAI Config"))
    return [p for p in pages if p is not None]


parser = argparse.ArgumentParser(description='Command line arguments')

parser.add_argument('--config', action='store', default="",
                    help="Config file")
parser.add_argument('--logging_config', action='store', default="",
                    help="Logging config file")

try:
    args = parser.parse_args()
    st.session_state['app_config'] = args.config
    st.session_state['logging_config'] = args.logging_config
except SystemExit as e:
    pass

if args.logging_config:
    configure_logging(config_path=args.logging_config)
else:
    package_dir = os.path.dirname(__file__)
    config_file = os.path.join(package_dir, "value_dashboard/config", "logging_config.yaml")
    configure_logging(config_path=config_file)
pages = get_pages()
pg = st.navigation(pages, expanded=False)
pg.run()

