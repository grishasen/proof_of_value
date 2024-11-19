import argparse
import os

import streamlit as st

from value_dashboard.utils.config import clv_metrics_avail, ih_metrics_avail, chat_with_data
from value_dashboard.utils.st_utils import get_page_configs

st.set_page_config(**get_page_configs())


def _create_page(relative_path, name):
    current_dir = os.path.dirname(__file__)
    return st.Page(os.path.join(current_dir, relative_path), title=name)


def get_pages():
    pages = (
            [
                _create_page("value_dashboard/pages/home.py", "Home"),
                _create_page("value_dashboard/pages/data_import.py", "Data Import")
            ]
            +
            (
                [
                    _create_page("value_dashboard/pages/dashboard.py", "Dashboard")
                ] if ih_metrics_avail() else []
            )
            +
            (
                [
                    _create_page("value_dashboard/pages/chat_with_data.py", "Chat with data")
                ] if (ih_metrics_avail() & chat_with_data()) else []
            )
            +
            (
                [
                    _create_page("value_dashboard/pages/clv_analysis.py", "CLV Analysis")
                ] if clv_metrics_avail() else []
            )
            +
            [
                _create_page("value_dashboard/pages/toml_editor.py", "Configuration")
            ]
    )
    return pages


parser = argparse.ArgumentParser(description='Command line arguments')

parser.add_argument('--config', action='store', default="",
                    help="Config file")

try:
    args = parser.parse_args()
    st.session_state['args'] = args
except SystemExit as e:
    pass

pages = get_pages()
pg = st.navigation(pages, expanded=False)
pg.run()
