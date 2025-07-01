import os

import streamlit as st

from value_dashboard.pipeline import holdings
from value_dashboard.utils.config import get_config
from value_dashboard.utils.config_builder import render_config_editor


def clear_config_cache():
    get_config.clear()
    holdings.get_reports_data.clear()


with st.sidebar:
    st.button("Clear config cache 🗑️", on_click=lambda: clear_config_cache())

tabs = ["📄 Configuration", "📝 Readme"]
conf, readme = st.tabs(tabs)

with conf:
    render_config_editor(get_config().copy())

with readme:
    with open(os.path.join(os.path.dirname(__file__), "../../README.md"), "r") as f:
        readme_line = f.readlines()
        readme_buffer = []

    for line in readme_line:
        readme_buffer.append(line)

    st.markdown("".join(readme_buffer))
