import os

import streamlit as st

from value_dashboard.config_generator import render_configuration_studio
from value_dashboard.pipeline import holdings
from value_dashboard.report_builder import render_report_inventory
from value_dashboard.utils.config import get_config

st.set_page_config(page_title="🔧 Configuration Editor", layout="wide")
st.markdown(
    """
<style>
    .stMainBlockContainer {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
    """,
    unsafe_allow_html=True
)


def clear_config_cache():
    get_config.clear()
    holdings.get_reports_data.clear()


with st.sidebar:
    st.button("Clear config cache 🗑️", on_click=lambda: clear_config_cache())

tabs = ["🧰 Configuration Editor", "📝 Readme", "📊 Reports"]
conf, readme, reports = st.tabs(tabs)

with conf:
    render_configuration_studio(get_config().copy())

with readme:
    with open(os.path.join(os.path.dirname(__file__), "../../README.md"), "r") as f:
        readme_line = f.readlines()
        readme_buffer = []

    for line in readme_line:
        readme_buffer.append(line)

    st.markdown("".join(readme_buffer))

with reports:
    render_report_inventory(get_config())
