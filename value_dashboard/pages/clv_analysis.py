import re
import time
from collections import defaultdict, OrderedDict

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_dimensions import st_dimensions
from streamlit_option_menu import option_menu
from streamlit_theme import st_theme

from value_dashboard.pipeline.holdings import get_reports_data
from value_dashboard.reports.reports import get_figures
from value_dashboard.utils.config import get_config
from value_dashboard.utils.st_utils import highlight_and_format, format_dates
from value_dashboard.utils.string_utils import strtobool


def download_clv_dataset(df):
    st.download_button(
        label="Export RFM data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='rfm_dataset.csv',
        mime="text/csv",
        help="Exported behavioural customer segmentation can be used to act on CLV in engagement policies, levers, etc."
    )


pd.options.styler.format.thousands = ','
pd.options.styler.format.na_rep = 'background-color: lightgrey;'
pd.options.styler.format.precision = 5
dataset_max_rows = 1000
pd.set_option("styler.render.max_elements", dataset_max_rows * 100)

theme = st_theme(key='clv_analysis')
st.session_state['theme'] = theme
dims = st_dimensions()
st.session_state['dashboard_dims'] = dims

if "holdings_data_loaded" not in st.session_state:
    st.warning("Please configure your files in the `data import` tab.")
    st.stop()

if strtobool(get_config()["ux"]["refresh_dashboard"]):
    count = st_autorefresh(interval=get_config()["ux"]["refresh_interval"], key="dashboard-counter")

tabs = ["🗃 Data Overview"]

reports_data = get_reports_data()

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 2rem;
                    padding-left: 1rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)
f"""## 📊 Customer Lifetime Value Analysis Dashboard"""
figures = get_figures()
reports_name_map = OrderedDict()
reports = get_config()["reports"]
for report in reports:
    params = reports[report]
    if params['metric'].startswith("clv"):
        reports_name_map[params["description"].strip()] = report
reports_list = list(reports_name_map.keys())
result = defaultdict(list)
pattern = re.compile(r'\[([^\[\]]+)\]')
for string in reports_list:
    matches = pattern.findall(string)
    for match in matches:
        cleaned_string = string
        result[match].append(cleaned_string)
result_dict = dict(result)
with st.sidebar:
    if 'clv_dashboard_last_access_time' not in st.session_state:
        st.session_state['clv_dashboard_last_access_time'] = time.time()
    if 'clv_selected_report' not in st.session_state:
        previous_selected_report = None
    else:
        previous_selected_report = st.session_state['clv_selected_report']

    previous_access_time = st.session_state['clv_dashboard_last_access_time']

    manual_select = None
    if strtobool(get_config()["ux"]["refresh_dashboard"]):
        if previous_selected_report:
            if (time.time() - previous_access_time + 5) * 1000 >= get_config()["ux"]["refresh_interval"]:
                idx = reports_list.index(previous_selected_report)
                manual_select = (idx + 1) % len(reports_list)

    with st.expander("**Select report**", icon=":material/analytics:"):
        selected_report = option_menu("", reports_list,
                                      manual_select=manual_select,
                                      icons=["-" for r in reports_list],
                                      styles={
                                          "container": {"padding": "0!important", "background-color": "transparent"},
                                          "nav-link": {"font-size": "12px", "text-align": "left", "margin": "0px"}
                                      },
                                      key='select_report_menu')
    st.session_state['dashboard_last_access_time'] = time.time()
    st.session_state['selected_report'] = selected_report

df = reports_data[reports_name_map[selected_report]][0].to_pandas()

params = reports_data[reports_name_map[selected_report]][1]
filtered_rep_data = figures[reports_name_map[selected_report]](df, params)

grp_by = reports[reports_name_map[selected_report]]["group_by"]
cols = list(set(filtered_rep_data.columns.tolist()) - set(grp_by))
column_order = grp_by + sorted(cols)
c1, c2, c3 = st.columns([0.55, 0.25, 0.2], gap="large", vertical_alignment="top")
c1.write("#### 🗃 Data Overview")
# col_order_on = c2.toggle("Reorder columns", value=False, help="Change dataframe table columns order.")
with c3:
    download_clv_dataset(filtered_rep_data)
# if col_order_on:
#    column_order = sort_items(items=column_order, header="Columns order: ")
st.data_editor(format_dates(filtered_rep_data.head(dataset_max_rows))
               .map(highlight_and_format),
               use_container_width=True,
               column_order=column_order, disabled=True)
