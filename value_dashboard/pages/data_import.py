import json
import os
import tempfile
import time
import typing
import zipfile

import streamlit as st

from value_dashboard.pipeline.holdings import load_holdings_data
from value_dashboard.pipeline.ih import load_data
from value_dashboard.utils.config import ih_metrics_avail, clv_metrics_avail, is_demo_mode
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.plotly_utils import init_plotly_theme

logger = get_logger(__name__)
init_plotly_theme()
st.title("Importing the data")


@st.fragment()
def download_collected_metrics(name, data_loaded):
    collected_metrics_data: typing.Dict[str, str] = {}
    for metric in data_loaded:
        frame = data_loaded[metric]
        collected_metrics_data[metric] = json.loads(frame.serialize(format='json'))
    st.download_button(
        label="Download aggregated data",
        data=json.dumps(collected_metrics_data),
        file_name=name,
        mime="text/json",
    )


@st.fragment
def import_data():
    if is_demo_mode():
        st.info("Application is in DEMO mode. Wait for data load.")
        use_aggregated = True
        st.session_state['use_aggregated'] = use_aggregated
        st.session_state['aggregated_path'] = 'data/demo_collected_ih_metrics_data.json'
        st.session_state['drop_cache'] = False
        data_loaded = load_data()
        st.session_state['data_loaded'] = True
        st.session_state['data_load_run'] = True
    else:
        use_aggregated = False
        raw_load = st.toggle("Import raw data", value=True,
                             help="Load raw data or pre-aggregated metrics data")
        if raw_load:
            reload_all = st.toggle("Remove existing data", value=False,
                                   help="Reload all data from the source, drop cache.")
            st.info("Enter folder name with interaction history files or upload files.")
            data_source = st.radio("Choose your data source", ('File Upload', 'Folder'))
            if data_source == 'Folder':
                folder_path = st.text_input("Enter folder path")
                if folder_path and st.button("Load Data", key='load_ih_data'):
                    st.session_state.clear()
                    st.cache_data.clear()
                    st.toast('Starting data processing...', icon="🗃")
                    if not folder_path.endswith(os.sep):
                        folder_path = folder_path + os.sep
                    if not os.path.isdir(folder_path):
                        st.error("Folder not found.")
                        st.stop()
                    st.session_state['ihfolder'] = folder_path
                    st.session_state['drop_cache'] = reload_all
                    data_loaded = load_data()
                    if not data_loaded:
                        st.error("Data could not be loaded. Please check the input.")
                        return
                    st.session_state['data_loaded'] = True
                    st.session_state['data_load_run'] = True
                    st.info(f"Data loaded from {folder_path}")

            elif data_source == 'File Upload':
                uploaded_files = st.file_uploader("Choose a file", type=["zip", "parquet", "json"],
                                                  accept_multiple_files=True)
                if uploaded_files and st.button("Upload", key='upload_ih_data'):
                    st.session_state.clear()
                    st.cache_data.clear()
                    st.toast('Starting data processing...', icon="🗃")
                    temp_dir = tempfile.TemporaryDirectory(prefix='tmp')
                    folder_path = os.path.abspath(temp_dir.name)
                    if not folder_path.endswith(os.sep):
                        folder_path = folder_path + os.sep
                    st.session_state['ihfolder'] = folder_path
                    st.session_state['drop_cache'] = True
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        if zipfile.is_zipfile(file_path):
                            st.write("The uploaded file is a zip file. Unzipping...")
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_dir.name)

                    data_loaded = load_data()
                    if not data_loaded:
                        st.error("Data could not be loaded. Please check the input.")
                        return
                    st.session_state['data_loaded'] = True
                    st.session_state['data_load_run'] = True
                    temp_dir.cleanup()
        else:
            st.info("Use JSON file with pre-aggregated data.")
            uploaded_file = st.file_uploader("Choose a file", type=["json"],
                                             accept_multiple_files=False)
            if uploaded_file and st.button("Upload", key='upload_ih_data'):
                st.session_state.clear()
                st.cache_data.clear()
                st.toast('Starting data processing...', icon="🗃")
                temp_dir = tempfile.TemporaryDirectory(prefix='tmp')
                folder_path = os.path.abspath(temp_dir.name)
                if not folder_path.endswith(os.sep):
                    folder_path = folder_path + os.sep
                st.session_state['ihfolder'] = folder_path
                file_path = os.path.join(temp_dir.name, 'collected_metrics_data.json')
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                use_aggregated = True
                st.session_state['use_aggregated'] = use_aggregated
                st.session_state['aggregated_path'] = file_path
                st.session_state['drop_cache'] = False
                data_loaded = load_data()
                if not data_loaded:
                    st.error("Data could not be loaded. Please check the input.")
                    return
                st.session_state['data_loaded'] = True
                st.session_state['data_load_run'] = True
                temp_dir.cleanup()

    # Check if data is loaded to unlock the reports page
    if (('data_loaded' in st.session_state and st.session_state['data_loaded'])
            and ('data_load_run' in st.session_state and st.session_state['data_load_run'])):
        st.session_state['data_load_run'] = False
        msg = st.toast('Data loaded.', icon="🗃")
        time.sleep(1)
        msg.toast('Generating plots...', icon="📈")
        time.sleep(1)
        msg.toast('Dashboard ready.', icon="📊")
        if not use_aggregated:
            download_collected_metrics("collected_ih_metrics_data.json", data_loaded)
        st.success("Go to the Dashboard page using the sidebar")


@st.fragment
def import_holdings_data():
    if is_demo_mode():
        st.info("Application is in DEMO mode. Wait for data load.")
        load_holdings_data.clear()
        st.toast('Starting data processing...', icon="🗃")
        st.session_state['holdingsfolder'] = 'data/PegaCDH-Data-ProductHolding_HoldingsDDS_20241010T145658_GMT'
        data_loaded = load_holdings_data()
        st.session_state['holdings_data_loaded'] = True
    else:
        data_loaded = None
        st.info("Enter folder name with product holdings files or upload files.")
        data_source = st.radio("Choose your data source", ('File Upload', 'Folder'), key='product_holdings_ds_radio')
        if data_source == 'Folder':
            folder_path = st.text_input("Enter folder path")
            if folder_path and st.button("Load Data", key='load_holdings_data'):
                load_holdings_data.clear()
                st.toast('Starting data processing...', icon="🗃")
                if not folder_path.endswith(os.sep):
                    folder_path = folder_path + os.sep
                if not os.path.isdir(folder_path):
                    st.error("Folder not found.")
                    st.stop()
                st.session_state['holdingsfolder'] = folder_path
                data_loaded = load_holdings_data()
                if not data_loaded:
                    st.error("Data could not be loaded. Please check the input.")
                    return
                st.session_state['holdings_data_loaded'] = True
                st.info(f"Data loaded from {folder_path}")

        elif data_source == 'File Upload':
            uploaded_files = st.file_uploader("Choose a file", type=["zip", "parquet", "json", "csv", "xlsx"],
                                              accept_multiple_files=True, key='product_holdings_ds_file_uploader')
            if uploaded_files and st.button("Upload", key='upload_holdings_data'):
                load_holdings_data.clear()
                st.toast('Starting data processing...', icon="🗃")
                temp_dir = tempfile.TemporaryDirectory(prefix='tmp')
                folder_path = os.path.abspath(temp_dir.name)
                if not folder_path.endswith(os.sep):
                    folder_path = folder_path + os.sep
                st.session_state['holdingsfolder'] = folder_path
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir.name, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if zipfile.is_zipfile(file_path) and not file_path.endswith('xlsx'):
                        st.write("The uploaded file is a zip file. Unzipping...")
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir.name)
                data_loaded = load_holdings_data()
                if not data_loaded:
                    st.error("Data could not be loaded. Please check the input.")
                    return
                st.session_state['holdings_data_loaded'] = True
                temp_dir.cleanup()

    if 'holdings_data_loaded' in st.session_state and st.session_state['holdings_data_loaded'] and data_loaded:
        msg = st.toast('Data loaded.', icon="🗃")
        time.sleep(1)
        msg.toast('Generating plots...', icon="📈")
        time.sleep(1)

        msg.toast('Dashboard ready.', icon="📊")
        if data_loaded:
            download_collected_metrics("collected_clv_data.json", data_loaded)
        st.success("Go to the CLV Analysis page using the sidebar")


tabs = ["Import Interaction History"] + (["Import Product Holdings"] if clv_metrics_avail() else [])
st_tabs = st.tabs(tabs)
with st_tabs[0]:
    if ih_metrics_avail():
        import_data()
    else:
        st.warning("Please configure IH metrics.")
if clv_metrics_avail():
    with st_tabs[1]:
        if clv_metrics_avail():
            import_holdings_data()
        else:
            st.warning("Please configure CLV metrics.")
