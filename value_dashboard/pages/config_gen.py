import os
import tempfile
import tomllib
import uuid

import pandas as pd
import polars as pl
import streamlit as st
import tomlkit
from pandasai.helpers.memory import Memory
from pandasai_openai import OpenAI

from value_dashboard.pipeline import holdings
from value_dashboard.pipeline import ih
from value_dashboard.utils.config import get_config
from value_dashboard.utils.file_utils import read_dataset_export
from value_dashboard.utils.string_utils import capitalize


def set_config(cfg_file: str):
    del st.session_state.app_config
    ih.get_reports_data.clear()
    holdings.get_reports_data.clear()
    st.session_state.app_config = cfg_file_name
    get_config.clear()
    get_config()


f"""## ✨ GenAI Config Generator"""
with st.sidebar:
    template_config_file = "value_dashboard/config/config_template.toml"
    try:
        with open(template_config_file, mode="rb") as fp:
            template_config = tomllib.load(fp)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {template_config_file}")
        st.stop()
    except tomllib.TOMLDecodeError as e:
        st.error("Configuration file is not valid TOML.")
        st.stop()

    api_key_input = st.text_input(
        "Enter API Key (Leave empty to use environment variable)",
        type="password",
        value=os.environ.get("OPENAI_API_KEY"),
    )
    st.markdown(
        """
    <style>
        [title="Show password text"] {
            display: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    openai_api_key = (
        api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY")
    )
    if not openai_api_key:
        st.error("Please configure LLM API key.")
        st.stop()
    model_choice = st.selectbox(
        "Choose Model",
        options=OpenAI._supported_chat_models,
        index=OpenAI._supported_chat_models.index("gpt-4o")
    )
    llm = OpenAI(
        api_token=openai_api_key,
        temperature=0,
        model=model_choice,
        max_tokens=8192
    )

st.subheader("Choose file with IH sample", divider='red')
uploaded_file = st.file_uploader("*", type=["zip", "parquet"],
                                 accept_multiple_files=False)
if uploaded_file and st.button("Generate Config"):
    temp_dir = tempfile.TemporaryDirectory(prefix='tmp')
    folder_path = os.path.abspath(temp_dir.name)
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = read_dataset_export(file_name=uploaded_file.name, src_folder=os.path.dirname(file_path), lazy=False)
    dframe_columns = df.collect_schema().names()
    capitalized = capitalize(dframe_columns)
    rename_map = dict(zip(dframe_columns, capitalized))
    df = df.rename(rename_map)
    temp_dir.cleanup()

    df = (
        df.with_columns([
            pl.col('OutcomeTime').cast(str).str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z").alias('OutcomeDateTime'),
            pl.col('DecisionTime').cast(str).str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z").alias('DecisionDateTime')
        ])
        .with_columns([
            pl.col("OutcomeDateTime").dt.date().alias("Day"),
            pl.col("OutcomeDateTime").dt.strftime("%Y-%m").alias("Month"),
            pl.col("OutcomeDateTime").dt.year().cast(pl.Utf8).alias("Year"),
            (pl.col("OutcomeDateTime").dt.year().cast(pl.Utf8) + "_Q" +
             pl.col("OutcomeDateTime").dt.quarter().cast(pl.Utf8)).alias("Quarter"),
            (pl.col("OutcomeDateTime") - pl.col("DecisionDateTime")).dt.total_seconds().alias("ResponseTime")
        ])
        .drop([
            "FactID", "Label", "UpdateDateTime", "OutcomeTime", "DecisionTime",
            "OutcomeDateTime", "StreamPartition", "EvaluationCriteria", "Organization",
            "Unit", "Division", "Component", "ApplicationVersion", "Strategy"
        ], strict=False)
    )

    schema_df = pd.DataFrame(
        [(col, str(dtype)) for col, dtype in df.schema.items()],
        columns=["Column", "Data Type"]
    ).sort_values('Column', axis=0)
    st.subheader("Schema", divider=True)
    st.data_editor(schema_df,
                   use_container_width=True,
                   disabled=True, height=300, hide_index=True)
    st.subheader("Data Summary", divider=True)
    st.write(df.describe())

    prompt = f"""
        Given interaction history dataset schema (column names and types) and configuration file template, please create 
        similar config file, suited for this data. 
        Keep all the reports, metrics and other settings, but adjust columns, so they correspond to the 
        data in the file provided. Check columns available in the schema and include in the configuration 
        only those available in the sample. Do not generate 'chat_with_data' section.
        Try to map column names in the schema to those in template (may differ by case or have different prefixes or suffixes).
        Set 'file_type' to either 'parquet' (use file name extension to determine file type) or 'pega_ds_export' otherwise.
        Set 'file_pattern' extension accordingly.
        File name: {str(file_path)}.
        Dataset schema: {str(df.schema)}.
        Template config file: {tomlkit.dumps(template_config)}.
        """

    st.write("## Generate config from sample schema")
    memory = Memory(agent_description="Config file generator.")
    with st.spinner("Wait for it...", show_time=True):
        new_config_text = llm.chat_completion(value=prompt, memory=memory)

    lines = new_config_text.splitlines(keepends=True)
    new_config_text = ''.join(lines[1:])
    new_config_text = new_config_text.replace('```', '')
    new_cfg = tomllib.loads(new_config_text)
    new_cfg["chat_with_data"] = get_config()["chat_with_data"]
    new_cfg["ux"]['chat_with_data'] = 'true'
    new_config_text = tomlkit.dumps(new_cfg)
    try:
        os.makedirs("temp_configs")
    except FileExistsError:
        pass
    cfg_file_name = "temp_configs/" + "config_" + uuid.uuid4().hex + '.toml'
    with open(cfg_file_name, "w") as f:
        f.write(new_config_text)

    set_config(cfg_file_name)
    st.download_button(
        label="Download",
        data=new_config_text,
        file_name="config.toml",
        mime="text/plain",
        type='primary'
    )
