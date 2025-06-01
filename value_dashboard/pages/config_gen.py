import os
import tempfile
import tomllib
import uuid
from traceback import print_stack

import polars as pl
import streamlit as st
import tomlkit
from pandasai.helpers.memory import Memory
from pandasai_openai import OpenAI

from value_dashboard.pipeline import holdings
from value_dashboard.pipeline import ih
from value_dashboard.utils.config import get_config
from value_dashboard.utils.file_utils import read_dataset_export
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.polars_utils import schema_with_unique_counts
from value_dashboard.utils.string_utils import capitalize

logger = get_logger(__name__)


def set_config(cfg_file: str):
    del st.session_state.app_config
    ih.get_reports_data.clear()
    holdings.get_reports_data.clear()
    st.session_state.app_config = cfg_file
    get_config.clear()
    get_config()


@st.fragment()
def generate_new_config(llm, prompt):
    memory = Memory(agent_description="Config file generator.")
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


f"""## ✨ GenAI Config Generator"""
with st.sidebar:
    template_config_file = "value_dashboard/config/config_template.toml"
    try:
        with open(template_config_file, mode="rb") as fp:
            template_config = tomllib.load(fp)
    except FileNotFoundError:
        print_stack()
        st.error(f"Configuration file not found: {template_config_file}")
        st.stop()
    except tomllib.TOMLDecodeError as e:
        print_stack()
        st.error(f"Configuration file is not valid TOML. {e}")
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
        index=OpenAI._supported_chat_models.index("gpt-4o-mini")
    )
    llm = OpenAI(
        api_token=openai_api_key,
        temperature=0,
        model=model_choice,
        max_tokens=8192
    )

st.subheader("Choose file with IH sample", divider='red')
uploaded_file = st.file_uploader("*", type=["zip", "parquet", "json", "gzip"],
                                 accept_multiple_files=False)
df = pl.DataFrame()
if uploaded_file:
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

if not df.is_empty():
    df = df.lazy()
    with_cols_list = []
    if 'default_values' in template_config["ih"]["extensions"].keys():
        default_values = template_config["ih"]["extensions"]["default_values"]
        for new_col in default_values.keys():
            if new_col not in capitalized:
                with_cols_list.append(pl.lit(default_values.get(new_col)).alias(new_col))
            else:
                with_cols_list.append(pl.col(new_col).fill_null(default_values.get(new_col)))
    if with_cols_list:
        df = df.with_columns(with_cols_list)

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
        .collect()
    )
    df = df.select(sorted(df.columns))
    schema_df = schema_with_unique_counts(df).sort('Column')
    st.subheader("Schema", divider=True)
    st.data_editor(schema_df,
                   use_container_width=True,
                   disabled=True, height=300, hide_index=True)
    st.subheader("Data Summary", divider=True)
    st.dataframe(df.describe())

    with st.expander("View Data Sample", expanded=False, icon=":material/analytics:"):
        st.dataframe(df)

    with pl.Config(tbl_cols=len(schema_df), tbl_rows=len(schema_df)):
        prompt = f"""
            Given interaction history dataset schema (column names and types) and configuration file template, please create 
            similar config file, suited for this data. 
            Keep all the reports, metrics and other settings, but adjust columns, so they correspond to the 
            data in the file provided. Check columns available in the schema and include in the configuration 
            only those available in the sample. Do not generate 'chat_with_data' section.
            Replace names in template (in filters, group-by and report parameters) with column names in the schema (they may differ by case or have different prefixes or suffixes).
            Set 'file_type' to either 'parquet' (use file name extension to determine file type) or 'pega_ds_export' otherwise.
            Set 'file_pattern' extension accordingly.
            If column of type 'String' is not identifier column (ends with ID) or 'Outcome' and has number of unique values more than 1 and less than 100 - 
            append this column name to 'group_by' property of each metric in 'metrics' section.
            Always include time columns (Year, Quarter, Month, Day) and global filters to 'group_by' parameters.
            File name: {str(uploaded_file.name)}.
            Dataset schema: {schema_df}.
            Template config file: {tomlkit.dumps(template_config)}.
            """

        st.write("## Config from sample")
        if st.button("Generate config", key='UploadBtn', type='primary'):
            logger.debug('LLM prompt: ' + prompt)
            with st.spinner("Generating config. Wait for it...", show_time=True):
                logger.info('LLM call: ' + prompt)
                generate_new_config(llm, prompt)
