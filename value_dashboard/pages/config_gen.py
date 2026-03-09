import os
import tempfile
import tomllib
import uuid
from traceback import print_stack

import polars as pl
import streamlit as st
import tomlkit
from jinja2 import Environment
from pandasai.core.prompts import BasePrompt
from pandasai_litellm import LiteLLM

from value_dashboard.metrics.constants import DROP_IH_COLUMNS, OUTCOME_TIME, DECISION_TIME
from value_dashboard.utils.config import get_config, set_config
from value_dashboard.utils.file_utils import read_dataset_export
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.polars_utils import schema_with_unique_counts
from value_dashboard.utils.py_utils import capitalize

logger = get_logger(__name__)

supported_responses_models = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.4",
    "gpt-5.4-pro",
]
model: str = "gpt-5.4"
reasoning_effort = "high"  # "minimal" | "low" | "medium" | "high"
verbosity = "medium"  # "low" | "medium" | "high"


@st.fragment()
def generate_new_config(llm, prompt):
    env = Environment()
    instruction = BasePrompt()
    instruction.prompt = env.from_string(prompt)
    new_config_text = llm.call(instruction=instruction)
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
    package_dir = os.path.dirname(__file__)
    template_config_file = os.path.join(package_dir, "../config", "config_template.toml")
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
        options=supported_responses_models,
        index=supported_responses_models.index(model)
    )
    llm = LiteLLM(model=model_choice, api_key=openai_api_key,
                  reasoning_effort=reasoning_effort, verbosity=verbosity)

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
    df = read_dataset_export(file_names=uploaded_file.name, src_folder=os.path.dirname(file_path), lazy=False)
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
            pl.col(OUTCOME_TIME).str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z"),
            pl.col(DECISION_TIME).str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z")
        ])
        .with_columns([
            pl.col(OUTCOME_TIME).dt.date().alias("Day"),
            pl.col(OUTCOME_TIME).dt.strftime("%Y-%m").alias("Month"),
            pl.col(OUTCOME_TIME).dt.year().cast(pl.Utf8).alias("Year"),
            (pl.col(OUTCOME_TIME).dt.year().cast(pl.Utf8) + "_Q" +
             pl.col(OUTCOME_TIME).dt.quarter().cast(pl.Utf8)).alias("Quarter"),
            (pl.col(OUTCOME_TIME) - pl.col(DECISION_TIME)).dt.total_seconds().alias("ResponseTime")
        ])
        .sort(DECISION_TIME, descending=True)
        .drop(DROP_IH_COLUMNS, strict=False)
        .collect()
    )
    df = df.select(sorted(df.columns))
    schema_df = schema_with_unique_counts(df).sort('Column')
    st.subheader("Schema", divider=True)
    st.data_editor(schema_df,
                   width='stretch',
                   disabled=True, height=300, hide_index=True)
    st.subheader("Data Summary", divider=True)
    st.data_editor(df.describe(), width='stretch', disabled=True, hide_index=True)

    with st.expander("View Data Sample", expanded=False, icon=":material/analytics:"):
        st.dataframe(df.head(100))

    with pl.Config(tbl_cols=len(schema_df), tbl_rows=len(schema_df)):
        prompt = f"""
        You generate only valid TOML configuration for the Value Dashboard application.

        Task:
        Create a configuration file from the provided template and dataset schema.
        Use the template as the structural baseline, but keep only metrics, fields, and reports that can be mapped safely to the provided schema.

        Available inputs:
        - File name: {str(uploaded_file.name)}
        - Dataset schema: {schema_df}
        - Template config file: {tomlkit.dumps(template_config)}
        - Columns that must not be used: {capitalize(DROP_IH_COLUMNS)}

        Derived fields that are available even if not present in the original file:
        - Day
        - Month
        - Year
        - Quarter
        - ResponseTime

        Hard rules:
        1. Output valid TOML only.
        2. Do not include markdown, explanations, comments, or code fences.
        3. Do not generate the [chat_with_data] section.
        4. Keep the same top-level TOML structure as the template whenever possible.
        5. Never invent new columns, but you can remove column from metric if it's not available in the dataset schema.
        6. Only use:
           - columns present in the dataset schema
           - derived fields: Day, Month, Year, Quarter, ResponseTime
        7. If report cannot be mapped confidently, omit it instead of guessing.
        8. Keep metric names and report keys unchanged when they are retained.
        9. Preserve the template style and value types as much as possible.
        10. Do not use any column listed in: {capitalize(DROP_IH_COLUMNS)}

        Column mapping rules:
        1. Match columns case-insensitively first.
        2. If no exact match exists, allow safe prefix/suffix variations.
        3. Prefer semantically obvious matches over approximate text similarity.
        4. Preserve the actual column spelling from the schema in the final TOML.
        5. Treat identifier columns (names ending with ID) as identifiers, not grouping dimensions, unless they are explicitly required for CLV settings.
        6. Prefer low-cardinality business dimensions for filters and group_by fields.
        7. Prefer categorical columns with unique count greater than 1 and less than 100 for filters and group_by.
        8. Keep Outcome-like columns available for descriptive and funnel reports.
        9. Keep numeric columns available for descriptive summaries and CLV settings.

        File settings rules:
        1. Set ih.file_type to "parquet" only if the uploaded file name ends with ".parquet".
        2. Otherwise set ih.file_type to "pega_ds_export".
        3. Set ih.file_pattern to "**/*.parquet" for parquet input.
        4. Set ih.file_pattern to "**/*.json" for non-parquet input.

        Metrics rules:
        1. Keep metrics.global_filters populated only with safe low-cardinality business dimensions.
        2. For each metric, include Day, Month, Year, Quarter in metrics.<metric>.group_by.
        3. Also include selected global_filters in metrics.<metric>.group_by.
        4. Add additional categorical business columns to metrics.<metric>.group_by only if they are present in schema and suitable for grouping.
        5. Remove metric-specific fields that cannot be mapped safely.
        7. Keep all metric definitions, but ensure grouping fields exist.

        Suggested semantic mappings:
        - Channel -> business channel field
        - PlacementType -> placement / placement type field
        - Issue -> issue / goal / business objective
        - Group -> business group / action group / product group field
        - PropensitySource -> model name field
        - Outcome -> outcome / response / result field
        - ExperimentName -> experiment name field
        - ExperimentGroup -> experiment variant / test-control group field
        - PurchasedDateTime -> purchase date / transaction timestamp field
        - CustomerID -> customer identifier
        - HoldingID -> order / holding / contract identifier
        - OneTimeCost -> monetary / revenue / amount field

        Descriptive metric rules:
        1. metrics.descriptive.columns must contain only columns that exist in schema or derived fields.
        2. Prefer these when available: Outcome, Propensity, FinalPropensity, Priority, ResponseTime, Weight, OutcomeWeight.
        3. Keep only columns that are meaningful for descriptive summaries.

        Report retention rules:
        1. Keep a report only if all required fields can be mapped safely.
        2. Remove any report that references a missing or ambiguous field.
        3. Create new reports if there are new business dimensions/group by fields identified. 
        4. Keep report descriptions from the template for retained reports.
        5. Keep generic reports for retained metrics even if specialized reports are dropped.

        Report field requirements:
        - line: requires x and y
        - heatmap: requires x, y, color
        - scatter: requires x, y, size, color, animation_frame, animation_group
        - treemap: requires group_by; also requires color when used by the template
        - gauge: requires value and valid group_by
        - bar_polar: requires r, theta, color
        - boxplot: requires x and y
        - histogram: requires x
        - funnel: requires x, color, stages
        - experiment z-score report: requires x="z_score" and y
        - experiment odds-ratio report: requires x mapped to a valid odds-ratio statistic and y
        - corr: requires x and y

        Metric-specific report rules:
        1. engagement reports may use line, gauge, treemap, heatmap, scatter, bar_polar.
        2. conversion reports may use line, gauge, treemap, heatmap, scatter, bar_polar.
        3. model_ml_scores reports may use line, treemap, heatmap, scatter.
        4. descriptive reports may use line, boxplot, histogram, heatmap, funnel.
        5. experiment reports may use z-score and odds-ratio style reports only when experiment fields are available.
        6. clv reports may use histogram, treemap, exposure, corr, model, rfm_density only when CLV fields are available.

        Output requirements:
        1. Return a complete valid TOML document.
        2. Exclude [chat_with_data].
        3. Keep retained sections in the same logical order as the template.
        4. Do not include any field, metric, report, or group_by value that is not supported by the provided schema or derived fields.

        Final self-check before output:
        1. Every referenced column exists in schema or is one of Day, Month, Year, Quarter, ResponseTime.
        2. Every report metric exists in [metrics].
        3. Every retained report has all required fields.
        4. No omitted field remains referenced anywhere else.
        5. Output is valid TOML only.
        """

        st.write("## Config from sample")
        if st.button("Generate config", key='UploadBtn', type='primary'):
            logger.debug('LLM prompt: ' + prompt)
            with st.spinner("Generating config. Wait for it...", show_time=True):
                logger.info('LLM call: ' + prompt)
                generate_new_config(llm, prompt)
