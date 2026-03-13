import os
import re
import tomllib
from copy import deepcopy

import polars as pl
import streamlit as st
import tomlkit

from value_dashboard.config_generator.ai import build_ai_config_prompt, build_ai_reports_refinement_prompt, \
    build_final_config, generate_ai_sections, save_generated_config
from value_dashboard.config_generator.preprocess import apply_ih_preprocessing, build_ih_config, build_schema_preview, \
    build_calculated_fields_config_text, compile_filter_rules, detect_ih_file_settings, load_ih_sample
from value_dashboard.metrics.constants import DECISION_TIME, OUTCOME_TIME, REQ_IH_COLUMNS
from value_dashboard.report_builder import render_report_builder
from value_dashboard.utils.config_builder import render_section, render_value, serialize_exprs
from value_dashboard.utils.llm_utils import render_litellm_sidebar

st.set_page_config(page_title="✨AI Configuration Studio", layout="wide")

FILTER_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "contains", "starts with", "in", "not in", "is null",
                    "is not null"]
IH_FILE_TYPES = ["parquet", "pega_ds_export"]
AI_SCHEMA_EXAMPLE_COLUMNS = ["Most occurring", "Values"]
HIVE_PARTITIONING_HELP = (
    "For parquet scans, infer statistics and schema from hive-partitioned paths and use them to prune reads."
)
STREAMING_HELP = (
    "Prefer Polars' streaming engine during collection instead of the default auto engine."
)
BACKGROUND_HELP = (
    "Run collection in the background and return a handle that can fetch or cancel the result. "
    "Polars marks background mode as unstable."
)
CONFIG_STUDIO_PRESERVE_KEYS = {
    "config_studio_api_key",
    "config_studio_model",
    "config_studio_reasoning_effort",
    "config_studio_verbosity",
}
STEP_OPTIONS = [
    "1. Sample",
    "2. Required Fields",
    "3. Defaults",
    "4. Filters",
    "5. Calculations",
    "6. Approve",
    "7. AI Draft",
    "8. Metrics",
    "9. AI Reports",
    "10. Reports",
    "11. Chat with Data",
    "12. Save/Export",
]


def _load_template_config() -> dict:
    package_dir = os.path.dirname(__file__)
    config_path = os.path.join(package_dir, "../config", "config_template.toml")
    with open(config_path, mode="rb") as handle:
        return tomllib.load(handle)


def _default_defaults_rows(template_config: dict) -> list[dict]:
    default_values = template_config.get("ih", {}).get("extensions", {}).get("default_values", {})
    return [
        {"Field": key, "Default Value": value, "Enabled": True}
        for key, value in default_values.items()
    ] or [_blank_default_row()]


def _default_filter_rows() -> list[dict]:
    return [_blank_filter_row()]


def _default_calculated_rows() -> list[dict]:
    return [_blank_calculated_row()]


def _blank_default_row() -> dict:
    return {"Field": "", "Default Value": "", "Enabled": True}


def _blank_filter_row() -> dict:
    return {"Field": "", "Operator": "==", "Value": "", "Enabled": True}


def _blank_calculated_row() -> dict:
    return {"Name": "", "Expression": "", "Enabled": True}


def _frame_records(frame) -> list[dict]:
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    return frame.to_dict("records")


def _is_missing_editor_value(value) -> bool:
    return value is None or value != value


def _normalize_rows(frame) -> list[dict]:
    rows = _frame_records(frame)
    cleaned = []
    for row in rows:
        cleaned.append({
            key: ("" if _is_missing_editor_value(value) else value)
            for key, value in row.items()
        })
    return cleaned


def _editor_frame(rows: list[dict], columns: list[str], blank_row_factory) -> pl.DataFrame:
    """Keep editor tables usable even if the user deletes every row."""
    editor_rows = rows or [blank_row_factory()]
    return pl.DataFrame({
        column: [
            (
                bool(row.get(column, False))
                if column == "Enabled"
                else ("" if _is_missing_editor_value(row.get(column, "")) else str(row.get(column, "")))
            )
            for row in editor_rows
        ]
        for column in columns
    })


def _append_editor_row(state_key: str, blank_row_factory):
    """Append one editable row to a session-backed table editor."""
    rows = list(st.session_state.get(state_key) or [])
    rows.append(blank_row_factory())
    st.session_state[state_key] = rows


def _default_time_column(columns: list[str], preferred: str) -> str:
    if preferred in columns:
        return preferred
    for column in columns:
        if preferred.lower() in column.lower():
            return column
    return columns[0] if columns else ""


def _default_subject_id_column(columns: list[str]) -> str:
    if "SubjectID" in columns:
        return "SubjectID"
    preferred_names = ["CustomerID", "CustomerId", "SubjectId", "CustomerKey", "Customer"]
    lower_to_original = {column.casefold(): column for column in columns}
    for preferred_name in preferred_names:
        match = lower_to_original.get(preferred_name.casefold())
        if match:
            return match
    for column in columns:
        column_lower = column.casefold()
        if "subject" in column_lower and "id" in column_lower:
            return column
    for column in columns:
        column_lower = column.casefold()
        if "customer" in column_lower and "id" in column_lower:
            return column
    for column in columns:
        if column.casefold().endswith("id"):
            return column
    return columns[0] if columns else ""


def _build_effective_calculated_rows(
        source_columns: list[str],
        default_values: dict[str, object],
        calculated_rows: list[dict],
) -> list[dict]:
    effective_rows = list(calculated_rows)
    has_subject_id = "SubjectID" in source_columns or "SubjectID" in default_values
    has_custom_subject_id = any(
        row.get("Enabled", True) and str(row.get("Name", "")).strip() == "SubjectID"
        for row in effective_rows
    )
    subject_id_source = st.session_state.get("config_studio_subject_id_source", "")
    if has_subject_id or has_custom_subject_id or not subject_id_source or subject_id_source == "SubjectID":
        return effective_rows
    return [
        {"Name": "SubjectID", "Expression": f'pl.col("{subject_id_source}")', "Enabled": True},
        *effective_rows,
    ]


def _sync_ai_example_field_selection(selected_fields: list[str]) -> list[str]:
    """Keep the example-sharing selection aligned with the currently approved field set."""
    selected_set = set(selected_fields)
    current_example_set = set(st.session_state.get("config_studio_ai_example_fields") or [])
    if "config_studio_ai_example_fields" not in st.session_state:
        synced_fields = sorted(selected_fields, key=str.casefold)
    else:
        synced_fields = sorted((current_example_set & selected_set), key=str.casefold)
    st.session_state["config_studio_ai_example_fields"] = synced_fields
    return synced_fields


def _mask_schema_preview_for_ai(schema_preview, example_fields: list[str]):
    """Hide sample values from AI for rows where the user has opted out of sharing examples."""
    preview_df = schema_preview if isinstance(schema_preview, pl.DataFrame) else pl.from_dicts(
        _frame_records(schema_preview))
    allowed_fields = set(example_fields)
    if "Column" not in preview_df.columns or not AI_SCHEMA_EXAMPLE_COLUMNS:
        return schema_preview
    available_example_columns = [column_name for column_name in AI_SCHEMA_EXAMPLE_COLUMNS if
                                 column_name in preview_df.columns]
    if not available_example_columns:
        return preview_df
    keep_examples = pl.col("Column").is_in(sorted(allowed_fields, key=str.casefold)) if allowed_fields else pl.lit(
        False)
    return preview_df.with_columns([
        pl.when(keep_examples).then(pl.col(column_name)).otherwise(pl.lit("")).alias(column_name)
        for column_name in available_example_columns
    ])


@st.fragment()
def _render_ai_schema_preview_table(schema_editor_df) -> None:
    """Render the AI schema preview editor and return the masked dataframe sent to AI."""
    edited_schema_preview = st.data_editor(
        schema_editor_df,
        width="stretch",
        hide_index=True,
        height=480,
        key="config_studio_ai_schema_preview_editor",
        disabled=[column for column in schema_editor_df.columns if column != "Send To AI"],
        column_config={
            "Send To AI": st.column_config.CheckboxColumn(
                "Sample To AI",
                help="Uncheck a row to hide **Most occurring** and **Values** for this field in the AI prompt.",
                width="small",
            ),
        },
    )
    st.session_state["config_studio_ai_example_fields"] = sorted(
        [
            str(row.get("Column", ""))
            for row in _frame_records(edited_schema_preview)
            if row.get("Send To AI")
        ],
        key=str.casefold,
    )


def _apply_ih_runtime_settings(ih_config: dict) -> dict:
    """Overlay the studio IH runtime controls onto the generated ih config."""
    ih_config["file_type"] = st.session_state.get("config_studio_file_type", ih_config.get("file_type", ""))
    ih_config["file_pattern"] = st.session_state.get("config_studio_file_pattern", ih_config.get("file_pattern", ""))
    ih_config["ih_group_pattern"] = st.session_state.get(
        "config_studio_ih_group_pattern", ih_config.get("ih_group_pattern", "")
    )
    ih_config["hive_partitioning"] = st.session_state.get(
        "config_studio_hive_partitioning", bool(ih_config.get("hive_partitioning", False))
    )
    ih_config["streaming"] = st.session_state.get(
        "config_studio_streaming", bool(ih_config.get("streaming", False))
    )
    ih_config["background"] = st.session_state.get(
        "config_studio_background", bool(ih_config.get("background", False))
    )
    return ih_config


def _reset_report_builder_state():
    """Clear shared report-builder widget state when the studio draft changes."""
    for key in ("rb_selected_report", "rb_draft_report", "rb_editor_token", "rb_search", "rb_metric_filter",
                "rb_type_filter", "rb_selected_report_picker"):
        if key in st.session_state:
            del st.session_state[key]


def _clear_config_studio_state():
    """Drop all file-specific Config Studio state before initializing a new IH sample."""
    for key in list(st.session_state.keys()):
        if key.startswith("config_studio") and key not in CONFIG_STUDIO_PRESERVE_KEYS:
            del st.session_state[key]
    _reset_report_builder_state()


def _set_step(step_label: str):
    """Queue a programmatic step change for the next rerun."""
    st.session_state["config_studio_step"] = step_label
    st.session_state["config_studio_pending_step"] = step_label


def _set_draft_config(config: dict):
    """Store the mutable generated config that will be reviewed in later steps."""
    st.session_state["config_studio_draft_config"] = deepcopy(config)
    st.session_state["config_studio_generated_toml"] = tomlkit.dumps(serialize_exprs(deepcopy(config)))


def _initialize_editor_state(template_config: dict, file_signature: str, sample_columns: list[str], file_name: str):
    if st.session_state.get("config_studio_file_signature") == file_signature:
        return
    _clear_config_studio_state()
    template_ih = template_config.get("ih", {})
    detected_file_type, detected_file_pattern = detect_ih_file_settings(file_name)
    st.session_state["config_studio_file_signature"] = file_signature
    st.session_state["config_studio_defaults_rows"] = _default_defaults_rows(template_config)
    st.session_state["config_studio_filter_rows"] = _default_filter_rows()
    st.session_state["config_studio_calculated_rows"] = _default_calculated_rows()
    st.session_state["config_studio_filter_mode"] = "Rules"
    st.session_state["config_studio_raw_filter"] = ""
    st.session_state["config_studio_file_type"] = detected_file_type
    st.session_state["config_studio_file_pattern"] = detected_file_pattern
    st.session_state["config_studio_ih_group_pattern"] = template_ih.get("ih_group_pattern", "")
    st.session_state["config_studio_hive_partitioning"] = bool(template_ih.get("hive_partitioning", False))
    st.session_state["config_studio_streaming"] = bool(template_ih.get("streaming", False))
    st.session_state["config_studio_background"] = bool(template_ih.get("background", False))
    st.session_state["config_studio_outcome_time"] = _default_time_column(sample_columns, OUTCOME_TIME)
    st.session_state["config_studio_decision_time"] = _default_time_column(sample_columns, DECISION_TIME)
    st.session_state["config_studio_subject_id_source"] = _default_subject_id_column(sample_columns)
    st.session_state["config_studio_step"] = STEP_OPTIONS[0]
    st.session_state["config_studio_step_selector"] = STEP_OPTIONS[0]
    st.session_state["config_studio_pending_step"] = None
    st.session_state["config_studio_ai_sections"] = None
    st.session_state["config_studio_ai_report_sections"] = None
    st.session_state["config_studio_generated_toml"] = None
    st.session_state["config_studio_draft_config"] = None
    st.session_state["config_studio_selected_fields"] = []


def _build_default_values_map(default_rows: list[dict]) -> dict:
    result = {}
    for row in default_rows:
        if not row.get("Enabled", True):
            continue
        field_name = str(row.get("Field", "")).strip()
        if not field_name:
            continue
        result[field_name] = row.get("Default Value", "")
    return result


def _get_missing_required_default_fields(sample_columns: list[str], default_rows: list[dict]) -> list[str]:
    """Identify required IH fields that are absent from the raw dataset and not yet covered by defaults."""
    default_values = _build_default_values_map(default_rows)
    subject_id_source = str(st.session_state.get("config_studio_subject_id_source", "")).strip()
    missing_fields = []
    for field_name in REQ_IH_COLUMNS:
        if field_name in sample_columns:
            continue
        if field_name == "SubjectID" and subject_id_source:
            continue
        if field_name in default_values:
            continue
        missing_fields.append(field_name)
    return missing_fields


def _extract_filter_fields(filter_expression: str) -> list[str]:
    """Extract field names referenced via pl.col("...") in a Polars filter expression."""
    if not filter_expression:
        return []
    pattern = r"""pl\.col\(\s*["']([^"']+)["']\s*\)"""
    return re.findall(pattern, filter_expression)


def _render_intro():
    st.header("✨AI Configuration Studio", divider='red')
    st.info(
        "Build preprocessing first, approve the working field catalog, then let AI draft metrics and reports from the cleaned schema. Derived time fields are surfaced early so the field-approval step reflects the real reporting surface, not just the raw upload.")


def _render_metrics_bar(sample_df, working_df, approved_fields: list[str]):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Columns", len(sample_df.columns))
    col2.metric("Working Columns", len(working_df.columns))
    col3.metric("Approved Fields", len(approved_fields))
    col4.metric("Rows In Sample", f"{working_df.height:,}")


def _render_sample_step(sample_df, file_name: str):
    with st.container(border=True):
        title_col, action_col = st.columns([0.8, 0.2])
        with title_col:
            st.write("### IH Runtime Settings")
            st.caption(
                "These controls define how the runtime reads IH files before defaults, filters, and calculated fields are applied.")
        with action_col:
            with st.popover("Runtime Behaviour", icon=":material/flare:"):
                st.markdown("- Defaults run before filters.")
                st.markdown("- The IH filter runs before calculated fields.")
                st.markdown("- Day / Month / Year / Quarter / ResponseTime are derived before field approval.")
                st.markdown("- Approved fields affect AI, not ingestion.")
        runtime_col1, runtime_col2, runtime_col3, runtime_col4, runtime_col5, runtime_col6 = st.columns(6)
        st.session_state["config_studio_file_type"] = runtime_col1.selectbox(
            "File Type",
            options=IH_FILE_TYPES,
            index=IH_FILE_TYPES.index(st.session_state["config_studio_file_type"])
            if st.session_state.get("config_studio_file_type") in IH_FILE_TYPES else 0,
            help="Choose the IH runtime reader. Supported values in this app are `parquet` and `pega_ds_export`.",
            key="config_studio_file_type_w",
        )
        st.session_state["config_studio_ih_group_pattern"] = runtime_col2.text_input(
            "IH Group Pattern",
            value=st.session_state.get("config_studio_ih_group_pattern", ""),
            help="Regex used by the runtime to derive IH file groups from file names.",
            key="config_studio_ih_group_pattern_w",
        )
        st.session_state["config_studio_file_pattern"] = runtime_col3.text_input(
            "File Pattern",
            value=st.session_state.get("config_studio_file_pattern", ""),
            help="Glob pattern used by the runtime to find IH files.",
            key="config_studio_file_pattern_w",
        )
        st.session_state["config_studio_hive_partitioning"] = runtime_col4.checkbox(
            "Hive Partitioning",
            value=st.session_state.get("config_studio_hive_partitioning", False),
            help=HIVE_PARTITIONING_HELP,
            key="config_studio_hive_partitioning_w",
        )
        st.session_state["config_studio_streaming"] = runtime_col5.checkbox(
            "Streaming",
            value=st.session_state.get("config_studio_streaming", False),
            help=STREAMING_HELP,
            key="config_studio_streaming_w",
        )
        st.session_state["config_studio_background"] = runtime_col6.checkbox(
            "Background",
            value=st.session_state.get("config_studio_background", False),
            help=BACKGROUND_HELP,
            key="config_studio_background_w",
        )

    with st.container(border=True):
        st.write("### Raw Schema")
        raw_schema = build_schema_preview(sample_df)
        st.dataframe(raw_schema, width="stretch", hide_index=True, height=360)
    with st.expander("Peek at raw sample rows", expanded=False, icon=":material/table_view:"):
        st.dataframe(sample_df.head(100), width="stretch", height=320)


def _render_required_fields_step(sample_df):
    st.write("### Required Field Mapping")
    st.caption(
        "Confirm the required identity and time fields before defaults, filters, and calculated fields are applied."
    )
    columns = sample_df.columns
    with st.container(border=True):
        st.write("#### SubjectID Mapping")
        if "SubjectID" in columns:
            st.success("`SubjectID` is already present in the uploaded dataset.")
            st.session_state["config_studio_subject_id_source"] = "SubjectID"
        else:
            st.selectbox(
                "SubjectID Source Column",
                options=columns,
                index=columns.index(st.session_state["config_studio_subject_id_source"])
                if st.session_state.get("config_studio_subject_id_source") in columns else 0,
                help=(
                    "Choose the source column that should be aliased to `SubjectID` for the working schema and the "
                    "generated IH config."
                ),
                key="config_studio_subject_id_source",
            )
            st.info(
                f"`{st.session_state['config_studio_subject_id_source']}` will be aliased to `SubjectID` during "
                "studio preprocessing and in the generated `[ih.extensions.columns]` config."
            )
    with st.container(border=True):
        st.write("#### Time Fields")
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            st.selectbox(
                "Outcome Time Column",
                options=columns,
                index=columns.index(st.session_state["config_studio_outcome_time"])
                if st.session_state["config_studio_outcome_time"] in columns else 0,
                help="Choose the source timestamp used to derive Day, Month, Year, Quarter, and ResponseTime.",
                key="config_studio_outcome_time"
            )
        with time_col2:
            st.selectbox(
                "Decision Time Column",
                options=columns,
                index=columns.index(st.session_state["config_studio_decision_time"])
                if st.session_state["config_studio_decision_time"] in columns else 0,
                help="Choose the source timestamp used as the baseline for ResponseTime.",
                key="config_studio_decision_time"
            )
        st.info(
            "These source columns will be aliased to `OutcomeTime` and `DecisionTime` inside the working schema."
        )


@st.fragment()
def _render_preprocess_settings_step(default_frame: pl.DataFrame, sample_columns: list[str]):
    with st.container(border=True):
        title_col, _ = st.columns([0.8, 0.2], vertical_alignment="center")
        with title_col:
            st.write("### Default Column Values")
        current_default_rows = st.session_state.get("config_studio_defaults_rows") or _normalize_rows(default_frame)
        missing_required_fields = _get_missing_required_default_fields(sample_columns, current_default_rows)
        if missing_required_fields:
            st.warning(
                "Required fields missing from the uploaded dataset should be provided with defaults here: `"
                + ", ".join(missing_required_fields)
                + "`."
            )
        st.caption("Defaults are applied before filters. They may fill nulls or create missing columns.")
        edited_defaults = st.data_editor(
            default_frame,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="config_studio_defaults_editor",
            column_config={
                "Field": st.column_config.TextColumn("Field", help="Column to fill or create.", width="medium"),
                "Default Value": st.column_config.TextColumn(
                    "Default Value", help="Literal value. Examples: N/A, 0.0, true, 1e-10.",
                    width="medium"
                ),
                "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
            },
        )
        st.session_state["config_studio_defaults_rows"] = _normalize_rows(edited_defaults)


@st.fragment()
def _render_filter_step(filter_field_options: list[str], filter_rows_frame: pl.DataFrame):
    with st.container(border=True):
        st.write("### Filters")
        st.caption("Define the dataset-level filters before derived and calculated fields are added.")
        st.session_state["config_studio_filter_mode"] = st.segmented_control(
            "Filter Mode",
            options=["Rules", "Raw Polars"],
            default=st.session_state["config_studio_filter_mode"],
            selection_mode="single",
            help="Rules are easier to author; raw mode gives full control over the final ih.extensions.filter expression.",
        )
        if st.session_state["config_studio_filter_mode"] == "Rules":
            edited_filters = st.data_editor(
                filter_rows_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="config_studio_filter_editor",
                column_config={
                    "Field": st.column_config.SelectboxColumn(
                        "Field", options=filter_field_options, required=False, width="medium"
                    ),
                    "Operator": st.column_config.SelectboxColumn(
                        "Operator", options=FILTER_OPERATORS, required=False, width="small"
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value", help="Comma-separated values for in/not in.", width="large"
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                },
            )
            st.session_state["config_studio_filter_rows"] = _normalize_rows(edited_filters)
            compiled_filter = compile_filter_rules(st.session_state["config_studio_filter_rows"])
            st.caption("Compiled filter")
            st.code(compiled_filter or "pl.lit(True)", language="python", wrap_lines=True, line_numbers=True)
        else:
            st.session_state["config_studio_raw_filter"] = st.text_area(
                "Raw Polars Filter",
                value=st.session_state["config_studio_raw_filter"],
                height=220,
                help="Full Polars predicate expression. This filter runs before calculated fields, like the runtime IH pipeline.",
            )
            st.caption("Example: pl.col(\"Outcome\").is_in([\"Pending\", \"Impression\", \"Clicked\"])")


@st.fragment()
def _render_calculated_fields_step(calculated_frame: pl.DataFrame):
    with st.container(border=True):
        title_col, _, example_col = st.columns([0.4, 0.3, 0.3], vertical_alignment="center")
        with title_col:
            st.write("### Calculated Fields")
        with example_col:
            with st.popover("Examples", icon=":material/flare:"):
                st.code(
                    """pl.when(pl.col("CustomerID").str.slice(0, 1) == "C").then(pl.lit("Customers known")).otherwise(pl.lit("Device/Anonymous")).alias("CustomerType")""",
                    language="python",
                )
                st.caption("Enter the expression body. The studio will alias it to the field name automatically.")
        edited_calculated = st.data_editor(
            calculated_frame,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="config_studio_calculated_editor",
            column_config={
                "Name": st.column_config.TextColumn("Name", help="Name of the generated field.", width="small"),
                "Expression": st.column_config.TextColumn(
                    "Expression",
                    help="Polars expression body using pl and np. Do not add .alias(...) unless you want to fully control the expression.",
                    width="large"
                ),
                "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
            },
        )
        st.session_state["config_studio_calculated_rows"] = _normalize_rows(edited_calculated)
        compiled_calc = build_calculated_fields_config_text(st.session_state["config_studio_calculated_rows"])
        st.caption("Compiled calculated fields")
        st.code(compiled_calc, language="python", wrap_lines=True, line_numbers=True)


@st.fragment()
def _render_field_step(working_df):
    available_fields = sorted(working_df.columns, key=str.casefold)
    current_selection = st.session_state.get("config_studio_selected_fields") or list(available_fields)
    current_selection = sorted(
        [field for field in current_selection if field in available_fields],
        key=str.casefold,
    )
    missing_required_fields = sorted([field for field in REQ_IH_COLUMNS if field not in available_fields],
                                     key=str.casefold)
    required_fields = sorted([field for field in REQ_IH_COLUMNS if field in available_fields], key=str.casefold)
    optional_fields = [field for field in available_fields if field not in required_fields]
    current_optional_selection = [field for field in current_selection if field in optional_fields]
    with st.container(border=True):
        st.write("### Approved Fields Catalog")
        st.caption("Only approved fields will be exposed to the AI stage.")
        if missing_required_fields:
            st.warning(
                "Required columns missing from the current working dataset: `"
                + ", ".join(missing_required_fields)
                + "` . Add defaults, filters, calculated fields, or the SubjectID mapping before continuing."
            )
        if required_fields:
            st.caption(
                "Required IH fields are always included: " + ", ".join(required_fields)
            )
        if optional_fields:
            selected_optional_fields = st.pills(
                "Optional Fields Available To AI",
                options=optional_fields,
                default=current_optional_selection,
                selection_mode="multi",
                help=(
                    "Choose the optional fields AI may use after defaults, filters, derived time fields, and "
                    "calculated fields are applied. Required IH fields stay locked in."
                ),
                key="selected_optional_fields"
            )
        else:
            selected_optional_fields = []
            st.info("Only required IH fields are available in the current working schema.")
        selected_fields = sorted(required_fields + selected_optional_fields, key=str.casefold)
        st.session_state["config_studio_selected_fields"] = selected_fields
    with st.container(border=True):
        st.subheader("Approve Data Sharing with AI", help="Review schema and data samples sent to AI.")
        st.caption("This is the post-processed schema. Derived time fields are already visible here.")
        schema_preview = build_schema_preview(working_df, selected_fields)
        selected_ai_example_fields = _sync_ai_example_field_selection(selected_fields)
        schema_editor_df = schema_preview.with_columns(
            pl.col("Column").is_in(selected_ai_example_fields).alias("Send To AI")
        ).select(["Send To AI", *schema_preview.columns])
        _render_ai_schema_preview_table(schema_editor_df)
    with st.expander("Preview working sample after preprocessing", expanded=False, icon=":material/preview:"):
        preview_fields = st.session_state["config_studio_selected_fields"] or available_fields
        st.dataframe(
            working_df.select([field for field in preview_fields if field in working_df.columns]).head(100),
            width="stretch",
        )


def _render_ai_step(template_config: dict, file_name: str, working_df: pl.DataFrame, schema_preview: pl.DataFrame,
                    default_values: dict, filter_expression: str, calculated_fields_text: str, llm,
                    preprocessing_error: str | None = None):
    st.write("### AI Configuration Draft")
    st.write("The AI sees only the approved working schema and the final IH preprocessing config.")
    if preprocessing_error:
        st.warning("Resolve preprocessing issues before generating an AI draft.")
        return
    approved_fields = st.session_state.get("config_studio_selected_fields") or list(working_df.columns)
    ih_config = build_ih_config(
        template_config=template_config,
        file_name=file_name,
        default_values=default_values,
        filter_expression=filter_expression,
        calculated_fields_text=calculated_fields_text,
    )
    ih_config = _apply_ih_runtime_settings(ih_config)
    prep_col1, prep_col2 = st.columns([0.5, 0.5])
    with prep_col1:
        with st.container(border=True):
            st.subheader("IH Config", help="Review schema and data samples sent to AI.", divider=True)
            st.code(tomlkit.dumps({"ih": ih_config}), language="toml", height=480)
    with prep_col2:
        with st.container(border=True):
            st.subheader("Approved Schema Preview", help="Review schema and data samples sent to AI.", divider=True)
            ai_schema_preview = _mask_schema_preview_for_ai(schema_preview,
                                                            st.session_state["config_studio_ai_example_fields"])
            st.dataframe(ai_schema_preview, width="stretch", height=480)

    if not llm:
        st.info("Configure an API key in the sidebar when you are ready to generate metrics and reports.")
        return

    prompt = build_ai_config_prompt(
        file_name=file_name,
        approved_schema=ai_schema_preview,
        approved_fields=approved_fields,
        template_config=template_config,
        ih_config=ih_config,
    )

    action_col1, action_col2 = st.columns([0.35, 0.65], vertical_alignment="center")
    with action_col1:
        if st.button("Generate AI Draft", type="primary", key="config_studio_generate_ai"):
            try:
                with st.status("Generating metrics and reports", expanded=True) as status:
                    status.write("Building AI prompt from the approved schema...")
                    sections = generate_ai_sections(llm, prompt)
                    final_config = build_final_config(template_config, ih_config, sections)
                    st.session_state["config_studio_ai_sections"] = sections
                    _set_draft_config(final_config)
                    _reset_report_builder_state()
                    status.write("AI sections parsed successfully.")
                    status.update(label="Draft ready", state="complete")
                _set_step(STEP_OPTIONS[7])
                st.rerun()
            except Exception as exc:
                st.error(f"AI draft generation failed: {exc}")
    with action_col2:
        with st.popover("Show AI prompt", icon=":material/psychology:"):
            st.code(prompt, language="text")

    draft_config = st.session_state.get("config_studio_draft_config")
    if draft_config is not None:
        draft_metrics = draft_config.get("metrics", {})
        draft_reports = draft_config.get("reports", {})
        st.success(
            f"Draft ready: {len(draft_metrics)} metrics and {len(draft_reports)} reports. "
            f"Continue with `8. Metrics`, `9. AI Reports`, and `10. Reports` before exporting."
        )


def _render_metrics_step():
    """Review the AI-generated metrics using the same widgets as the TOML editor."""
    cfg = st.session_state.get("config_studio_draft_config")
    if cfg is None:
        st.info("Generate an AI draft first.")
        return

    st.write("### Metrics Review")
    st.caption(
        "Adjust metric-level grouping, filters, and response mappings before report review. Only these fields may be used for metrics.global_filters, metric group_by, and metric filters.")
    approved_fields = sorted(st.session_state.get("config_studio_selected_fields") or [], key=str.casefold)

    metrics = cfg.get("metrics", {})
    updated_metrics = {}
    current_global_filters = [field for field in metrics.get("global_filters", []) if field in approved_fields]
    dropped_global_filters = [field for field in metrics.get("global_filters", []) if field not in approved_fields]
    with st.container(border=True):
        st.subheader("global_filters")
        st.caption("Choose the approved fields that should appear as top-level metric filters.")
        if dropped_global_filters:
            st.warning(
                "Removed unsupported global filters: " + ", ".join(sorted(dropped_global_filters, key=str.casefold))
            )
        updated_metrics["global_filters"] = st.multiselect(
            "Global Filters",
            options=approved_fields,
            default=current_global_filters,
            key="config_studio_metrics_global_filters",
            help="Only approved fields can be exposed as global filters in the generated config.",
        )

    field_reference_keys = {
        "experiment_name",
        "experiment_group",
        "order_id_col",
        "customer_id_col",
        "monetary_value_col",
        "purchase_date_col",
        "recurring_period",
        "recurring_cost",
    }

    for key, value in metrics.items():
        if key == "global_filters":
            continue
        st.subheader(key)
        if isinstance(value, dict):
            with st.container(border=True):
                updated_metric = {}
                for metric_key, metric_value in value.items():
                    widget_path = f"config_studio.metrics.{key}"
                    if metric_key == "scores":
                        st.markdown(f"**{metric_key}**: _not editable_")
                        st.json(metric_value)
                        updated_metric[metric_key] = metric_value
                    elif metric_key == "group_by":
                        current_group_by = [field for field in metric_value if field in approved_fields]
                        dropped_group_by = [field for field in metric_value if field not in approved_fields]
                        if dropped_group_by:
                            st.warning(
                                f"{key}: removed unsupported group_by fields: "
                                + ", ".join(sorted(dropped_group_by, key=str.casefold))
                            )
                        updated_metric[metric_key] = st.multiselect(
                            f"{key}.group_by",
                            options=approved_fields,
                            default=current_group_by,
                            key=f"{widget_path}.group_by",
                            help="Only approved fields can be used in metric group_by.",
                        )
                    elif metric_key == "columns" and isinstance(metric_value, list):
                        current_columns = [field for field in metric_value if field in approved_fields]
                        dropped_columns = [field for field in metric_value if field not in approved_fields]
                        if dropped_columns:
                            st.warning(
                                f"{key}: removed unsupported columns: "
                                + ", ".join(sorted(dropped_columns, key=str.casefold))
                            )
                        updated_metric[metric_key] = st.multiselect(
                            f"{key}.columns",
                            options=approved_fields,
                            default=current_columns,
                            key=f"{widget_path}.columns",
                            help="Choose approved descriptive or metric fields only.",
                        )
                    elif metric_key == "filter":
                        filter_key = f"{widget_path}.filter"
                        if f"{filter_key}_draft" not in st.session_state:
                            st.session_state[f"{filter_key}_draft"] = metric_value
                        draft_filter = st.text_area(
                            f"{key}.filter",
                            value=st.session_state[f"{filter_key}_draft"],
                            key=f"{filter_key}_draft",
                            help=(
                                "Use a Polars expression. Any pl.col(...) references must use approved fields only."
                            ),
                            height=160,
                        )
                        referenced_fields = sorted(set(_extract_filter_fields(draft_filter)), key=str.casefold)
                        invalid_fields = [field for field in referenced_fields if field not in approved_fields]
                        if referenced_fields:
                            st.caption("Filter references: " + ", ".join(referenced_fields))
                        if invalid_fields:
                            st.error(
                                "Unapproved fields in filter: " + ", ".join(invalid_fields)
                            )
                            updated_metric[metric_key] = metric_value
                        else:
                            updated_metric[metric_key] = draft_filter
                    elif metric_key in field_reference_keys:
                        if metric_value and metric_value not in approved_fields:
                            st.warning(f"{key}.{metric_key} is not in the approved field catalog and must be updated.")
                        if approved_fields:
                            select_options = approved_fields
                        elif metric_value:
                            select_options = [metric_value]
                        else:
                            select_options = [""]
                        selected_index = (
                            select_options.index(metric_value)
                            if metric_value in select_options else 0
                        )
                        updated_metric[metric_key] = st.selectbox(
                            f"{key}.{metric_key}",
                            options=select_options,
                            index=selected_index,
                            key=f"{widget_path}.{metric_key}",
                            help="This field reference must come from the approved field catalog.",
                        )
                    else:
                        updated_metric[metric_key] = render_value(metric_key, metric_value, widget_path)
                updated_metrics[key] = updated_metric
        else:
            updated_metrics[key] = render_value(key, value, "config_studio.metrics")
    cfg["metrics"] = updated_metrics
    st.info(
        "When you finish metric edits, continue to `9. AI Reports` to refresh the report set from the updated grouping fields.")


def _render_ai_reports_step(file_name: str, working_df: pl.DataFrame, schema_preview: pl.DataFrame, llm, template_config:dict):
    """Regenerate only the reports section after metric edits, keeping the rest of the draft intact."""
    cfg = st.session_state.get("config_studio_draft_config")
    if cfg is None:
        st.info("Generate an AI draft first.")
        return

    st.write("### AI Report Refresh")
    st.caption(
        "Use a second AI pass to rebuild only the `[reports]` section from the current draft metrics and approved schema."
    )
    approved_fields = st.session_state.get("config_studio_selected_fields") or list(working_df.columns)
    prompt = build_ai_reports_refinement_prompt(
        file_name=file_name,
        approved_schema=schema_preview,
        approved_fields=approved_fields,
        current_config=cfg,
        template_config=template_config
    )

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        with st.container(border=True):
            st.write("#### Current Metrics")
            st.metric("Metric Sections", len(cfg.get("metrics", {})))
            st.code(tomlkit.dumps({"metrics": cfg.get("metrics", {})}), language="toml", height=360)
    with summary_col2:
        with st.container(border=True):
            st.write("#### Current Reports")
            st.metric("Report Count", len(cfg.get("reports", {})))
            st.code(tomlkit.dumps({"reports": cfg.get("reports", {})}), language="toml", height=360)

    if not llm:
        st.info("Configure an API key in the sidebar when you are ready to refresh reports from the edited metrics.")
        return

    action_col1, action_col2 = st.columns([0.35, 0.65], vertical_alignment="center")
    with action_col1:
        if st.button("Refresh Reports From Metrics", type="primary", key="config_studio_refresh_reports"):
            try:
                with st.status("Refreshing reports from the current metrics", expanded=True) as status:
                    status.write("Building a report-only AI prompt from the current draft config...")
                    sections = generate_ai_sections(llm, prompt)
                    if "reports" not in sections:
                        raise ValueError("AI response did not include a [reports] section.")
                    updated_cfg = deepcopy(cfg)
                    updated_cfg["reports"] = sections["reports"]
                    st.session_state["config_studio_ai_report_sections"] = sections
                    _set_draft_config(updated_cfg)
                    _reset_report_builder_state()
                    status.write("Reports section parsed successfully.")
                    status.update(label="Reports refreshed", state="complete")
                _set_step(STEP_OPTIONS[9])
                st.rerun()
            except Exception as exc:
                st.error(f"Report refresh failed: {exc}")
    with action_col2:
        with st.popover("Show report refresh prompt", icon=":material/psychology:"):
            st.code(prompt, language="text")

    st.info(
        "This step replaces only `[reports]`. Your current `[metrics]`, preprocessing settings, and other draft sections stay unchanged."
    )


def _render_reports_step():
    """Review, select, and edit reports on top of the current draft config."""
    cfg = st.session_state.get("config_studio_draft_config")
    if cfg is None:
        st.info("Generate an AI draft first.")
        return

    st.write("### Reports Review")
    st.caption("Review the AI-refreshed report set, remove weak reports, and refine the remaining definitions.")
    render_report_builder(cfg)


def _render_chat_step():
    """Review Chat with Data settings after metrics and reports are finalized."""
    cfg = st.session_state.get("config_studio_draft_config")
    if cfg is None:
        st.info("Generate an AI draft first.")
        return

    st.write("### Chat With Data")
    st.caption("Tune the assistant-facing settings and metric descriptions before final export.")
    ux = cfg.get("ux", {})
    if "chat_with_data" in ux:
        ux["chat_with_data"] = render_value("chat_with_data", ux.get("chat_with_data", False), "config_studio.ux")
        cfg["ux"] = ux

    chat = cfg.get("chat_with_data", {})
    cfg["chat_with_data"] = render_section(chat, "config_studio.chat_with_data")


def _render_save_step():
    """Show the final config and enable apply/download only after all review steps."""
    cfg = st.session_state.get("config_studio_draft_config")
    if cfg is None:
        st.info("Generate an AI draft first.")
        return

    st.write("### Save & Export")
    st.caption("This is the final reviewed config assembled from preprocessing, AI draft, and manual adjustments.")
    safe_cfg = serialize_exprs(deepcopy(cfg))
    toml_text = tomlkit.dumps(safe_cfg)
    st.session_state["config_studio_generated_toml"] = toml_text

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Metrics", len(safe_cfg.get("metrics", {})))
    summary_col2.metric("Reports", len(safe_cfg.get("reports", {})))
    summary_col3.metric("Chat Settings", len(safe_cfg.get("chat_with_data", {})))

    st.code(toml_text, language="toml", height=520)
    action_col1, action_col2, others = st.columns([1, 2, 5])
    action_col2.download_button(
        "Download Draft Config",
        data=toml_text,
        file_name="config_studio.toml",
        mime="text/plain",
        type="secondary",
        help="Download the fully reviewed TOML config.",
    )
    if action_col1.button("Apply Draft In App", type="primary", key="config_studio_activate_config"):
        _, saved_toml = save_generated_config(safe_cfg)
        st.session_state["config_studio_generated_toml"] = saved_toml
        st.success("Reviewed draft config activated for the current app session.")


def main():
    template_config = _load_template_config()
    _render_intro()

    with st.sidebar:
        st.write("### Studio Controls")
        st.caption("AI is optional until the final draft step.")
        sample_size = st.number_input(
            "Max Sample Rows",
            min_value=1_000,
            max_value=1_000_000,
            value=100_000,
            step=1_000,
            help="If the uploaded IH sample is larger than this limit, AI Configuration Studio will use a random sample of this many rows.",
        )
        llm = render_litellm_sidebar(
            key_prefix="config_studio",
            default_model="gpt-5.4",
            reasoning_effort="low",
            verbosity="medium",
            missing_key_message="Please configure LLM API key.",
            require_api_key=False,
        )
    if 'file_expanded' not in st.session_state:
        st.session_state['file_expanded'] = True
    filecontainer = st.expander("Sample upload", expanded=st.session_state['file_expanded'])
    with filecontainer:
        uploaded_file = st.file_uploader(
            "Choose Interaction History sample",
            type=["zip", "parquet", "json", "gzip"],
            accept_multiple_files=False,
            help="Start with one representative IH sample. The studio will use it to build a preprocessing plan and AI field catalog.",
        )
        if not uploaded_file:
            st.info("Upload an IH sample to start the studio.")
            st.session_state['file_expanded'] = True
            return
        elif st.session_state['file_expanded']:
            st.session_state['file_expanded'] = False
            st.rerun()

        st.session_state['file_expanded'] = False
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        sample_df = load_ih_sample(file_name, file_bytes, sample_size=sample_size)
        _initialize_editor_state(template_config, f"{file_name}:{len(file_bytes)}", sample_df.columns, file_name)

    filter_expression = (
        compile_filter_rules(st.session_state["config_studio_filter_rows"])
        if st.session_state["config_studio_filter_mode"] == "Rules"
        else st.session_state["config_studio_raw_filter"]
    )
    default_values = _build_default_values_map(st.session_state["config_studio_defaults_rows"])
    effective_calculated_rows = _build_effective_calculated_rows(
        sample_df.columns,
        default_values,
        st.session_state["config_studio_calculated_rows"],
    )
    calculated_fields_text = build_calculated_fields_config_text(effective_calculated_rows)
    preprocessing_error = None
    try:
        working_df, working_columns, _ = apply_ih_preprocessing(
            file_name=file_name,
            file_bytes=file_bytes,
            sample_size=sample_size,
            outcome_time_col=st.session_state["config_studio_outcome_time"],
            decision_time_col=st.session_state["config_studio_decision_time"],
            default_values=default_values,
            filter_expression=filter_expression,
            calculated_field_rows=effective_calculated_rows,
        )
    except Exception as exc:
        preprocessing_error = str(exc)
        working_df = sample_df
        working_columns = sample_df.columns
        calculated_fields_text = "[]"

    empty_after_preprocessing = not preprocessing_error and working_df.is_empty()
    if empty_after_preprocessing:
        st.warning(
            "IH preprocessing returned zero rows. Adjust the runtime settings, default values, filter, or "
            "calculated fields in steps `1. Sample`, `3. Defaults`, `4. Filters`, or `5. Calculated Fields`, then try again."
        )
        working_df = sample_df
        working_columns = sample_df.columns
        calculated_fields_text = "[]"

    approved_fields = st.session_state.get("config_studio_selected_fields") or list(working_columns)
    approved_fields = [field for field in approved_fields if field in working_columns]
    if not approved_fields:
        approved_fields = list(working_columns)
    schema_preview = build_schema_preview(working_df, approved_fields)
    ai_schema_preview = _mask_schema_preview_for_ai(
        schema_preview,
        _sync_ai_example_field_selection(approved_fields),
    )
    _render_metrics_bar(sample_df, working_df, approved_fields)

    if "config_studio_step_selector" not in st.session_state:
        st.session_state["config_studio_step_selector"] = st.session_state.get("config_studio_step", STEP_OPTIONS[0])

    pending_step = st.session_state.pop("config_studio_pending_step", None)
    desired_step = pending_step or st.session_state.get("config_studio_step_selector", STEP_OPTIONS[0])
    if empty_after_preprocessing and desired_step in set(STEP_OPTIONS[5:]):
        st.info(
            "Field approval, AI draft, and downstream review steps are disabled until preprocessing returns at "
            "least one row."
        )
        desired_step = STEP_OPTIONS[3]
    draft_config = st.session_state.get("config_studio_draft_config")
    if draft_config is None and desired_step in set(STEP_OPTIONS[7:]):
        st.info("Generate an AI draft before reviewing metrics, refreshing reports, chat settings, and final export.")
        desired_step = STEP_OPTIONS[6]
    st.session_state["config_studio_step_selector"] = desired_step
    st.session_state["config_studio_step"] = desired_step

    selected_step = st.segmented_control(
        "Studio Steps",
        options=STEP_OPTIONS,
        selection_mode="single",
        key="config_studio_step_selector",
    )
    st.session_state["config_studio_step"] = selected_step

    if selected_step == STEP_OPTIONS[0]:
        _render_sample_step(sample_df, file_name)
    elif selected_step == STEP_OPTIONS[1]:
        _render_required_fields_step(sample_df)
        st.write("### Derived Field Preview")
        st.caption("These fields will appear in the working schema before field approval.")
        preview_fields = [field for field in ["Day", "Month", "Year", "Quarter", "ResponseTime"] if
                          field in working_df.columns]
        st.dataframe(
            working_df.select([st.session_state["config_studio_outcome_time"],
                               st.session_state["config_studio_decision_time"]] + preview_fields)
            .head(25),
            width="stretch",
            hide_index=True,
            height=320,
            key="config_studio_time_preview",
        )
    elif selected_step == STEP_OPTIONS[2]:
        default_frame = _editor_frame(
            st.session_state["config_studio_defaults_rows"],
            ["Field", "Default Value", "Enabled"],
            _blank_default_row,
        )
        _render_preprocess_settings_step(default_frame, list(sample_df.columns))
    elif selected_step == STEP_OPTIONS[3]:
        filter_field_options = sorted(set(sample_df.columns).union(default_values.keys()), key=str.casefold)
        filter_rows_frame = _editor_frame(
            st.session_state["config_studio_filter_rows"],
            ["Field", "Operator", "Value", "Enabled"],
            _blank_filter_row,
        )
        _render_filter_step(filter_field_options, filter_rows_frame)
    elif selected_step == STEP_OPTIONS[4]:
        calculated_frame = _editor_frame(
            st.session_state["config_studio_calculated_rows"],
            ["Name", "Expression", "Enabled"],
            _blank_calculated_row,
        )
        _render_calculated_fields_step(calculated_frame)
    elif selected_step == STEP_OPTIONS[5]:
        _render_field_step(working_df)
    elif selected_step == STEP_OPTIONS[6]:
        _render_ai_step(
            template_config=template_config,
            file_name=file_name,
            working_df=working_df,
            schema_preview=schema_preview,
            default_values=default_values,
            filter_expression=filter_expression,
            calculated_fields_text=calculated_fields_text,
            llm=llm,
            preprocessing_error=preprocessing_error,
        )
    elif selected_step == STEP_OPTIONS[7]:
        _render_metrics_step()
    elif selected_step == STEP_OPTIONS[8]:
        _render_ai_reports_step(
            file_name=file_name,
            working_df=working_df,
            schema_preview=ai_schema_preview,
            llm=llm,
            template_config=template_config,
        )
    elif selected_step == STEP_OPTIONS[9]:
        _render_reports_step()
    elif selected_step == STEP_OPTIONS[10]:
        _render_chat_step()
    else:
        _render_save_step()


main()
