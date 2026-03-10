import os
import re
import tomllib
from copy import deepcopy

import pandas as pd
import streamlit as st
import tomlkit

from value_dashboard.config_generator.ai import build_ai_config_prompt, build_final_config, generate_ai_sections, \
    save_generated_config
from value_dashboard.config_generator.preprocess import apply_ih_preprocessing, build_ih_config, build_schema_preview, \
    compile_filter_rules, detect_ih_file_settings, load_ih_sample
from value_dashboard.metrics.constants import DECISION_TIME, OUTCOME_TIME, REQ_IH_COLUMNS
from value_dashboard.report_builder import render_report_builder
from value_dashboard.utils.config_builder import render_section, render_value, serialize_exprs
from value_dashboard.utils.llm_utils import render_litellm_sidebar

st.set_page_config(page_title="Config Studio", layout="wide")

FILTER_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "contains", "starts with", "in", "not in", "is null",
                    "is not null"]
STEP_OPTIONS = [
    "1. Sample",
    "2. Time Fields",
    "3. Preprocess",
    "4. Approve Fields",
    "5. AI Draft",
    "6. Metrics",
    "7. Reports",
    "8. Chat with Data",
    "9. Save & Export",
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


def _normalize_rows(frame: pd.DataFrame) -> list[dict]:
    rows = frame.to_dict("records")
    cleaned = []
    for row in rows:
        cleaned.append({
            key: ("" if pd.isna(value) else value)
            for key, value in row.items()
        })
    return cleaned


def _editor_frame(rows: list[dict], columns: list[str], blank_row_factory) -> pd.DataFrame:
    """Keep editor tables usable even if the user deletes every row."""
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame([blank_row_factory()], columns=columns)


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


def _reset_report_builder_state():
    """Clear shared report-builder widget state when the studio draft changes."""
    for key in ("rb_selected_report", "rb_draft_report", "rb_editor_token", "rb_search", "rb_metric_filter",
                "rb_type_filter", "rb_selected_report_picker"):
        if key in st.session_state:
            del st.session_state[key]


def _set_step(step_label: str):
    """Queue a programmatic step change for the next rerun."""
    st.session_state["config_studio_step"] = step_label
    st.session_state["config_studio_pending_step"] = step_label


def _set_draft_config(config: dict):
    """Store the mutable generated config that will be reviewed in later steps."""
    st.session_state["config_studio_draft_config"] = deepcopy(config)
    st.session_state["config_studio_generated_toml"] = tomlkit.dumps(serialize_exprs(deepcopy(config)))


def _initialize_editor_state(template_config: dict, file_signature: str, sample_columns: list[str]):
    if st.session_state.get("config_studio_file_signature") == file_signature:
        return
    st.session_state["config_studio_file_signature"] = file_signature
    st.session_state["config_studio_defaults_rows"] = _default_defaults_rows(template_config)
    st.session_state["config_studio_filter_rows"] = _default_filter_rows()
    st.session_state["config_studio_calculated_rows"] = _default_calculated_rows()
    st.session_state["config_studio_filter_mode"] = "Rules"
    st.session_state["config_studio_raw_filter"] = ""
    st.session_state["config_studio_outcome_time"] = _default_time_column(sample_columns, OUTCOME_TIME)
    st.session_state["config_studio_decision_time"] = _default_time_column(sample_columns, DECISION_TIME)
    st.session_state["config_studio_step"] = STEP_OPTIONS[0]
    st.session_state["config_studio_step_selector"] = STEP_OPTIONS[0]
    st.session_state["config_studio_pending_step"] = None
    st.session_state["config_studio_ai_sections"] = None
    st.session_state["config_studio_generated_toml"] = None
    st.session_state["config_studio_draft_config"] = None
    st.session_state["config_studio_selected_fields"] = []
    _reset_report_builder_state()


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


def _extract_filter_fields(filter_expression: str) -> list[str]:
    """Extract field names referenced via pl.col("...") in a Polars filter expression."""
    if not filter_expression:
        return []
    pattern = r"""pl\.col\(\s*["']([^"']+)["']\s*\)"""
    return re.findall(pattern, filter_expression)


def _render_intro():
    st.header("✨ Config Studio", divider='red')
    st.info(
        "Build preprocessing first, approve the working field catalog, then let AI draft metrics and reports from the cleaned schema. Derived time fields are surfaced early so the field-approval step reflects the real reporting surface, not just the raw upload.")


def _render_metrics_bar(sample_df, working_df, approved_fields: list[str]):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Columns", len(sample_df.columns))
    col2.metric("Working Columns", len(working_df.columns))
    col3.metric("Approved Fields", len(approved_fields))
    col4.metric("Rows In Sample", f"{working_df.height:,}")


def _render_sample_step(sample_df, file_name: str):
    file_type, file_pattern = detect_ih_file_settings(file_name)
    meta_col1, meta_col2 = st.columns([1.2, 1.8], gap="large")
    with meta_col1:
        with st.container(border=True):
            st.write("### Sample Source")
            st.caption("The new studio keeps the upload local to the app session and profiles it before any AI call.")
            st.markdown(f"<span class='config-studio-chip'>File type: {file_type}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='config-studio-chip'>Pattern: {file_pattern}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='config-studio-chip'>Columns: {len(sample_df.columns)}</span>",
                        unsafe_allow_html=True)
    with meta_col2:
        with st.container(border=True):
            st.write("### Raw Schema")
            raw_schema = build_schema_preview(sample_df)
            st.dataframe(raw_schema.to_pandas(), width="stretch", hide_index=True, height=360)
    with st.expander("Peek at raw sample rows", expanded=False, icon=":material/table_view:"):
        st.dataframe(sample_df.head(100).to_pandas(), width="stretch", height=320)


def _render_time_step(sample_df):
    st.write("### Time Mapping")
    st.caption(
        "These canonical time fields are used to derive Day, Month, Year, Quarter, and ResponseTime before field approval.")
    columns = sample_df.columns
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["config_studio_outcome_time"] = st.selectbox(
            "Outcome Time Column",
            options=columns,
            index=columns.index(st.session_state["config_studio_outcome_time"])
            if st.session_state["config_studio_outcome_time"] in columns else 0,
            help="Choose the source timestamp used to derive Day, Month, Year, Quarter, and ResponseTime.",
        )
    with col2:
        st.session_state["config_studio_decision_time"] = st.selectbox(
            "Decision Time Column",
            options=columns,
            index=columns.index(st.session_state["config_studio_decision_time"])
            if st.session_state["config_studio_decision_time"] in columns else 0,
            help="Choose the source timestamp used as the baseline for ResponseTime.",
        )
    st.info("These two source columns will be aliased to OutcomeTime and DecisionTime inside the generated IH config.")


def _render_preprocess_step(sample_df, filter_field_options: list[str]):
    editor_col1, editor_col2 = st.columns([1.05, 0.95], gap="large")
    with editor_col1:
        with st.container(border=True):
            title_col, action_col = st.columns([0.78, 0.22], vertical_alignment="center")
            with title_col:
                st.write("### Default Values")
            with action_col:
                if st.button("Add Row", key="config_studio_add_default_row", use_container_width=True):
                    _append_editor_row("config_studio_defaults_rows", _blank_default_row)
                    st.rerun()
            st.caption("Defaults are applied before filters. They may fill nulls or create missing columns.")
            default_frame = _editor_frame(
                st.session_state["config_studio_defaults_rows"],
                ["Field", "Default Value", "Enabled"],
                _blank_default_row,
            )
            edited_defaults = st.data_editor(
                default_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="config_studio_defaults_editor",
                column_config={
                    "Field": st.column_config.TextColumn("Field", help="Column to fill or create."),
                    "Default Value": st.column_config.TextColumn(
                        "Default Value", help="Literal value. Examples: N/A, 0.0, true, 1e-10."
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled"),
                },
            )
            st.session_state["config_studio_defaults_rows"] = _normalize_rows(edited_defaults)

        with st.container(border=True):
            title_col, example_col = st.columns([0.8, 0.2], vertical_alignment="center")
            with title_col:
                heading_col, action_col = st.columns([0.68, 0.32], vertical_alignment="center")
                with heading_col:
                    st.write("### Calculated Fields")
                with action_col:
                    if st.button("Add Row", key="config_studio_add_calculated_row", use_container_width=True):
                        _append_editor_row("config_studio_calculated_rows", _blank_calculated_row)
                        st.rerun()
            with example_col:
                with st.popover("Examples", icon=":material/flare:"):
                    st.code(
                        """pl.col("Treatment")
pl.when(pl.col("PlacementType") != "").then(pl.col("PlacementType")).otherwise(pl.lit("Hero"))""",
                        language="python",
                    )
                    st.caption("Enter the expression body. The studio will alias it to the field name automatically.")
            calculated_frame = _editor_frame(
                st.session_state["config_studio_calculated_rows"],
                ["Name", "Expression", "Enabled"],
                _blank_calculated_row,
            )
            edited_calculated = st.data_editor(
                calculated_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="config_studio_calculated_editor",
                column_config={
                    "Name": st.column_config.TextColumn("Name", help="Name of the generated field."),
                    "Expression": st.column_config.TextColumn(
                        "Expression",
                        help="Polars expression body using pl and np. Do not add .alias(...) unless you want to fully control the expression.",
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled"),
                },
            )
            st.session_state["config_studio_calculated_rows"] = _normalize_rows(edited_calculated)

    with editor_col2:
        with st.container(border=True):
            title_col, action_col = st.columns([0.72, 0.28], vertical_alignment="center")
            with title_col:
                st.write("### IH Filter")
            with action_col:
                if st.button("Add Rule", key="config_studio_add_filter_row", use_container_width=True):
                    _append_editor_row("config_studio_filter_rows", _blank_filter_row)
                    st.rerun()
            st.session_state["config_studio_filter_mode"] = st.segmented_control(
                "Filter Mode",
                options=["Rules", "Raw Polars"],
                default=st.session_state["config_studio_filter_mode"],
                selection_mode="single",
                help="Rules are easier to author; raw mode gives full control over the final ih.extensions.filter expression.",
            )
            if st.session_state["config_studio_filter_mode"] == "Rules":
                filter_rows_frame = _editor_frame(
                    st.session_state["config_studio_filter_rows"],
                    ["Field", "Operator", "Value", "Enabled"],
                    _blank_filter_row,
                )
                edited_filters = st.data_editor(
                    filter_rows_frame,
                    num_rows="dynamic",
                    width="stretch",
                    hide_index=True,
                    key="config_studio_filter_editor",
                    column_config={
                        "Field": st.column_config.SelectboxColumn(
                            "Field", options=filter_field_options, required=False
                        ),
                        "Operator": st.column_config.SelectboxColumn(
                            "Operator", options=FILTER_OPERATORS, required=False
                        ),
                        "Value": st.column_config.TextColumn(
                            "Value", help="Comma-separated values for in/not in."
                        ),
                        "Enabled": st.column_config.CheckboxColumn("Enabled"),
                    },
                )
                st.session_state["config_studio_filter_rows"] = _normalize_rows(edited_filters)
                compiled_filter = compile_filter_rules(st.session_state["config_studio_filter_rows"])
                st.caption("Compiled filter")
                st.code(compiled_filter or "pl.lit(True)", language="python")
            else:
                st.session_state["config_studio_raw_filter"] = st.text_area(
                    "Raw Polars Filter",
                    value=st.session_state["config_studio_raw_filter"],
                    height=220,
                    help="Full Polars predicate expression. This filter runs before calculated fields, like the runtime IH pipeline.",
                )
                st.caption("Example: pl.col(\"Outcome\").is_in([\"Pending\", \"Impression\", \"Clicked\"])")
        with st.container(border=True):
            st.write("### Runtime Notes")
            st.caption("The generator follows runtime IH semantics as closely as possible.")
            st.markdown("- Defaults run before filters.")
            st.markdown("- The IH filter runs before calculated fields.")
            st.markdown("- Day / Month / Year / Quarter / ResponseTime are derived before field approval.")
            st.markdown("- Approved fields affect AI, not ingestion.")


def _render_field_step(working_df):
    available_fields = sorted(working_df.columns, key=str.casefold)
    current_selection = st.session_state.get("config_studio_selected_fields") or list(available_fields)
    current_selection = sorted(
        [field for field in current_selection if field in available_fields],
        key=str.casefold,
    )
    required_fields = sorted([field for field in REQ_IH_COLUMNS if field in available_fields], key=str.casefold)
    optional_fields = [field for field in available_fields if field not in required_fields]
    current_optional_selection = [field for field in current_selection if field in optional_fields]
    selection_col1, selection_col2 = st.columns([1.15, 0.85], gap="large")
    with selection_col1:
        with st.container(border=True):
            st.write("### Approved Field Catalog")
            st.caption("Only approved fields will be exposed to the AI stage.")
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
                )
            else:
                selected_optional_fields = []
                st.info("Only required IH fields are available in the current working schema.")
            selected_fields = sorted(required_fields + selected_optional_fields, key=str.casefold)
            st.session_state["config_studio_selected_fields"] = selected_fields
    with selection_col2:
        with st.container(border=True):
            st.write("### Working Schema")
            st.caption("This is the post-processed schema. Derived time fields are already visible here.")
            schema_preview = build_schema_preview(working_df, selected_fields)
            st.dataframe(schema_preview.to_pandas(), width="stretch", hide_index=True, height=360)
    with st.expander("Preview working sample after preprocessing", expanded=False, icon=":material/preview:"):
        preview_fields = st.session_state["config_studio_selected_fields"] or available_fields
        st.dataframe(
            working_df.select([field for field in preview_fields if field in working_df.columns]).head(100).to_pandas(),
            width="stretch",
            height=320,
        )


def _render_ai_step(template_config: dict, file_name: str, working_df: pd.DataFrame, schema_preview: pd.DataFrame,
                    default_values: dict, filter_expression: str, calculated_fields_text: str, llm,
                    preprocessing_error: str | None = None):
    st.write("### AI Draft")
    st.caption("The AI sees only the approved working schema and the final IH preprocessing config.")
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

    prep_col1, prep_col2 = st.columns([1, 1], gap="large")
    with prep_col1:
        with st.container(border=True):
            st.write("#### IH Config Preview")
            st.code(tomlkit.dumps({"ih": ih_config}), language="toml", height=480)
    with prep_col2:
        with st.container(border=True):
            st.write("#### Approved Schema Preview")
            st.dataframe(schema_preview.to_pandas(), width="stretch", hide_index=True, height=480)

    if not llm:
        st.info("Configure an API key in the sidebar when you are ready to generate metrics and reports.")
        return

    prompt = build_ai_config_prompt(
        file_name=file_name,
        approved_schema=schema_preview,
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
                _set_step(STEP_OPTIONS[5])
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
            f"Continue with `6. Metrics`, `7. Reports`, and `8. Chat with Data` before exporting."
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
    # if approved_fields:
    #    with st.container(border=True):
    #        st.pills(
    #            "",
    #            options=approved_fields,
    #            default=approved_fields,
    #            selection_mode="multi",
    #            disabled=True,
    #            label_visibility="collapsed",
    #        )

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


def _render_reports_step():
    """Review, select, and edit reports on top of the current draft config."""
    cfg = st.session_state.get("config_studio_draft_config")
    if cfg is None:
        st.info("Generate an AI draft first.")
        return

    st.write("### Reports Review")
    st.caption("Keep the useful reports, delete weak ones, and refine the remaining report definitions.")
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
        llm = render_litellm_sidebar(
            key_prefix="config_studio",
            default_model="gpt-5.4",
            reasoning_effort="low",
            verbosity="low",
            missing_key_message="Please configure LLM API key.",
            require_api_key=False,
        )

    uploaded_file = st.file_uploader(
        "Choose Interaction History sample",
        type=["zip", "parquet", "json", "gzip"],
        accept_multiple_files=False,
        help="Start with one representative IH sample. The studio will use it to build a preprocessing plan and AI field catalog.",
    )
    if not uploaded_file:
        st.info("Upload an IH sample to start the studio.")
        return

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    sample_df = load_ih_sample(file_name, file_bytes)
    _initialize_editor_state(template_config, f"{file_name}:{len(file_bytes)}", sample_df.columns)

    filter_expression = (
        compile_filter_rules(st.session_state["config_studio_filter_rows"])
        if st.session_state["config_studio_filter_mode"] == "Rules"
        else st.session_state["config_studio_raw_filter"]
    )
    default_values = _build_default_values_map(st.session_state["config_studio_defaults_rows"])
    preprocessing_error = None
    try:
        working_df, working_columns, calculated_fields_text = apply_ih_preprocessing(
            file_name=file_name,
            file_bytes=file_bytes,
            outcome_time_col=st.session_state["config_studio_outcome_time"],
            decision_time_col=st.session_state["config_studio_decision_time"],
            default_values=default_values,
            filter_expression=filter_expression,
            calculated_field_rows=st.session_state["config_studio_calculated_rows"],
        )
    except Exception as exc:
        preprocessing_error = str(exc)
        working_df = sample_df
        working_columns = sample_df.columns
        calculated_fields_text = "[]"

    empty_after_preprocessing = not preprocessing_error and working_df.is_empty()
    if empty_after_preprocessing:
        st.warning(
            "IH preprocessing returned zero rows. Adjust the filter, defaults, or calculated fields in "
            "`3. Preprocess`, then try again."
        )
        working_df = sample_df
        working_columns = sample_df.columns
        calculated_fields_text = "[]"

    approved_fields = st.session_state.get("config_studio_selected_fields") or list(working_columns)
    approved_fields = [field for field in approved_fields if field in working_columns]
    if not approved_fields:
        approved_fields = list(working_columns)
    schema_preview = build_schema_preview(working_df, approved_fields)
    _render_metrics_bar(sample_df, working_df, approved_fields)

    if "config_studio_step_selector" not in st.session_state:
        st.session_state["config_studio_step_selector"] = st.session_state.get("config_studio_step", STEP_OPTIONS[0])

    pending_step = st.session_state.pop("config_studio_pending_step", None)
    desired_step = pending_step or st.session_state.get("config_studio_step_selector", STEP_OPTIONS[0])
    if empty_after_preprocessing and desired_step in {STEP_OPTIONS[3], STEP_OPTIONS[4]}:
        st.info("Field approval and AI draft are disabled until preprocessing returns at least one row.")
        desired_step = STEP_OPTIONS[2]
    draft_config = st.session_state.get("config_studio_draft_config")
    if draft_config is None and desired_step in set(STEP_OPTIONS[5:]):
        st.info("Generate an AI draft before reviewing metrics, reports, chat settings, and final export.")
        desired_step = STEP_OPTIONS[4]
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
        _render_time_step(sample_df)
        with st.container(border=True):
            st.write("### Derived Field Preview")
            st.caption("These fields will appear in the working schema before field approval.")
            preview_fields = [field for field in ["Day", "Month", "Year", "Quarter", "ResponseTime"] if
                              field in working_df.columns]
            st.dataframe(
                working_df.select([st.session_state["config_studio_outcome_time"],
                                   st.session_state["config_studio_decision_time"]] + preview_fields)
                .head(25).to_pandas(),
                width="stretch",
                hide_index=True,
                height=320,
            )
    elif selected_step == STEP_OPTIONS[2]:
        filter_field_options = sorted(set(sample_df.columns).union(default_values.keys()))
        _render_preprocess_step(sample_df, filter_field_options)
    elif selected_step == STEP_OPTIONS[3]:
        _render_field_step(working_df)
    elif selected_step == STEP_OPTIONS[4]:
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
    elif selected_step == STEP_OPTIONS[5]:
        _render_metrics_step()
    elif selected_step == STEP_OPTIONS[6]:
        _render_reports_step()
    elif selected_step == STEP_OPTIONS[7]:
        _render_chat_step()
    else:
        _render_save_step()


main()
