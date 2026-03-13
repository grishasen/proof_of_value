import copy
import uuid

import pandas as pd
import streamlit as st

from value_dashboard.report_builder.field_catalog import build_metric_field_catalog, ensure_current_option, \
    ensure_current_options, get_metric_options
from value_dashboard.report_builder.metric_display import get_metric_display_name
from value_dashboard.report_builder.models import ReportBuilderState
from value_dashboard.report_builder.recipes import get_default_recipe, get_recipe, get_recipe_display_name, \
    get_supported_recipes
from value_dashboard.report_builder.serialization import serialize_report_state
from value_dashboard.report_builder.service import NEW_REPORT_KEY, build_state, build_toml_preview
from value_dashboard.report_builder.ui_library import render_report_library
from value_dashboard.report_builder.ui_raw import render_raw_report_editor
from value_dashboard.report_builder.validation import validate_report_state

DIMENSION_HELP = "Choose a field derived from metrics.<metric>.group_by or metrics.global_filters."
MEASURE_HELP = "Choose a score derived from metrics.<metric>.scores."
PROPERTY_HELP = "Choose a descriptive property derived from metrics.descriptive.columns."
GROUP_BY_HELP = "Add extra dimensions that should remain in report.group_by when the TOML is saved."
EXPERIMENT_DIMENSION_HELP = (
    "Choose an experiment dimension derived from metrics.<metric>.group_by, "
    "metrics.global_filters, experiment_name or experiment_group."
)
REPORT_BUILDER_MEMO = (
    "Choose a metric first, then map the chart fields from the dimensions and scores defined in the metrics section. "
    "Visual mode covers the supported report shapes and preserves manual TOML extras on save; switch to raw mode "
    "when a report needs fields the builder does not manage."
)


def _optional_selectbox(label: str, options: list, value, key: str, help_text: str = ""):
    """Render an optional selectbox that keeps legacy values visible for backward-compatible edits."""
    select_options = ["---"] + ensure_current_option(options, value)
    current_value = value if value not in (None, "", []) else "---"
    selected = st.selectbox(
        label,
        select_options,
        index=select_options.index(current_value),
        key=key,
        help=help_text,
    )
    return None if selected == "---" else selected


def _required_selectbox(label: str, options: list, value, key: str, help_text: str = ""):
    """Render a required selectbox while keeping manual values selectable if they already exist."""
    select_options = ensure_current_option(options, value)
    index = select_options.index(value) if value in select_options and value not in (None, "") else 0
    return st.selectbox(label, select_options, index=index, key=key, help=help_text)


def _multiselect(label: str, options: list, values: list, key: str, help_text: str = ""):
    """Render a multiselect that preserves existing manual selections."""
    select_options = ensure_current_options(options, values)
    return st.multiselect(
        label,
        select_options,
        default=[value for value in values if value in select_options],
        key=key,
        help=help_text,
    )


def _render_reference_editor(reference: dict, key_base: str) -> dict:
    """Edit gauge reference values as a simple two-column table."""
    rows = [{"Name": key, "Reference": value} for key, value in reference.items()]
    st.caption("Add one reference target per gauge label. Empty names are ignored when saving.")
    edited_df = st.data_editor(
        pd.DataFrame(rows, columns=["Name", "Reference"]),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"{key_base}_reference",
    )
    result = {}
    for _, row in edited_df.iterrows():
        name = row.get("Name")
        if pd.isna(name) or str(name).strip() == "":
            continue
        result[str(name)] = row.get("Reference")
    return result


def _render_stages_editor(stages: list, key_base: str) -> list:
    """Edit funnel stages as an ordered multiline list."""
    value = st.text_area(
        "Stages",
        value="\n".join(stages),
        key=f"{key_base}_stages",
        help="Enter one funnel stage per line in the order it should appear.",
    )
    return [line.strip() for line in value.splitlines() if line.strip()]


def _render_recipe_fields(state: ReportBuilderState, catalog: dict, key_base: str):
    """Render only the field roles required by the selected recipe."""
    dimensions = catalog["dimensions"]
    measures = catalog["measures"]
    properties = catalog["properties"]

    recipe_key = state.chart_key

    if recipe_key == "line":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("X-Axis", dimensions, state.x, f"{key_base}_x", DIMENSION_HELP)
        with c2:
            state.y = _required_selectbox("Y-Axis", measures, state.y, f"{key_base}_y", MEASURE_HELP)
        with c3:
            state.color = _optional_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        c4, c5 = st.columns(2)
        with c4:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", DIMENSION_HELP)
        with c5:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", DIMENSION_HELP
            )
    elif recipe_key == "gauge":
        state.value = _required_selectbox("Value", measures, state.value, f"{key_base}_value", MEASURE_HELP)
        state.group_by = _multiselect(
            "Group By", dimensions, state.group_by, f"{key_base}_group_by",
            help_text="Gauge reports support one or two dimensions selected from the metric field catalog."
        )
        st.write("Reference Values")
        state.reference = _render_reference_editor(state.reference, key_base)
    elif recipe_key == "treemap":
        state.group_by = _multiselect("Group By", dimensions, state.group_by, f"{key_base}_group_by", GROUP_BY_HELP)
        if not state.metric.startswith("clv"):
            state.color = _required_selectbox("Colour Metric", measures, state.color, f"{key_base}_color",
                                              MEASURE_HELP)
    elif recipe_key == "heatmap":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("X-Axis", dimensions, state.x, f"{key_base}_x", DIMENSION_HELP)
        with c2:
            state.y = _required_selectbox("Y-Axis", dimensions, state.y, f"{key_base}_y", DIMENSION_HELP)
        with c3:
            state.color = _required_selectbox("Colour Metric", measures, state.color, f"{key_base}_color",
                                              MEASURE_HELP)
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "scatter":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("X-Axis", measures, state.x, f"{key_base}_x", MEASURE_HELP)
        with c2:
            state.y = _required_selectbox("Y-Axis", measures, state.y, f"{key_base}_y", MEASURE_HELP)
        with c3:
            state.size = _required_selectbox("Size", measures, state.size, f"{key_base}_size", MEASURE_HELP)
        c4, c5, c6 = st.columns(3)
        with c4:
            state.color = _required_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        with c5:
            state.animation_frame = _required_selectbox(
                "Animation Frame", dimensions, state.animation_frame, f"{key_base}_animation_frame", DIMENSION_HELP
            )
        with c6:
            state.animation_group = _required_selectbox(
                "Animation Group", dimensions, state.animation_group, f"{key_base}_animation_group", DIMENSION_HELP
            )
        c7, c8 = st.columns(2)
        with c7:
            state.log_x = st.checkbox(
                "Log X",
                value=state.log_x,
                key=f"{key_base}_log_x",
                help="Apply logarithmic scaling to the X axis.",
            )
        with c8:
            state.log_y = st.checkbox(
                "Log Y",
                value=state.log_y,
                key=f"{key_base}_log_y",
                help="Apply logarithmic scaling to the Y axis.",
            )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "bar_polar":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.r = _required_selectbox("Radial Measure", measures, state.r, f"{key_base}_r", MEASURE_HELP)
        with c2:
            state.theta = _required_selectbox("Theta", dimensions, state.theta, f"{key_base}_theta", DIMENSION_HELP)
        with c3:
            state.color = _required_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        state.showlegend = st.checkbox(
            "Show Legend",
            value=state.showlegend,
            key=f"{key_base}_showlegend",
            help="Display the color legend on the saved chart.",
        )
    elif recipe_key == "descriptive_line":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("X-Axis", dimensions, state.x, f"{key_base}_x", DIMENSION_HELP)
        with c2:
            state.property = _required_selectbox("Property", properties, state.property, f"{key_base}_property",
                                                 PROPERTY_HELP)
        with c3:
            state.score = _required_selectbox("Score", measures, state.score, f"{key_base}_score", MEASURE_HELP)
        c4, c5, c6 = st.columns(3)
        with c4:
            state.color = _optional_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        with c5:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", DIMENSION_HELP)
        with c6:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", DIMENSION_HELP
            )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "descriptive_boxplot":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("X-Axis", dimensions, state.x, f"{key_base}_x", DIMENSION_HELP)
        with c2:
            state.property = _required_selectbox("Property", properties, state.property, f"{key_base}_property",
                                                 PROPERTY_HELP)
        with c3:
            state.color = _optional_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        c4, c5 = st.columns(2)
        with c4:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", DIMENSION_HELP)
        with c5:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", DIMENSION_HELP
            )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "descriptive_histogram":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.property = _required_selectbox("Property", properties, state.property, f"{key_base}_property",
                                                 PROPERTY_HELP)
        with c2:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", DIMENSION_HELP)
        with c3:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", DIMENSION_HELP
            )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "descriptive_heatmap":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            state.x = _required_selectbox("X-Axis", dimensions, state.x, f"{key_base}_x", DIMENSION_HELP)
        with c2:
            state.y = _required_selectbox("Y-Axis", dimensions, state.y, f"{key_base}_y", DIMENSION_HELP)
        with c3:
            state.property = _required_selectbox("Property", properties, state.property, f"{key_base}_property",
                                                 PROPERTY_HELP)
        with c4:
            state.score = _required_selectbox("Score", measures, state.score, f"{key_base}_score", MEASURE_HELP)
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "descriptive_funnel":
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("X Field", properties, state.x, f"{key_base}_x", PROPERTY_HELP)
        with c2:
            state.color = _required_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        with c3:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", DIMENSION_HELP
            )
        state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row, f"{key_base}_facet_row",
                                              DIMENSION_HELP)
        state.stages = _render_stages_editor(state.stages, key_base)
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "experiment_z_score":
        state.x = "z_score"
        c1, c2, c3 = st.columns(3)
        with c1:
            state.y = _required_selectbox("Y-Axis", dimensions, state.y, f"{key_base}_y", EXPERIMENT_DIMENSION_HELP)
        with c2:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", EXPERIMENT_DIMENSION_HELP)
        with c3:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column",
                EXPERIMENT_DIMENSION_HELP
            )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "experiment_odds_ratio":
        odds_ratio_measures = [
            value for value in measures
            if isinstance(value, str) and (value.startswith("g") or value.startswith("chi2"))
        ]
        c1, c2, c3 = st.columns(3)
        with c1:
            state.x = _required_selectbox("Statistic", odds_ratio_measures, state.x, f"{key_base}_x",
                                          "Choose a supported experiment statistic from metrics.<metric>.scores.")
        with c2:
            state.y = _required_selectbox("Y-Axis", dimensions, state.y, f"{key_base}_y", EXPERIMENT_DIMENSION_HELP)
        with c3:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", EXPERIMENT_DIMENSION_HELP)
        state.facet_column = _optional_selectbox(
            "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", EXPERIMENT_DIMENSION_HELP
        )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "clv_histogram":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            state.x = _required_selectbox("X-Axis", measures, state.x, f"{key_base}_x", MEASURE_HELP)
        with c2:
            state.color = _optional_selectbox("Colour By", dimensions, state.color, f"{key_base}_color",
                                              DIMENSION_HELP)
        with c3:
            state.facet_row = _optional_selectbox("Row Facets", dimensions, state.facet_row,
                                                  f"{key_base}_facet_row", DIMENSION_HELP)
        with c4:
            state.facet_column = _optional_selectbox(
                "Column Facets", dimensions, state.facet_column, f"{key_base}_facet_column", DIMENSION_HELP
            )
        state.group_by = _multiselect("Additional Group By", dimensions, state.group_by, f"{key_base}_group_by",
                                      GROUP_BY_HELP)
    elif recipe_key == "clv_treemap":
        state.group_by = _multiselect("Group By", dimensions, state.group_by, f"{key_base}_group_by", GROUP_BY_HELP)
    elif recipe_key == "clv_corr":
        c1, c2 = st.columns(2)
        with c1:
            state.x = _required_selectbox("X-Axis", measures, state.x, f"{key_base}_x", MEASURE_HELP)
        with c2:
            state.y = _required_selectbox("Y-Axis", measures, state.y, f"{key_base}_y", MEASURE_HELP)
    elif recipe_key == "clv_exposure":
        st.caption("This report type does not require additional configuration.")
    elif recipe_key == "clv_model":
        st.caption("This report type does not require additional configuration.")
    elif recipe_key == "clv_rfm_density":
        st.caption("This report type does not require additional configuration.")

    return state


def _render_visual_editor(cfg: dict, state: ReportBuilderState, original_name: str, key_base: str):
    """Render the guided visual editor for reports supported by recipe metadata."""
    metrics_options = get_metric_options(cfg)
    if not metrics_options:
        st.error("No metrics are defined. Configure the metrics section before creating reports.")
        return

    current_metric = state.metric if state.metric in metrics_options else (
        metrics_options[0] if metrics_options else "")
    state.metric = st.selectbox(
        "Metric",
        metrics_options,
        index=metrics_options.index(current_metric) if current_metric in metrics_options else 0,
        key=f"{key_base}_metric",
        format_func=lambda value: get_metric_display_name(value, include_symbol=False),
        help="The metric controls which dimensions, measures and visual recipes are available."
    )

    supported_recipes = get_supported_recipes(state.metric)
    if not supported_recipes:
        st.error("No visual recipes are available for the selected metric. Use raw mode instead.")
        return

    default_recipe = state.chart_key if state.chart_key in supported_recipes else get_default_recipe(state.metric)
    if default_recipe not in supported_recipes and supported_recipes:
        default_recipe = supported_recipes[0]

    recipe_labels = {recipe_key: get_recipe_display_name(recipe_key, include_symbol=False) for recipe_key in
                     supported_recipes}
    state.chart_key = st.selectbox(
        "Plot Type",
        supported_recipes,
        index=supported_recipes.index(default_recipe) if default_recipe in supported_recipes else 0,
        format_func=lambda value: recipe_labels[value],
        key=f"{key_base}_chart_key",
        help="Pick the chart recipe to author. The corresponding TOML report type is generated automatically.",
    )
    state.type = get_recipe(state.chart_key)["type"]

    state.name = st.text_input(
        "Report Name",
        value=state.name,
        key=f"{key_base}_name",
        help="Unique report key under [reports.<name>] in the TOML file.",
    )
    state.description = state.description if state.description else state.name
    state.description = st.text_area(
        "Description",
        value=state.description,
        key=f"{key_base}_description",
        help="Short business-facing description shown in report listings and dashboards.",
    )

    catalog = build_metric_field_catalog(cfg, state.metric)
    state = _render_recipe_fields(state, catalog, key_base)

    issues = validate_report_state(state, cfg)

    if state.extras:
        st.caption("Manual fields preserved on save: " + ", ".join(sorted(state.extras.keys())))
    if st.button(
            "Save Report",
            key=f"{key_base}_save",
            type="primary",
            help="Validate the current mapping and write the generated TOML back into the config draft.",
    ):
        if state.name != original_name and state.name in cfg.setdefault("reports", {}):
            st.error(f"Report '{state.name}' already exists.")
        elif issues:
            for issue in issues:
                st.error(issue)
        else:
            reports = cfg.setdefault("reports", {})
            serialized_report = serialize_report_state(copy.deepcopy(state))
            if original_name in reports and original_name != state.name:
                del reports[original_name]
            reports[state.name] = serialized_report
            cfg["reports"] = reports
            st.session_state.rb_selected_report = state.name
            st.session_state.rb_draft_report = None
            st.session_state.rb_editor_token = uuid.uuid4().hex[:8]
            st.success(f"Report '{state.name}' saved.")

    st.write("### Validation")
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("Configuration is valid.")

    st.write(
        f"### Generated TOML for {get_metric_display_name(state.metric)} {get_recipe_display_name(state.chart_key)}")
    st.code(build_toml_preview(copy.deepcopy(state)), language="toml")

@st.fragment()
def render_report_builder(cfg: dict):
    """Render the report library and editor while keeping the current TOML structure intact."""
    layout_col1, layout_col2 = st.columns([0.9, 2.1], gap="large")
    with layout_col1:
        selected_report = render_report_library(cfg)
    state = build_state(cfg, selected_report, st.session_state.get("rb_draft_report"))
    key_base = st.session_state.get("rb_editor_token", "rb_report_builder")

    reports = cfg.get("reports", {})
    if selected_report != NEW_REPORT_KEY and selected_report in reports:
        source_report = copy.deepcopy(reports[selected_report])
        original_name = selected_report
    elif st.session_state.get("rb_draft_report") is not None:
        source_report = copy.deepcopy(st.session_state["rb_draft_report"]["__report__"])
        original_name = NEW_REPORT_KEY
    else:
        source_report = serialize_report_state(copy.deepcopy(state)) if state.chart_key else {
            "metric": state.metric,
            "type": state.type,
            "description": state.description,
            "group_by": state.group_by,
        }
        original_name = NEW_REPORT_KEY

    with layout_col2:
        st.write("### Report Editor")
        st.info(REPORT_BUILDER_MEMO)
        if state.chart_key:
            mode_options = ["visual", "raw"]
            default_mode = 0 if state.mode == "visual" else 1
            mode = st.radio(
                "Editing Mode",
                mode_options,
                index=default_mode,
                horizontal=True,
                key=f"{key_base}_mode",
                help="Use visual mode for supported recipes. Use raw mode to edit the underlying TOML fields directly.",
            )
        else:
            mode = "raw"
            st.warning(state.reason)

        if mode == "raw":
            render_raw_report_editor(cfg, state.name, source_report, original_name, key_base)
        else:
            _render_visual_editor(cfg, state, original_name, key_base)
