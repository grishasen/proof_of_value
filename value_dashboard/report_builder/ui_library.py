import copy
import uuid

import streamlit as st
import tomlkit

from value_dashboard.report_builder.field_catalog import get_metric_options
from value_dashboard.report_builder.recipes import get_report_type_display
from value_dashboard.report_builder.service import NEW_REPORT_KEY, build_blank_report, build_new_report_name, \
    build_report_summaries, get_default_metric


def _reset_editor_token():
    st.session_state.rb_editor_token = uuid.uuid4().hex[:8]


def _set_new_draft(cfg: dict, report: dict, report_name: str):
    st.session_state.rb_selected_report = NEW_REPORT_KEY
    st.session_state.rb_draft_report = {
        "__name__": report_name,
        "__report__": copy.deepcopy(report),
    }
    _reset_editor_token()


def _format_report_label(summary: dict) -> str:
    mode = "visual" if summary["mode"] == "visual" else "raw"
    return f"{summary['name']} · {summary['metric']} / {summary['type_display']} · {mode}"


@st.dialog("Report Parameters", width="large")
def _show_report_parameters_dialog(report_name: str, report: dict):
    st.write(f"### {report_name}")
    st.code(
        tomlkit.dumps({"reports": {report_name: report}}),
        language="toml",
    )


def render_report_inventory(cfg: dict):
    reports = cfg.get("reports", {})
    summaries = build_report_summaries(cfg)
    for summary in summaries:
        report = reports[summary["name"]]
        summary["type_display"] = get_report_type_display(summary["metric"], report)

    st.write("### Available Reports")
    if not summaries:
        st.info("No reports defined yet.")
        return

    widths = [2.2, 1.2, 1.1, 0.9, 2.5, 2.0, 0.6]
    headers = ["Name", "Metric", "Type", "Mode", "Description", "Group By", ""]
    header_cols = st.columns(widths, vertical_alignment="center")
    for idx, header in enumerate(headers):
        if header:
            header_cols[idx].markdown(f"**{header}**")

    for summary in summaries:
        report_name = summary["name"]
        report = reports[report_name]
        group_by = report.get("group_by", [])
        group_by_text = ", ".join(group_by) if isinstance(group_by, list) else str(group_by)

        with st.container(border=True):
            row_cols = st.columns(widths, vertical_alignment="center")
            row_cols[0].write(report_name)
            row_cols[1].write(summary["metric"])
            row_cols[2].write(summary["type_display"])
            row_cols[3].write(summary["mode"])
            row_cols[4].write(summary["description"])
            row_cols[5].write(group_by_text)
            if row_cols[6].button(
                    "",
                    icon=":material/tune:",
                    key=f"rb_inventory_params_{report_name}",
                    help="Show report parameters",
            ):
                _show_report_parameters_dialog(report_name, report)


def render_report_library(cfg: dict) -> str:
    reports = cfg.setdefault("reports", {})
    summaries = build_report_summaries(cfg)
    for summary in summaries:
        report = reports[summary["name"]]
        summary["type_display"] = get_report_type_display(summary["metric"], report, include_symbol=False)

    if "rb_selected_report" not in st.session_state:
        st.session_state.rb_selected_report = summaries[0]["name"] if summaries else NEW_REPORT_KEY
    if "rb_draft_report" not in st.session_state:
        st.session_state.rb_draft_report = None
    if "rb_editor_token" not in st.session_state:
        _reset_editor_token()

    st.write("### Report Library")
    st.text_input("Search", key="rb_search")
    metric_options = ["All"] + sorted(get_metric_options(cfg))
    st.selectbox("Metric Filter", metric_options, key="rb_metric_filter")
    type_options = ["All"] + sorted(list({summary["type"] for summary in summaries if summary["type"]}))
    st.selectbox("Type Filter", type_options, key="rb_type_filter")

    filtered = summaries
    search_value = st.session_state.get("rb_search", "").strip().lower()
    metric_filter = st.session_state.get("rb_metric_filter", "All")
    type_filter = st.session_state.get("rb_type_filter", "All")

    if search_value:
        filtered = [
            summary for summary in filtered
            if search_value in summary["name"].lower() or search_value in summary["description"].lower()
        ]
    if metric_filter != "All":
        filtered = [summary for summary in filtered if summary["metric"] == metric_filter]
    if type_filter != "All":
        filtered = [summary for summary in filtered if summary["type"] == type_filter]

    filtered_names = [summary["name"] for summary in filtered]
    label_map = {summary["name"]: _format_report_label(summary) for summary in filtered}

    current_selection = st.session_state.rb_selected_report
    if current_selection in filtered_names:
        selected_index = filtered_names.index(current_selection)
    else:
        selected_index = 0 if filtered_names else None

    if filtered_names:
        selected_report = st.selectbox(
            "Select Report",
            filtered_names,
            index=selected_index,
            format_func=lambda value: label_map[value],
            key="rb_selected_report_picker",
        )
        draft_active = (
                current_selection == NEW_REPORT_KEY and
                st.session_state.get("rb_draft_report") is not None
        )
        if not draft_active and selected_report != current_selection:
            st.session_state.rb_selected_report = selected_report
            st.session_state.rb_draft_report = None
            _reset_editor_token()
        elif draft_active:
            st.caption("Draft is active. Save it or open an existing report explicitly.")
            if st.button("Open Selected Report", key="rb_open_selected_report"):
                st.session_state.rb_selected_report = selected_report
                st.session_state.rb_draft_report = None
                _reset_editor_token()
    else:
        selected_report = current_selection
        st.caption("No reports match the current filters.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("New", key="rb_new_report"):
            metric_name = get_default_metric(cfg)
            report_name = build_new_report_name(reports)
            _set_new_draft(cfg, build_blank_report(metric_name), report_name)
    with col2:
        if st.button("Duplicate", key="rb_duplicate_report", disabled=current_selection not in reports):
            source_report = copy.deepcopy(reports[current_selection])
            draft_name = build_new_report_name(reports, current_selection)
            _set_new_draft(cfg, source_report, draft_name)
    with col3:
        if st.button("Delete", key="rb_delete_report", disabled=current_selection not in reports):
            del reports[current_selection]
            cfg["reports"] = reports
            st.session_state.rb_draft_report = None
            if reports:
                st.session_state.rb_selected_report = list(reports.keys())[0]
            else:
                st.session_state.rb_selected_report = NEW_REPORT_KEY
            _reset_editor_token()

    if st.session_state.rb_selected_report == NEW_REPORT_KEY and st.session_state.rb_draft_report:
        st.info(f"Editing draft report: {st.session_state.rb_draft_report['__name__']}")

    return st.session_state.rb_selected_report
