import copy
import uuid

import pandas as pd
import streamlit as st

from value_dashboard.report_builder.field_catalog import get_metric_options
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
    return f"{summary['name']} · {summary['metric']} / {summary['type']} · {mode}"


def render_report_library(cfg: dict) -> str:
    reports = cfg.setdefault("reports", {})
    summaries = build_report_summaries(cfg)

    if "rb_selected_report" not in st.session_state:
        st.session_state.rb_selected_report = summaries[0]["name"] if summaries else NEW_REPORT_KEY
    if "rb_draft_report" not in st.session_state:
        st.session_state.rb_draft_report = None
    if "rb_editor_token" not in st.session_state:
        _reset_editor_token()

    st.write("### Available Reports")
    if summaries:
        st.dataframe(
            pd.DataFrame(summaries)[["name", "metric", "type", "mode", "description"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No reports defined yet.")

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
