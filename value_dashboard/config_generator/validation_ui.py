from collections.abc import Iterable

import streamlit as st

from value_dashboard.config_generator.validation import ValidationIssue


def count_validation_issues(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    return counts


def render_validation_details(
        issues: list[ValidationIssue],
        *,
        expanded: bool | None = None,
        label: str = "Validation Details",
) -> None:
    if not issues:
        st.success("No validation issues found.")
        return

    issue_counts = count_validation_issues(issues)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Errors", issue_counts["error"])
    summary_cols[1].metric("Warnings", issue_counts["warning"])
    summary_cols[2].metric("Info", issue_counts["info"])

    expand_details = bool(issue_counts["error"]) if expanded is None else expanded
    with st.expander(label, expanded=expand_details):
        for issue in issues:
            prefix = issue.severity.upper()
            step_hint = f" ({issue.step_hint})" if issue.step_hint else ""
            message = f"**{prefix}** `{issue.path}`{step_hint}: {issue.message}"
            if issue.severity == "error":
                st.error(message)
            elif issue.severity == "warning":
                st.warning(message)
            else:
                st.info(message)


def render_config_health_panel(
        issues: list[ValidationIssue] | None,
        *,
        title: str = "Config Health",
        caption: str = "",
        pending_message: str = "",
) -> None:
    with st.container(border=True):
        st.write(f"### {title}")
        if caption:
            st.caption(caption)
        if issues is None:
            st.info(pending_message or "Validation will run after a draft config is available.")
            return
        render_validation_details(issues, expanded=False, label="Health Details")

