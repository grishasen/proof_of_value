from collections.abc import Iterable

import streamlit as st

from value_dashboard.config_generator.validation import ValidationIssue

BADGE_SETTINGS = {
    "Ready": ("green", ":material/check_circle:"),
    "Needs Review": ("orange", ":material/rate_review:"),
    "Blocked": ("red", ":material/error:"),
    "Pending": ("gray", ":material/hourglass_empty:"),
}


def validation_status_for_issues(issues: list[ValidationIssue]) -> str:
    """Return the overall validation status for a list of issues."""
    if any(issue.severity == "error" for issue in issues):
        return "Blocked"
    if any(issue.severity in {"warning", "info"} for issue in issues):
        return "Needs Review"
    return "Ready"


def validation_issue_note(issues: list[ValidationIssue]) -> str:
    """Render a short note for a validation issue."""
    if not issues:
        return "No validation issues."
    counts = count_validation_issues(issues)
    parts = []
    if counts["error"]:
        parts.append(f"{counts['error']} error{'s' if counts['error'] != 1 else ''}")
    if counts["warning"]:
        parts.append(f"{counts['warning']} warning{'s' if counts['warning'] != 1 else ''}")
    if counts["info"]:
        parts.append(f"{counts['info']} info")
    return ", ".join(parts)


def render_review_progress_badges(
        items: list[dict],
        *,
        title: str = "Review Progress",
        caption: str = "",
) -> None:
    """Render review progress badges."""
    with st.container(border=True):
        st.write(f"### {title}")
        if caption:
            st.caption(caption)
        if not items:
            st.info("No review steps are available yet.")
            return

        cols = st.columns(min(4, len(items)))
        for idx, item in enumerate(items):
            with cols[idx % len(cols)]:
                status = item.get("status", "Pending")
                color, icon = BADGE_SETTINGS.get(status, BADGE_SETTINGS["Pending"])
                st.badge(status, color=color, icon=icon, help=item.get("help"))
                st.write(f"**{item.get('label', 'Review')}**")
                if item.get("note"):
                    st.caption(item["note"])


def _status_for_issues(issues: list[ValidationIssue]) -> str:
    """Return the strongest status represented by a set of issues."""
    if any(issue.severity == "error" for issue in issues):
        return "Error"
    if any(issue.severity == "warning" for issue in issues):
        return "Warning"
    if any(issue.severity == "info" for issue in issues):
        return "Info"
    return "Ready"


def _issue_summary(report_name: str, issues: list[ValidationIssue]) -> str:
    """Return a display summary for validation issue counts."""
    if not issues:
        return "Ready"
    messages = []
    prefix = f"Report '{report_name}': "
    for issue in issues[:2]:
        message = issue.message
        if message.startswith(prefix):
            message = message[len(prefix):]
        messages.append(message)
    if len(issues) > 2:
        messages.append(f"{len(issues) - 2} more")
    return " | ".join(messages)


def _match_report_name(issue_path: str, report_names: list[str]) -> str | None:
    """Return whether an issue path belongs to a report."""
    for report_name in sorted(report_names, key=len, reverse=True):
        report_path = f"reports.{report_name}"
        if issue_path == report_path or issue_path.startswith(f"{report_path}."):
            return report_name
    return None


def _build_report_issue_map(
        reports: dict,
        issues: list[ValidationIssue],
) -> tuple[dict[str, list[ValidationIssue]], list[ValidationIssue]]:
    """Group validation issues by report name."""
    issue_map = {report_name: [] for report_name in reports}
    section_issues = []
    report_names = list(reports.keys())
    for issue in issues:
        if issue.section != "reports":
            continue
        report_name = _match_report_name(issue.path, report_names)
        if report_name is None:
            section_issues.append(issue)
        else:
            issue_map.setdefault(report_name, []).append(issue)
    return issue_map, section_issues


def count_validation_issues(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    """Count validation issues."""
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
    """Render validation details."""
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
    """Render the configuration health summary panel."""
    with st.container(border=True):
        st.write(f"### {title}")
        if caption:
            st.caption(caption)
        if issues is None:
            st.info(pending_message or "Validation will run after a draft config is available.")
            return
        render_validation_details(issues, expanded=False, label="Health Details")


def render_report_validation_summary(
        cfg: dict,
        issues: list[ValidationIssue],
        *,
        title: str = "Report Validation Summary",
        caption: str = "",
) -> None:
    """Render report validation summary."""
    reports = cfg.get("reports", {})
    if not isinstance(reports, dict):
        reports = {}
    issue_map, section_issues = _build_report_issue_map(reports, issues)

    rows = []
    status_counts = {"Ready": 0, "Info": 0, "Warning": 0, "Error": 0}
    for report_name, report in sorted(reports.items(), key=lambda item: str(item[0]).casefold()):
        report_config = report if isinstance(report, dict) else {}
        report_issues = issue_map.get(report_name, [])
        status = _status_for_issues(report_issues)
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.append({
            "Status": status,
            "Report": report_name,
            "Metric": report_config.get("metric", ""),
            "Type": report_config.get("type", ""),
            "Issue Summary": _issue_summary(report_name, report_issues),
        })

    with st.container(border=True):
        st.write(f"### {title}")
        if caption:
            st.caption(caption)

        section_error_count = sum(1 for issue in section_issues if issue.severity == "error")
        section_warning_count = sum(1 for issue in section_issues if issue.severity == "warning")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Reports", len(rows))
        summary_cols[1].metric("Ready", status_counts["Ready"])
        summary_cols[2].metric("Warnings", status_counts["Warning"] + section_warning_count)
        summary_cols[3].metric("Errors", status_counts["Error"] + section_error_count)

        if section_issues:
            for issue in section_issues:
                if issue.severity == "error":
                    st.error(issue.message)
                elif issue.severity == "warning":
                    st.warning(issue.message)
                else:
                    st.info(issue.message)

        if not rows:
            st.info("No reports are defined yet.")
            return

        st.dataframe(rows, hide_index=True, width="stretch")

        report_issues = [issue for issue in issues if issue.section == "reports"]
        if report_issues:
            with st.expander("Report Issue Details", expanded=bool(status_counts["Error"])):
                for issue in report_issues:
                    message = f"**{issue.severity.upper()}** `{issue.path}`: {issue.message}"
                    if issue.severity == "error":
                        st.error(message)
                    elif issue.severity == "warning":
                        st.warning(message)
                    else:
                        st.info(message)
