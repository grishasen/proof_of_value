import re
from dataclasses import dataclass
from typing import Iterable, Literal

from value_dashboard.metrics.constants import DECISION_TIME, OUTCOME_TIME, REQ_IH_COLUMNS
from value_dashboard.report_builder.field_catalog import build_metric_field_catalog
from value_dashboard.report_builder.serialization import deserialize_report
from value_dashboard.report_builder.validation import validate_report_state

Severity = Literal["error", "warning", "info"]

METRIC_FIELD_REFERENCE_KEYS = {
    "experiment_name",
    "experiment_group",
    "order_id_col",
    "customer_id_col",
    "monetary_value_col",
    "purchase_date_col",
    "recurring_period",
    "recurring_cost",
}
RUNTIME_REQUIRED_FIELDS = [
    OUTCOME_TIME,
    DECISION_TIME,
    "Name",
    "Issue",
    "Group",
    "InteractionID",
    "Rank",
    "Outcome",
]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    section: str
    path: str
    message: str
    step_hint: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.severity == "error"


def _field_set(fields: Iterable[str] | None) -> set[str] | None:
    if fields is None:
        return None
    return {str(field) for field in fields if field not in (None, "")}


def _clean_list(values) -> list[str]:
    if isinstance(values, list):
        return [str(value) for value in values if value not in (None, "", [])]
    if isinstance(values, str):
        return [part.strip() for part in values.replace("\n", ",").split(",") if part.strip()]
    return []


def _extract_filter_fields(filter_expression: str) -> list[str]:
    if not filter_expression:
        return []
    pattern = r"""pl\.col\(\s*["']([^"']+)["']\s*\)"""
    return re.findall(pattern, str(filter_expression))


def _add_unknown_field_issues(
        issues: list[ValidationIssue],
        *,
        values: Iterable[str],
        available_fields: set[str] | None,
        section: str,
        path: str,
        step_hint: str,
        message_prefix: str,
):
    if available_fields is None:
        return
    unknown_fields = sorted(
        {field for field in values if field not in available_fields},
        key=str.casefold,
    )
    if not unknown_fields:
        return
    issues.append(ValidationIssue(
        severity="error",
        section=section,
        path=path,
        message=message_prefix + ": " + ", ".join(unknown_fields),
        step_hint=step_hint,
    ))


def _validate_runtime_fields(
        cfg: dict,
        runtime_fields: set[str] | None,
) -> list[ValidationIssue]:
    if runtime_fields is None:
        return []

    issues = []
    ih_extensions = cfg.get("ih", {}).get("extensions", {})
    default_values = ih_extensions.get("default_values", {})
    default_fields = set(default_values.keys()) if isinstance(default_values, dict) else set()
    available_runtime_fields = runtime_fields | default_fields

    missing_runtime_fields = [
        field_name
        for field_name in RUNTIME_REQUIRED_FIELDS
        if field_name not in available_runtime_fields
    ]
    if missing_runtime_fields:
        issues.append(ValidationIssue(
            severity="error",
            section="ih",
            path="ih",
            message=(
                    "Required runtime fields are missing before Interaction History preprocessing: "
                    + ", ".join(missing_runtime_fields)
            ),
            step_hint="Required Fields",
        ))
    return issues


def _validate_metric_filters(
        issues: list[ValidationIssue],
        *,
        metric_name: str,
        filter_expression: str,
        available_fields: set[str] | None,
):
    referenced_fields = _extract_filter_fields(filter_expression)
    _add_unknown_field_issues(
        issues,
        values=referenced_fields,
        available_fields=available_fields,
        section="metrics",
        path=f"metrics.{metric_name}.filter",
        step_hint="Metrics",
        message_prefix=f"{metric_name}.filter references fields outside the approved field catalog",
    )


def _validate_metrics(
        cfg: dict,
        approved_fields: set[str] | None,
) -> list[ValidationIssue]:
    issues = []
    metrics = cfg.get("metrics", {})
    if not isinstance(metrics, dict):
        return [
            ValidationIssue(
                severity="error",
                section="metrics",
                path="metrics",
                message="The metrics section must be a table.",
                step_hint="Metrics",
            )
        ]

    global_filters = _clean_list(metrics.get("global_filters", []))
    _add_unknown_field_issues(
        issues,
        values=global_filters,
        available_fields=approved_fields,
        section="metrics",
        path="metrics.global_filters",
        step_hint="Metrics",
        message_prefix="metrics.global_filters references fields outside the approved field catalog",
    )

    for metric_name, metric_config in metrics.items():
        if metric_name == "global_filters":
            continue
        if not isinstance(metric_config, dict):
            issues.append(ValidationIssue(
                severity="warning",
                section="metrics",
                path=f"metrics.{metric_name}",
                message=f"Metric '{metric_name}' is not a table and cannot be validated deeply.",
                step_hint="Metrics",
            ))
            continue

        group_by = _clean_list(metric_config.get("group_by", []))
        if "group_by" in metric_config and not group_by:
            issues.append(ValidationIssue(
                severity="error",
                section="metrics",
                path=f"metrics.{metric_name}.group_by",
                message=f"{metric_name}.group_by must contain at least one field.",
                step_hint="Metrics",
            ))
        elif "group_by" not in metric_config:
            issues.append(ValidationIssue(
                severity="warning",
                section="metrics",
                path=f"metrics.{metric_name}",
                message=f"Metric '{metric_name}' does not define group_by.",
                step_hint="Metrics",
            ))

        _add_unknown_field_issues(
            issues,
            values=group_by,
            available_fields=approved_fields,
            section="metrics",
            path=f"metrics.{metric_name}.group_by",
            step_hint="Metrics",
            message_prefix=f"{metric_name}.group_by references fields outside the approved field catalog",
        )

        if isinstance(metric_config.get("columns"), list):
            _add_unknown_field_issues(
                issues,
                values=_clean_list(metric_config.get("columns", [])),
                available_fields=approved_fields,
                section="metrics",
                path=f"metrics.{metric_name}.columns",
                step_hint="Metrics",
                message_prefix=f"{metric_name}.columns references fields outside the approved field catalog",
            )

        for field_key in METRIC_FIELD_REFERENCE_KEYS:
            field_name = metric_config.get(field_key)
            if isinstance(field_name, str) and field_name:
                _add_unknown_field_issues(
                    issues,
                    values=[field_name],
                    available_fields=approved_fields,
                    section="metrics",
                    path=f"metrics.{metric_name}.{field_key}",
                    step_hint="Metrics",
                    message_prefix=(
                        f"{metric_name}.{field_key} references a field outside the approved field catalog"
                    ),
                )

        filter_expression = metric_config.get("filter", "")
        if isinstance(filter_expression, str) and filter_expression:
            _validate_metric_filters(
                issues,
                metric_name=metric_name,
                filter_expression=filter_expression,
                available_fields=approved_fields,
            )
    return issues


def _validate_required_ih_fields(approved_fields: set[str] | None) -> list[ValidationIssue]:
    if approved_fields is None:
        return []
    missing_fields = [
        field_name
        for field_name in REQ_IH_COLUMNS
        if field_name not in approved_fields
    ]
    if not missing_fields:
        return []
    return [ValidationIssue(
        severity="error",
        section="ih",
        path="ih",
        message="Required Interaction History fields are missing: " + ", ".join(missing_fields),
        step_hint="Required Fields",
    )]


def _validate_report_group_by(
        report_name: str,
        report: dict,
        cfg: dict,
) -> list[ValidationIssue]:
    metric_name = report.get("metric", "")
    if not metric_name or metric_name not in cfg.get("metrics", {}):
        return []

    catalog = build_metric_field_catalog(cfg, metric_name)
    valid_dimensions = set(catalog["dimensions"])
    unknown_group_by = sorted(
        [
            field_name
            for field_name in _clean_list(report.get("group_by", []))
            if field_name not in valid_dimensions
        ],
        key=str.casefold,
    )
    if not unknown_group_by:
        return []
    return [ValidationIssue(
        severity="error",
        section="reports",
        path=f"reports.{report_name}.group_by",
        message=(
                f"Report '{report_name}' group_by fields must come from the report metric group_by "
                "or metrics.global_filters: " + ", ".join(unknown_group_by)
        ),
        step_hint="Reports",
    )]


def _validate_reports(cfg: dict) -> list[ValidationIssue]:
    issues = []
    reports = cfg.get("reports", {})
    if not isinstance(reports, dict):
        return [
            ValidationIssue(
                severity="error",
                section="reports",
                path="reports",
                message="The reports section must be a table.",
                step_hint="Reports",
            )
        ]

    for report_name, report in reports.items():
        if not isinstance(report, dict):
            issues.append(ValidationIssue(
                severity="error",
                section="reports",
                path=f"reports.{report_name}",
                message=f"Report '{report_name}' must be a table.",
                step_hint="Reports",
            ))
            continue

        state = deserialize_report(report_name, report)
        for message in validate_report_state(state, cfg):
            severity: Severity = (
                "warning"
                if "cannot be edited visually" in message
                else "error"
            )
            issues.append(ValidationIssue(
                severity=severity,
                section="reports",
                path=f"reports.{report_name}",
                message=f"Report '{report_name}': {message}",
                step_hint="Reports",
            ))
        issues.extend(_validate_report_group_by(report_name, report, cfg))
    return issues


def validate_config(
        cfg: dict,
        *,
        approved_fields: Iterable[str] | None = None,
        runtime_fields: Iterable[str] | None = None,
) -> list[ValidationIssue]:
    """Validate a dashboard config against known fields and config-derived report metadata.

    `approved_fields` represents the post-preprocessing field catalog available to metrics
    and reports. `runtime_fields` represents the raw/defaulted fields available before the
    Interaction History runtime derives time fields and ActionID.
    """
    approved_field_set = _field_set(approved_fields)
    runtime_field_set = _field_set(runtime_fields)
    issues: list[ValidationIssue] = []
    issues.extend(_validate_runtime_fields(cfg, runtime_field_set))
    issues.extend(_validate_required_ih_fields(approved_field_set))
    issues.extend(_validate_metrics(cfg, approved_field_set))
    issues.extend(_validate_reports(cfg))
    return issues


def has_blocking_issues(issues: Iterable[ValidationIssue]) -> bool:
    return any(issue.is_blocking for issue in issues)
