import copy
import re
from typing import Any

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
REPORT_REFERENCE_KEYS = {
    "animation_frame",
    "animation_group",
    "color",
    "facet_column",
    "facet_row",
    "group_by",
    "property",
    "r",
    "score",
    "size",
    "stages",
    "theta",
    "value",
    "x",
    "y",
}


def _section(config: dict, section_name: str) -> dict:
    value = config.get(section_name, {})
    return value if isinstance(value, dict) else {}


def _key_diff(base: dict, candidate: dict) -> dict[str, list[str]]:
    base_keys = set(base.keys())
    candidate_keys = set(candidate.keys())
    shared_keys = base_keys & candidate_keys
    return {
        "added": sorted(candidate_keys - base_keys, key=str.casefold),
        "removed": sorted(base_keys - candidate_keys, key=str.casefold),
        "changed": sorted(
            [key for key in shared_keys if base.get(key) != candidate.get(key)],
            key=str.casefold,
        ),
        "unchanged": sorted(
            [key for key in shared_keys if base.get(key) == candidate.get(key)],
            key=str.casefold,
        ),
    }


def build_ai_config_diff(template_config: dict, generated_sections: dict) -> dict[str, dict[str, list[str]]]:
    return {
        "metrics": _key_diff(
            _section(template_config, "metrics"),
            _section(generated_sections, "metrics"),
        ),
        "reports": _key_diff(
            _section(template_config, "reports"),
            _section(generated_sections, "reports"),
        ),
        "variants": _key_diff(
            _section(template_config, "variants"),
            _section(generated_sections, "variants"),
        ),
    }


def generated_metric_names(generated_sections: dict) -> list[str]:
    metrics = _section(generated_sections, "metrics")
    return sorted(
        [metric_name for metric_name in metrics.keys() if metric_name != "global_filters"],
        key=str.casefold,
    )


def generated_report_names(generated_sections: dict) -> list[str]:
    return sorted(_section(generated_sections, "reports").keys(), key=str.casefold)


def reports_for_metrics(generated_sections: dict, selected_metrics: list[str]) -> list[str]:
    selected_metric_set = set(selected_metrics)
    reports = _section(generated_sections, "reports")
    return sorted(
        [
            report_name
            for report_name, report_config in reports.items()
            if isinstance(report_config, dict) and report_config.get("metric") in selected_metric_set
        ],
        key=str.casefold,
    )


def _extract_filter_fields(filter_expression: str) -> list[str]:
    if not filter_expression:
        return []
    pattern = r"""pl\.col\(\s*["']([^"']+)["']\s*\)"""
    return re.findall(pattern, str(filter_expression))


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict) or value in (None, ""):
        return []
    return [value]


def collect_referenced_fields(generated_sections: dict) -> list[str]:
    referenced = []
    metrics = _section(generated_sections, "metrics")
    for metric_name, metric_config in metrics.items():
        if metric_name == "global_filters":
            referenced.extend(_as_list(metric_config))
            continue
        if not isinstance(metric_config, dict):
            continue
        referenced.extend(_as_list(metric_config.get("group_by")))
        referenced.extend(_as_list(metric_config.get("columns")))
        for key in METRIC_FIELD_REFERENCE_KEYS:
            referenced.extend(_as_list(metric_config.get(key)))
        referenced.extend(_extract_filter_fields(metric_config.get("filter", "")))

    reports = _section(generated_sections, "reports")
    for report_config in reports.values():
        if not isinstance(report_config, dict):
            continue
        for key in REPORT_REFERENCE_KEYS:
            referenced.extend(_as_list(report_config.get(key)))

    return sorted(
        {
            str(field_name)
            for field_name in referenced
            if not isinstance(field_name, (dict, list, tuple, set)) and field_name not in (None, "")
        },
        key=str.casefold,
    )


def filter_generated_sections(
        generated_sections: dict,
        *,
        selected_metrics: list[str],
        selected_reports: list[str],
) -> dict:
    filtered_sections = copy.deepcopy(generated_sections)
    metrics = _section(filtered_sections, "metrics")
    reports = _section(filtered_sections, "reports")
    selected_metric_set = set(selected_metrics)
    selected_report_set = set(selected_reports)

    filtered_metrics = {}
    if "global_filters" in metrics:
        filtered_metrics["global_filters"] = metrics["global_filters"]
    filtered_metrics.update({
        metric_name: metric_config
        for metric_name, metric_config in metrics.items()
        if metric_name in selected_metric_set
    })

    filtered_reports = {
        report_name: report_config
        for report_name, report_config in reports.items()
        if (
                report_name in selected_report_set
                and isinstance(report_config, dict)
                and report_config.get("metric") in selected_metric_set
        )
    }

    filtered_sections["metrics"] = filtered_metrics
    filtered_sections["reports"] = filtered_reports
    return filtered_sections
