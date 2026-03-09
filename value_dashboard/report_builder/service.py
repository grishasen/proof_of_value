import copy
import uuid

import tomlkit

from value_dashboard.report_builder.field_catalog import get_metric_options
from value_dashboard.report_builder.recipes import detect_recipe, get_default_recipe, get_recipe
from value_dashboard.report_builder.serialization import deserialize_report, serialize_report_state

NEW_REPORT_KEY = "__new_report__"


def build_report_summaries(cfg: dict) -> list[dict]:
    summaries = []
    reports = cfg.get("reports", {})
    for name, report in reports.items():
        metric_name = report.get("metric", "")
        chart_key = detect_recipe(metric_name, report)
        summaries.append({
            "name": name,
            "metric": metric_name,
            "type": report.get("type", ""),
            "description": report.get("description", ""),
            "chart_key": chart_key,
            "mode": "visual" if chart_key else "raw",
        })
    return summaries


def get_default_metric(cfg: dict) -> str:
    metrics = get_metric_options(cfg)
    return metrics[0] if metrics else ""


def build_blank_report(metric_name: str) -> dict:
    return {
        "metric": metric_name,
        "type": "",
        "description": "",
        "group_by": [],
    }


def build_new_report_name(reports: dict, source_name: str | None = None) -> str:
    base_name = source_name + "_" if source_name else "report_"
    name = base_name + uuid.uuid4().hex[:8]
    while name in reports:
        name = base_name + uuid.uuid4().hex[:8]
    return name


def get_editor_seed_report(cfg: dict, selected_report: str, draft_report: dict | None) -> tuple[str, dict]:
    reports = cfg.get("reports", {})
    if selected_report != NEW_REPORT_KEY and selected_report in reports:
        return selected_report, copy.deepcopy(reports[selected_report])
    if draft_report is not None:
        return draft_report["__name__"], copy.deepcopy(draft_report["__report__"])
    metric_name = get_default_metric(cfg)
    report_name = build_new_report_name(reports)
    return report_name, build_blank_report(metric_name)


def build_state(cfg: dict, selected_report: str, draft_report: dict | None):
    report_name, report = get_editor_seed_report(cfg, selected_report, draft_report)
    state = deserialize_report(report_name, report)

    if not state.chart_key and not report.get("type") and state.metric:
        default_recipe = get_default_recipe(state.metric)
        if default_recipe:
            state.chart_key = default_recipe
            state.type = get_recipe(default_recipe)["type"]
            state.mode = "visual"
            state.reason = ""
    return state


def build_toml_preview(state) -> str:
    report = serialize_report_state(state)
    return tomlkit.dumps({"reports": {state.name: report}})
