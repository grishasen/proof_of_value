import copy
from typing import Any, Dict

from value_dashboard.report_builder.models import ReportBuilderState
from value_dashboard.report_builder.recipes import detect_recipe, get_recipe


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _parse_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    return []


def deserialize_report(report_name: str, report: dict) -> ReportBuilderState:
    metric_name = report.get("metric", "")
    chart_key = detect_recipe(metric_name, report)
    state = ReportBuilderState(
        name=report_name,
        description=report.get("description", ""),
        metric=metric_name,
        type=report.get("type", ""),
        chart_key=chart_key or "",
        group_by=_parse_list(report.get("group_by", [])),
        mode="visual" if chart_key else "raw",
        reason="" if chart_key else "This report is not supported by the visual editor.",
    )

    known_keys = {"metric", "type", "description", "group_by"}

    if chart_key == "line":
        state.x = report.get("x")
        state.y = report.get("y")
        state.color = report.get("color")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        state.log_y = _parse_bool(report.get("log_y", False))
        known_keys.update({"x", "y", "color", "facet_row", "facet_column", "log_y"})
    elif chart_key == "gauge":
        state.value = report.get("value")
        state.reference = copy.deepcopy(report.get("reference", {}))
        known_keys.update({"value", "reference"})
    elif chart_key == "treemap":
        state.color = report.get("color")
        known_keys.update({"color"})
    elif chart_key == "heatmap":
        state.x = report.get("x")
        state.y = report.get("y")
        state.color = report.get("color")
        known_keys.update({"x", "y", "color"})
    elif chart_key == "scatter":
        state.x = report.get("x")
        state.y = report.get("y")
        state.color = report.get("color")
        state.size = report.get("size")
        state.animation_frame = report.get("animation_frame")
        state.animation_group = report.get("animation_group")
        state.log_x = _parse_bool(report.get("log_x", False))
        state.log_y = _parse_bool(report.get("log_y", False))
        known_keys.update(
            {"x", "y", "color", "size", "animation_frame", "animation_group", "log_x", "log_y"}
        )
    elif chart_key == "bar_polar":
        state.r = report.get("r")
        state.theta = report.get("theta")
        state.color = report.get("color")
        state.showlegend = _parse_bool(report.get("showlegend", False))
        known_keys.update({"r", "theta", "color", "showlegend"})
    elif chart_key == "descriptive_line":
        state.x = report.get("x")
        state.property = report.get("y")
        state.score = report.get("score")
        state.color = report.get("color")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        known_keys.update({"x", "y", "score", "color", "facet_row", "facet_column"})
    elif chart_key == "descriptive_boxplot":
        state.x = report.get("x")
        state.property = report.get("y")
        state.color = report.get("color")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        known_keys.update({"x", "y", "color", "facet_row", "facet_column"})
    elif chart_key == "descriptive_histogram":
        state.property = report.get("x")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        known_keys.update({"x", "facet_row", "facet_column"})
    elif chart_key == "descriptive_heatmap":
        state.x = report.get("x")
        state.y = report.get("y")
        state.property = report.get("property")
        state.score = report.get("score")
        known_keys.update({"x", "y", "property", "score"})
    elif chart_key == "descriptive_funnel":
        state.x = report.get("x")
        state.color = report.get("color")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        state.stages = _parse_list(report.get("stages", []))
        known_keys.update({"x", "color", "facet_row", "facet_column", "stages"})
    elif chart_key == "experiment_z_score":
        state.x = "z_score"
        state.y = report.get("y")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        known_keys.update({"x", "y", "facet_row", "facet_column"})
    elif chart_key == "experiment_odds_ratio":
        state.x = report.get("x")
        state.y = report.get("y")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        known_keys.update({"x", "y", "facet_row", "facet_column"})
    elif chart_key == "clv_histogram":
        state.x = report.get("x")
        state.color = report.get("color")
        state.facet_row = report.get("facet_row")
        state.facet_column = report.get("facet_column")
        known_keys.update({"x", "color", "facet_row", "facet_column"})
    elif chart_key == "clv_corr":
        state.x = report.get("x")
        state.y = report.get("y")
        known_keys.update({"x", "y"})

    state.extras = {
        key: copy.deepcopy(value)
        for key, value in report.items()
        if key not in known_keys
    }
    return state


def _assign_if_value(payload: Dict[str, Any], key: str, value: Any):
    if value in (None, "", [], {}):
        return
    payload[key] = value


def _merge_group_by(state: ReportBuilderState) -> list:
    recipe = get_recipe(state.chart_key)
    group_by = list(state.group_by)
    for field_name in recipe.get("group_by_fields", []):
        value = getattr(state, field_name, None)
        if value not in (None, "", []) and value not in group_by:
            group_by.append(value)
    return group_by


def serialize_report_state(state: ReportBuilderState) -> dict:
    report = copy.deepcopy(state.extras)
    report["metric"] = state.metric
    report["type"] = get_recipe(state.chart_key)["type"]
    report["description"] = state.description
    report["group_by"] = _merge_group_by(state)

    if state.chart_key == "line":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.y)
        _assign_if_value(report, "color", state.color)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
        if state.log_y:
            report["log_y"] = "true"
    elif state.chart_key == "gauge":
        report["value"] = state.value
        report["reference"] = state.reference
    elif state.chart_key == "treemap":
        _assign_if_value(report, "color", state.color)
    elif state.chart_key == "heatmap":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.y)
        _assign_if_value(report, "color", state.color)
    elif state.chart_key == "scatter":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.y)
        _assign_if_value(report, "color", state.color)
        _assign_if_value(report, "size", state.size)
        _assign_if_value(report, "animation_frame", state.animation_frame)
        _assign_if_value(report, "animation_group", state.animation_group)
        report["log_x"] = "true" if state.log_x else "false"
        report["log_y"] = "true" if state.log_y else "false"
    elif state.chart_key == "bar_polar":
        _assign_if_value(report, "r", state.r)
        _assign_if_value(report, "theta", state.theta)
        _assign_if_value(report, "color", state.color)
        report["showlegend"] = "true" if state.showlegend else "false"
    elif state.chart_key == "descriptive_line":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.property)
        _assign_if_value(report, "score", state.score)
        _assign_if_value(report, "color", state.color)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
    elif state.chart_key == "descriptive_boxplot":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.property)
        _assign_if_value(report, "color", state.color)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
    elif state.chart_key == "descriptive_histogram":
        _assign_if_value(report, "x", state.property)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
    elif state.chart_key == "descriptive_heatmap":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.y)
        _assign_if_value(report, "property", state.property)
        _assign_if_value(report, "score", state.score)
    elif state.chart_key == "descriptive_funnel":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "color", state.color)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
        report["stages"] = list(state.stages)
    elif state.chart_key == "experiment_z_score":
        report["x"] = "z_score"
        _assign_if_value(report, "y", state.y)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
    elif state.chart_key == "experiment_odds_ratio":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.y)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
    elif state.chart_key == "clv_histogram":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "color", state.color)
        _assign_if_value(report, "facet_row", state.facet_row)
        _assign_if_value(report, "facet_column", state.facet_column)
    elif state.chart_key == "clv_treemap":
        pass
    elif state.chart_key == "clv_exposure":
        pass
    elif state.chart_key == "clv_corr":
        _assign_if_value(report, "x", state.x)
        _assign_if_value(report, "y", state.y)
    elif state.chart_key == "clv_model":
        pass
    elif state.chart_key == "clv_rfm_density":
        pass

    return report
