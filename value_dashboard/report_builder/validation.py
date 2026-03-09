from typing import List

from value_dashboard.report_builder.field_catalog import build_metric_field_catalog
from value_dashboard.report_builder.recipes import get_recipe


def _value_in_catalog(value, options) -> bool:
    """Treat empty values as separately validated and otherwise require catalog membership."""
    return value in (None, "", []) or value in options


def validate_report_state(state, cfg: dict) -> List[str]:
    """Validate only against config-derived metadata so manual TOML edits remain compatible."""
    issues = []
    metrics = cfg.get("metrics", {})
    metric_name = state.metric

    if not state.name:
        issues.append("Report name is required.")
    if not metric_name:
        issues.append("Metric is required.")
        return issues
    if metric_name not in metrics:
        issues.append(f"Metric '{metric_name}' is not defined in the metrics section.")
        return issues
    if not state.chart_key:
        issues.append("This report cannot be edited visually. Switch to raw mode.")
        return issues

    catalog = build_metric_field_catalog(cfg, metric_name)
    recipe = get_recipe(state.chart_key)

    if "x" in recipe["required_fields"] and not state.x:
        issues.append("X field is required.")
    if "y" in recipe["required_fields"] and not state.y:
        issues.append("Y field is required.")
    if "value" in recipe["required_fields"] and not state.value:
        issues.append("Value field is required.")
    if "color" in recipe["required_fields"] and not state.color:
        issues.append("Color field is required.")
    if "size" in recipe["required_fields"] and not state.size:
        issues.append("Size field is required.")
    if "animation_frame" in recipe["required_fields"] and not state.animation_frame:
        issues.append("Animation frame is required.")
    if "animation_group" in recipe["required_fields"] and not state.animation_group:
        issues.append("Animation group is required.")
    if "r" in recipe["required_fields"] and not state.r:
        issues.append("Radial measure is required.")
    if "theta" in recipe["required_fields"] and not state.theta:
        issues.append("Theta field is required.")
    if "property" in recipe["required_fields"] and not state.property:
        issues.append("Property field is required.")
    if "score" in recipe["required_fields"] and not state.score:
        issues.append("Score is required.")
    if "stages" in recipe["required_fields"] and not state.stages:
        issues.append("At least one funnel stage is required.")

    dimensions = catalog["dimensions"]
    measures = catalog["measures"]
    properties = catalog["properties"]

    if state.chart_key == "line":
        if not _value_in_catalog(state.x, dimensions):
            issues.append("Line chart X field must come from metric group_by or global_filters.")
        if not _value_in_catalog(state.y, measures):
            issues.append("Line chart Y field must come from metric scores.")
        for field_name in ("color", "facet_row", "facet_column"):
            if not _value_in_catalog(getattr(state, field_name), dimensions):
                issues.append(f"{field_name} must come from metric group_by or global_filters.")
    elif state.chart_key == "gauge":
        if not _value_in_catalog(state.value, measures):
            issues.append("Gauge value must come from metric scores.")
        if len(state.group_by) == 0:
            issues.append("Gauge reports require at least one group_by field.")
        if len(state.group_by) > 2:
            issues.append("Gauge reports support at most two group_by fields.")
        for value in state.group_by:
            if value not in dimensions:
                issues.append("Gauge group_by fields must come from metric group_by or global_filters.")
                break
    elif state.chart_key == "treemap":
        if metric_name.startswith("clv"):
            for value in state.group_by:
                if value not in dimensions:
                    issues.append("CLV treemap group_by fields must come from metric group_by or global_filters.")
                    break
        else:
            if not _value_in_catalog(state.color, measures):
                issues.append("Treemap color must come from metric scores.")
    elif state.chart_key == "heatmap":
        if not _value_in_catalog(state.x, dimensions) or not _value_in_catalog(state.y, dimensions):
            issues.append("Heatmap axes must come from metric group_by or global_filters.")
        if not _value_in_catalog(state.color, measures):
            issues.append("Heatmap color must come from metric scores.")
    elif state.chart_key == "scatter":
        for field_name in ("x", "y", "size"):
            if not _value_in_catalog(getattr(state, field_name), measures):
                issues.append(f"Scatter {field_name} must come from metric scores.")
        for field_name in ("color", "animation_frame", "animation_group"):
            if not _value_in_catalog(getattr(state, field_name), dimensions):
                issues.append(f"Scatter {field_name} must come from metric group_by or global_filters.")
    elif state.chart_key == "bar_polar":
        if not _value_in_catalog(state.r, measures):
            issues.append("Polar chart radial value must come from metric scores.")
        if not _value_in_catalog(state.theta, dimensions) or not _value_in_catalog(state.color, dimensions):
            issues.append("Polar chart theta/color must come from metric group_by or global_filters.")
    elif state.chart_key == "descriptive_line":
        if not _value_in_catalog(state.x, dimensions):
            issues.append("Descriptive line X field must come from metric group_by or global_filters.")
        if not _value_in_catalog(state.property, properties):
            issues.append("Descriptive property must come from metrics.descriptive.columns.")
        if not _value_in_catalog(state.score, measures):
            issues.append("Descriptive score must come from metric scores.")
    elif state.chart_key == "descriptive_boxplot":
        if not _value_in_catalog(state.x, dimensions):
            issues.append("Descriptive boxplot X field must come from metric group_by or global_filters.")
        if not _value_in_catalog(state.property, properties):
            issues.append("Descriptive property must come from metrics.descriptive.columns.")
    elif state.chart_key == "descriptive_histogram":
        if not _value_in_catalog(state.property, properties):
            issues.append("Descriptive histogram property must come from metrics.descriptive.columns.")
    elif state.chart_key == "descriptive_heatmap":
        if not _value_in_catalog(state.x, dimensions) or not _value_in_catalog(state.y, dimensions):
            issues.append("Descriptive heatmap axes must come from metric group_by or global_filters.")
        if not _value_in_catalog(state.property, properties):
            issues.append("Descriptive property must come from metrics.descriptive.columns.")
        if not _value_in_catalog(state.score, measures):
            issues.append("Descriptive score must come from metric scores.")
    elif state.chart_key == "descriptive_funnel":
        if not _value_in_catalog(state.x, properties):
            issues.append("Funnel X field must come from metrics.descriptive.columns.")
        if not _value_in_catalog(state.color, dimensions):
            issues.append("Funnel color must come from metric group_by or global_filters.")
    elif state.chart_key == "experiment_z_score":
        if not _value_in_catalog(state.y, dimensions):
            issues.append("Experiment Y field must come from metric group_by, global_filters or experiment settings.")
    elif state.chart_key == "experiment_odds_ratio":
        if not _value_in_catalog(state.x, measures):
            issues.append("Experiment X field must come from metric scores.")
        if not _value_in_catalog(state.y, dimensions):
            issues.append("Experiment Y field must come from metric group_by, global_filters or experiment settings.")
    elif state.chart_key == "clv_histogram":
        if not _value_in_catalog(state.x, measures):
            issues.append("CLV histogram X field must come from metric scores.")
    elif state.chart_key == "clv_corr":
        if not _value_in_catalog(state.x, measures) or not _value_in_catalog(state.y, measures):
            issues.append("CLV correlation axes must come from metric scores.")

    return issues
