from typing import Dict, List, Optional


REPORT_RECIPES: Dict[str, dict] = {
    "line": {
        "label": "Line / Bar",
        "metric_prefixes": ["engagement", "conversion", "model_ml_scores"],
        "type": "line",
        "required_fields": ["x", "y"],
        "group_by_fields": ["x", "color", "facet_row", "facet_column"],
    },
    "gauge": {
        "label": "Gauge",
        "metric_prefixes": ["engagement", "conversion"],
        "type": "gauge",
        "required_fields": ["value"],
        "group_by_fields": [],
    },
    "treemap": {
        "label": "Treemap",
        "metric_prefixes": ["engagement", "conversion", "model_ml_scores"],
        "type": "treemap",
        "required_fields": ["color"],
        "group_by_fields": [],
    },
    "heatmap": {
        "label": "Heatmap",
        "metric_prefixes": ["engagement", "conversion", "model_ml_scores"],
        "type": "heatmap",
        "required_fields": ["x", "y", "color"],
        "group_by_fields": ["x", "y"],
    },
    "scatter": {
        "label": "Scatter",
        "metric_prefixes": ["engagement", "conversion", "model_ml_scores"],
        "type": "scatter",
        "required_fields": ["x", "y", "size", "color", "animation_frame", "animation_group"],
        "group_by_fields": ["color", "animation_frame", "animation_group"],
    },
    "bar_polar": {
        "label": "Polar Bar",
        "metric_prefixes": ["engagement", "conversion"],
        "type": "bar_polar",
        "required_fields": ["r", "theta", "color"],
        "group_by_fields": ["theta", "color"],
    },
    "descriptive_line": {
        "label": "Descriptive Line",
        "metric_prefixes": ["descriptive"],
        "type": "line",
        "required_fields": ["x", "property", "score"],
        "group_by_fields": ["x", "color", "facet_row", "facet_column"],
    },
    "descriptive_boxplot": {
        "label": "Descriptive Box Plot",
        "metric_prefixes": ["descriptive"],
        "type": "boxplot",
        "required_fields": ["x", "property"],
        "group_by_fields": ["x", "color", "facet_row", "facet_column"],
    },
    "descriptive_histogram": {
        "label": "Descriptive Histogram",
        "metric_prefixes": ["descriptive"],
        "type": "histogram",
        "required_fields": ["property"],
        "group_by_fields": ["facet_row", "facet_column"],
    },
    "descriptive_heatmap": {
        "label": "Descriptive Heatmap",
        "metric_prefixes": ["descriptive"],
        "type": "heatmap",
        "required_fields": ["x", "y", "property", "score"],
        "group_by_fields": ["x", "y"],
    },
    "descriptive_funnel": {
        "label": "Descriptive Funnel",
        "metric_prefixes": ["descriptive"],
        "type": "funnel",
        "required_fields": ["x", "color", "stages"],
        "group_by_fields": ["color", "facet_row", "facet_column"],
    },
    "experiment_z_score": {
        "label": "Experiment Z-Score",
        "metric_prefixes": ["experiment"],
        "type": "line",
        "required_fields": ["y"],
        "group_by_fields": ["y", "facet_row", "facet_column"],
    },
    "experiment_odds_ratio": {
        "label": "Experiment Odds Ratio",
        "metric_prefixes": ["experiment"],
        "type": "line",
        "required_fields": ["x", "y"],
        "group_by_fields": ["y", "facet_row", "facet_column"],
    },
    "clv_histogram": {
        "label": "CLV Histogram",
        "metric_prefixes": ["clv"],
        "type": "histogram",
        "required_fields": ["x"],
        "group_by_fields": ["color", "facet_row", "facet_column"],
    },
    "clv_treemap": {
        "label": "CLV Treemap",
        "metric_prefixes": ["clv"],
        "type": "treemap",
        "required_fields": [],
        "group_by_fields": [],
    },
    "clv_exposure": {
        "label": "Customer Exposure",
        "metric_prefixes": ["clv"],
        "type": "exposure",
        "required_fields": [],
        "group_by_fields": [],
    },
    "clv_corr": {
        "label": "CLV Correlation",
        "metric_prefixes": ["clv"],
        "type": "corr",
        "required_fields": ["x", "y"],
        "group_by_fields": [],
    },
    "clv_model": {
        "label": "CLV Model",
        "metric_prefixes": ["clv"],
        "type": "model",
        "required_fields": [],
        "group_by_fields": [],
    },
    "clv_rfm_density": {
        "label": "RFM Density",
        "metric_prefixes": ["clv"],
        "type": "rfm_density",
        "required_fields": [],
        "group_by_fields": [],
    },
}


def get_recipe(recipe_key: str) -> dict:
    return REPORT_RECIPES[recipe_key]


def recipe_supports_metric(recipe_key: str, metric_name: str) -> bool:
    recipe = REPORT_RECIPES[recipe_key]
    return any(metric_name.startswith(prefix) for prefix in recipe["metric_prefixes"])


def get_supported_recipes(metric_name: str) -> List[str]:
    return [
        recipe_key for recipe_key in REPORT_RECIPES
        if recipe_supports_metric(recipe_key, metric_name)
    ]


def get_default_recipe(metric_name: str) -> Optional[str]:
    if metric_name.startswith("engagement"):
        return "line"
    if metric_name.startswith("conversion"):
        return "line"
    if metric_name.startswith("model_ml_scores"):
        return "line"
    if metric_name.startswith("descriptive"):
        return "descriptive_line"
    if metric_name.startswith("experiment"):
        return "experiment_z_score"
    if metric_name.startswith("clv"):
        return "clv_histogram"
    return None


def detect_recipe(metric_name: str, report: dict) -> Optional[str]:
    report_type = report.get("type")
    x_axis = report.get("x")

    if metric_name.startswith("descriptive"):
        if report_type == "line":
            return "descriptive_line"
        if report_type == "boxplot":
            return "descriptive_boxplot"
        if report_type == "histogram":
            return "descriptive_histogram"
        if report_type == "heatmap":
            return "descriptive_heatmap"
        if report_type == "funnel":
            return "descriptive_funnel"
        return None

    if metric_name.startswith("experiment"):
        if x_axis == "z_score":
            return "experiment_z_score"
        if isinstance(x_axis, str) and (x_axis.startswith("g") or x_axis.startswith("chi2")):
            return "experiment_odds_ratio"
        return None

    if metric_name.startswith("clv"):
        if report_type == "histogram":
            return "clv_histogram"
        if report_type == "treemap":
            return "clv_treemap"
        if report_type == "exposure":
            return "clv_exposure"
        if report_type == "corr":
            return "clv_corr"
        if report_type == "model":
            return "clv_model"
        if report_type == "rfm_density":
            return "clv_rfm_density"
        return None

    if report_type in {"line", "gauge", "treemap", "heatmap", "scatter", "bar_polar"}:
        return report_type
    return None
