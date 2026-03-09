from typing import Any, Dict, Iterable, List


def stable_dedup(values: Iterable[Any]) -> List[Any]:
    """Preserve config order while removing empty values and duplicates."""
    seen = []
    for value in values:
        if value in (None, "", []):
            continue
        if value not in seen:
            seen.append(value)
    return seen


def get_metric_options(cfg: dict) -> List[str]:
    """List metric names that can be selected by the report builder."""
    metrics = list(cfg.get("metrics", {}).keys())
    if "global_filters" in metrics:
        metrics.remove("global_filters")
    return metrics


def build_metric_field_catalog(cfg: dict, metric_name: str) -> Dict[str, List[str]]:
    """Derive selectable fields strictly from the metrics configuration section."""
    metrics = cfg.get("metrics", {})
    metric_cfg = metrics.get(metric_name, {})
    global_filters = metrics.get("global_filters", [])
    dimensions = stable_dedup(global_filters + metric_cfg.get("group_by", []))

    if metric_name.startswith("experiment"):
        experiment_name = metric_cfg.get("experiment_name")
        experiment_group = metric_cfg.get("experiment_group")
        dimensions = stable_dedup(dimensions + [experiment_name, experiment_group])

    measures = stable_dedup(metric_cfg.get("scores", []))
    properties = stable_dedup(metric_cfg.get("columns", [])) if metric_name.startswith("descriptive") else []

    return {
        "dimensions": dimensions,
        "measures": measures,
        "properties": properties,
    }


def ensure_current_option(options: Iterable[str], current: Any) -> List[Any]:
    """Keep legacy/manual values selectable even if they are missing from the current catalog."""
    result = list(options)
    if current not in (None, "", []) and current not in result:
        result.append(current)
    return result


def ensure_current_options(options: Iterable[str], current_values: Iterable[Any]) -> List[Any]:
    """Keep all current multi-select values visible for backward-compatible editing."""
    result = list(options)
    for value in current_values or []:
        if value not in (None, "", []) and value not in result:
            result.append(value)
    return result
