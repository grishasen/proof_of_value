from typing import Dict

METRIC_DISPLAY: Dict[str, dict] = {
    "engagement": {
        "symbol": ":material/ads_click:",
        "label": "Engagement",
    },
    "conversion": {
        "symbol": ":material/task_alt:",
        "label": "Conversion",
    },
    "model_ml_scores": {
        "symbol": ":material/model_training:",
        "label": "Model ML Scores",
    },
    "descriptive": {
        "symbol": ":material/category_search:",
        "label": "Descriptive Analysis",
    },
    "experiment": {
        "symbol": ":material/science:",
        "label": "Experiment Analysis",
    },
    "clv": {
        "symbol": ":material/account_balance_wallet:",
        "label": "Customer Lifetime Value",
    },
}


def get_metric_display(metric_name: str) -> dict:
    """Return metric display metadata based on the configured metric family."""
    for metric_prefix, display in METRIC_DISPLAY.items():
        if metric_name == metric_prefix or metric_name.startswith(metric_prefix):
            return display
    return {
        "symbol": ":material/insights:",
        "label": metric_name.replace("_", " ").title(),
    }


def get_metric_display_name(metric_name: str, include_symbol: bool = True) -> str:
    """Build a user-facing metric label while keeping the stored TOML metric key unchanged."""
    display = get_metric_display(metric_name)
    if include_symbol and display.get("symbol"):
        return f"{display['symbol']} {display['label']}"
    return display["label"]
