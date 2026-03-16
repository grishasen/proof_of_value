import re
from datetime import datetime, date

import pandas as pd
import polars as pl
import streamlit as st

from value_dashboard.utils.common_constants import CONFIG_FILE_TYPES


def serialize_exprs(obj):
    if isinstance(obj, dict):
        return {k: serialize_exprs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_exprs(item) for item in obj]
    elif isinstance(obj, pl.Expr):
        return str(obj)
    else:
        return obj


def is_date_field(key, value):
    key_lower = key.lower()
    if any(s in key_lower for s in ["date", "time", "datetime", "timestamp"]):
        return True

    if isinstance(value, str):
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # 2024-07-06
            r"^\d{8}$",  # 20240706
            r"^\d{4}/\d{2}/\d{2}$",  # 2024/07/06
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}.*",  # 2024-07-06T12:00:00
            r"^\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}.*",  # 20240706T120000
        ]
        return any(re.match(p, value) for p in date_patterns)
    return False


def parse_date_str(val):
    """Try to parse various date strings to datetime.date or datetime."""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H%M%S.%fZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(val, fmt)
            if fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
                return dt.date()
            return dt
        except Exception:
            continue
    return None


def date_to_str(dt):
    """Format date/datetime object to string."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    elif isinstance(dt, date):
        return dt.strftime("%Y-%m-%d")
    return str(dt)


def parse_list(val):
    if isinstance(val, list):
        return val
    elif isinstance(val, str):
        vals = [x.strip() for x in val.replace("\n", ",").split(",") if x.strip()]
        return vals
    else:
        return []


def _unique_non_empty(values):
    result = []
    for value in values or []:
        if value in (None, ""):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if value not in result:
            result.append(value)
    return result


def ensure_metric_group_by(metric_name, selected_group_by, current_group_by=None, global_filters=None, available_fields=None):
    """Ensure metric group_by keeps at least one field and provide a UI-safe fallback."""
    cleaned_selection = _unique_non_empty(selected_group_by)
    if cleaned_selection:
        return cleaned_selection, None

    previous_selection = _unique_non_empty(current_group_by)
    if previous_selection:
        return previous_selection, (
            f"{metric_name}.group_by must contain at least one field. Keeping the previous selection."
        )

    cleaned_global_filters = _unique_non_empty(global_filters)
    if cleaned_global_filters:
        fallback = [cleaned_global_filters[0]]
        return fallback, (
            f"{metric_name}.group_by must contain at least one field. "
            f"Using `{cleaned_global_filters[0]}` from global_filters."
        )

    cleaned_available_fields = _unique_non_empty(available_fields)
    if cleaned_available_fields:
        fallback = [cleaned_available_fields[0]]
        return fallback, (
            f"{metric_name}.group_by must contain at least one field. "
            f"Using `{cleaned_available_fields[0]}` as a fallback."
        )

    return [], f"{metric_name}.group_by must contain at least one field."


def find_metrics_without_group_by(cfg):
    """Return metric section names that still have an empty or missing group_by list."""
    metrics = cfg.get("metrics", {})
    invalid_metrics = []
    for metric_name, metric_config in metrics.items():
        if metric_name == "global_filters" or not isinstance(metric_config, dict):
            continue
        if "group_by" not in metric_config:
            continue
        if not _unique_non_empty(metric_config.get("group_by", [])):
            invalid_metrics.append(metric_name)
    return invalid_metrics


def render_value(key, value, path=""):
    """Render an appropriate Streamlit widget for the value, and return updated value."""
    ui_key = f"{path}.{key}" if path else key
    label = key
    if key == "file_type":
        return st.selectbox(
            label,
            CONFIG_FILE_TYPES,
            index=CONFIG_FILE_TYPES.index(value),
            key=ui_key,
        )
    if is_date_field(key, value):
        dt_val = None
        if isinstance(value, (datetime, date)):
            dt_val = value
        elif isinstance(value, str):
            parsed = parse_date_str(value)
            if parsed:
                dt_val = parsed
        if isinstance(dt_val, datetime):
            new_dt = st.datetime_input(label, value=dt_val, key=ui_key)
            return date_to_str(new_dt)
        else:
            new_dt = st.date_input(label, value=dt_val or date.today(), key=ui_key)
            return date_to_str(new_dt)
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=ui_key)
    elif isinstance(value, int):
        return st.number_input(label, value=value, step=1, width=200, key=ui_key)
    elif isinstance(value, float):
        return st.number_input(label, value=value, format="%.6f", width=200, key=ui_key)
    elif isinstance(value, list):
        new_val = st.multiselect(
            label=label, options=value, key=ui_key + " (list)", default=value,
            accept_new_options=True
        )
        return parse_list(new_val)
    elif isinstance(value, dict):
        return render_section(value, path=label)
    elif isinstance(value, str):
        if value.lower() in ("true", "false"):
            return st.checkbox(label, value=value.lower() == "true", key=ui_key)
        if len(str(value)) < 80:
            return st.text_input(label, value, key=ui_key)
        else:
            return st.text_area(label, value, height=204, key=ui_key)
    elif isinstance(value, pl.expr.expr.Expr):
        return st.text_input(label, str(value), key=ui_key)
    else:
        st.warning(f"Unknown type for {label}: {type(value)}")
        return value


def display_dict_as_table(values, read_only=False):
    report_data = []
    for key, val in values.items():
        report_data.append([key, val])

    df = pd.DataFrame(report_data, columns=["Name", "Value"])
    if read_only:
        edited_df = st.dataframe(df, hide_index=True, width='stretch')
    else:
        edited_df = st.data_editor(
            df, num_rows="dynamic", hide_index=True, width='stretch'
        )
    edited_df.set_index("Name", inplace=True)
    return edited_df.to_dict()["Value"]


def render_section(section: dict, path=""):
    """Recursively render all fields in a section and return new section dict."""
    updated = {}
    for k, v in section.items():
        if k == "scores":
            st.markdown(f"**{path}.{k}**: _not editable_")
            st.json(v)
            updated[k] = v
            continue
        if (k == "extensions") and isinstance(v, dict):
            st.markdown(f"{'######'} **{k}**:")
            with st.expander(f"{path}.{k}", expanded=False):
                updated[k] = render_section(v, f"{path}.{k}")
        elif isinstance(v, dict):
            st.markdown(f"{'######'} **{k}**:")
            updated[k] = render_section(v, f"{path}.{k}")
        elif (k == "filter" or k == "columns") and "extensions" in path:
            updated[k] = st.text_area(k, value=str(v), key=f"{path}.{k}")
        else:
            updated[k] = render_value(k, v, path)
    return updated
