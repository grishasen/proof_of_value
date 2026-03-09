import copy
import os
import re
import uuid
from datetime import datetime, date

import pandas as pd
import polars as pl
import streamlit as st
import tomlkit
from streamlit_tags import st_tags

from value_dashboard.report_builder import render_report_builder
from value_dashboard.utils.config import set_config
from value_dashboard.utils.py_utils import isBool


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


def render_value(key, value, path=""):
    """Render an appropriate Streamlit widget for the value, and return updated value."""
    label = f"{path}.{key}" if path else key
    if key == "file_type":
        file_types = ("parquet", "pega_ds_export", "gzip")
        return st.selectbox(
            label,
            file_types,
            index=file_types.index(value)
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
            new_dt = st.datetime_input(label, value=dt_val)
            return date_to_str(new_dt)
        else:
            new_dt = st.date_input(label, value=dt_val or date.today())
            return date_to_str(new_dt)
    if isinstance(value, bool):
        return st.checkbox(label, value=value)
    elif isinstance(value, int):
        return st.number_input(label, value=value, step=1)
    elif isinstance(value, float):
        return st.number_input(label, value=value, format="%.6f")
    elif isinstance(value, list):
        new_val = st_tags(
            label=label, text="", value=value, key=label + " (list)"
        )
        return parse_list(new_val)
    elif isinstance(value, dict):
        return render_section(value, path=label)
    elif isinstance(value, str):
        if value.lower() in ("true", "false"):
            return st.checkbox(label, value=value.lower() == "true")
        if len(str(value)) < 80:
            return st.text_input(label, value)
        else:
            return st.text_area(label, value, height=204)
    elif isinstance(value, pl.expr.expr.Expr):
        return st.text_input(label, str(value))
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
        elif isinstance(v, dict) and k == 'default_values':
            st.markdown(f"{'######'} **{k}**:")
            updated[k] = display_dict_as_table(v, read_only=False)
        elif (k == "filter" or k == "columns") and "extensions" in path:
            updated[k] = st.text_area(k, value=str(v), key=f"{path}.{k}")
        else:
            updated[k] = render_value(k, v, path)
    return updated


def render_report(report, metrics_options, report_name=None):
    st.write(f"Report: {report_name or '<New>'}")
    metric = st.selectbox("Metric", metrics_options,
                          index=metrics_options.index(report.get("metric", metrics_options[0])) if report.get(
                              "metric") else 0)
    rtype = st.text_input("Type", value=report.get("type", ""))
    desc = st.text_area("Description", value=report.get("description", ""))
    group_by = report.get("group_by", [])
    group_by_val = st.text_area("Group By (one per line)",
                                value="\n".join(group_by) if isinstance(group_by, list) else group_by)
    new_report = dict(report)
    new_report["metric"] = metric
    new_report["type"] = rtype
    new_report["description"] = desc
    new_report["group_by"] = parse_list(group_by_val)
    for k, v in report.items():
        if k in ["metric", "type", "description", "group_by"]:
            continue
        new_report[k] = render_value(k, v, path=f"reports.{report_name or '<new>'}")
    return new_report


def render_config_editor(cfg):
    st.set_page_config(page_title="Config Editor", layout="wide")
    st.title("🔧 Visual Config File Editor")

    tabs = st.tabs(
        ["Branding", "UX", "Interaction History", "Holdings", "Metrics", "Variants", "Chat with Data", "Report Builder",
         "Save & Export"])

    with tabs[0]:
        st.header("Branding (copyright)")
        branding = cfg.get("copyright", {})
        new_branding = render_section(branding, "copyright")
        cfg["copyright"] = new_branding

    with tabs[1]:
        st.header("UX")
        ux = cfg.get("ux", {})
        new_ux = render_section(ux, "ux")
        cfg["ux"] = new_ux

    with tabs[2]:
        st.header("Interaction History (IH)")
        ih = cfg.get("ih", {})
        new_ih = render_section(ih, "ih")
        cfg["ih"] = new_ih

    with tabs[3]:
        st.header("Holdings")
        holdings = cfg.get("holdings", {})
        new_holdings = render_section(holdings, "holdings")
        cfg["holdings"] = new_holdings

    with tabs[4]:
        st.header("Metrics (Scores are read-only)")
        metrics = cfg.get("metrics", {})
        updated_metrics = {}
        for k, v in metrics.items():
            st.subheader(k)
            if isinstance(v, dict):
                updated_metrics[k] = render_section(v, f"metrics.{k}")
            else:
                updated_metrics[k] = render_value(k, v, "metrics")
        cfg["metrics"] = updated_metrics

    with tabs[5]:
        st.header("Variants")
        variants = cfg.get("variants", {})
        new_variants = render_section(variants, "variants")
        cfg["variants"] = new_variants

    with tabs[6]:
        st.header("Chat With Data")
        chat = cfg.get("chat_with_data", {})
        new_chat = render_section(chat, "chat_with_data")
        cfg["chat_with_data"] = new_chat

    with tabs[7]:
        st.header("Report Configuration Builder")
        render_report_builder(cfg)

    with tabs[8]:
        st.header("Save & Export")
        cfg = serialize_exprs(cfg)
        if st.button("Apply New Config", type='primary'):
            new_config_text = tomlkit.dumps(cfg)
            try:
                os.makedirs("temp_configs")
            except FileExistsError:
                pass
            cfg_file_name = "temp_configs/" + "config_" + uuid.uuid4().hex + '.toml'
            with open(cfg_file_name, "w") as f:
                f.write(new_config_text)

            set_config(cfg_file_name)
            st.success("All changes saved!")

        st.download_button(
            "Download Config",
            data=tomlkit.dumps(cfg),
            file_name="config.toml",
            mime="text/plain"
        )
