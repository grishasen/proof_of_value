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

from value_dashboard.utils.config import set_config


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
        edited_df = st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        edited_df = st.data_editor(
            df, num_rows="dynamic", hide_index=True, use_container_width=True
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

        reports = cfg.get("reports", {})
        metrics_options = list(cfg["metrics"].keys())
        if "global_filters" in metrics_options:
            metrics_options.remove("global_filters")

        report_data = []
        for name, rep in reports.items():
            report_data.append({
                "Name": name,
                "Metric": rep.get("metric", ""),
                "Type": rep.get("type", ""),
                "Description": rep.get("description", ""),
                "Group By": ", ".join(rep.get("group_by", [])) if isinstance(rep.get("group_by", []),
                                                                             list) else rep.get(
                    "group_by", ""),
            })

        df = pd.DataFrame(report_data)
        st.write("### Available Reports")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No reports defined yet.")

        st.write("---")
        st.write("### Create/Edit Report Configuration")

        if "selected_report" not in st.session_state:
            st.session_state.selected_report = None
        if "report_copy_template" not in st.session_state:
            st.session_state.report_copy_template = None
        if "report_form_mode" not in st.session_state:
            st.session_state.report_form_mode = "edit"

        col1, col2 = st.columns(2)
        with col1:
            selected_report = st.selectbox(
                "Edit Existing Report (or select to prefill form below)",
                ["<New Report>"] + list(reports.keys()),
                key="report_edit_select"
            )
            if selected_report != "<New Report>":
                if st.button("Load Report for Editing", key="load_report"):
                    st.session_state.selected_report = selected_report
                    st.session_state.report_form_mode = "edit"
            else:
                if st.button("New Report", key="new_report"):
                    st.session_state.selected_report = None
                    st.session_state.report_copy_template = None
                    st.session_state.report_form_mode = "edit"
        with col2:
            if selected_report == "<New Report>" and len(reports) > 0:
                copy_template = st.selectbox("Copy definition from report", ["<None>"] + list(reports.keys()),
                                             key="report_copy_select")
                if st.button("Copy Report Definition", key="copy_report_btn") and copy_template != "<None>":
                    st.session_state.report_copy_template = copy_template
                    st.session_state.report_form_mode = "copy"

        if st.session_state.report_form_mode == "edit":
            if st.session_state.selected_report and st.session_state.selected_report in reports:
                rep = reports[st.session_state.selected_report]
                edit_mode = True
                default_name = st.session_state.selected_report
            else:
                rep = {}
                edit_mode = False
                default_name = str(uuid.uuid4())
        elif st.session_state.report_form_mode == "copy" and st.session_state.report_copy_template in reports:
            rep = copy.deepcopy(reports[st.session_state.report_copy_template])
            default_name = st.session_state.report_copy_template + '_' + str(uuid.uuid4())
            edit_mode = False
        else:
            rep = {}
            edit_mode = False
            default_name = st.session_state.report_copy_template + '_' + str(uuid.uuid4())

        st.write("#### Report Details")
        with st.form(key="report_form", clear_on_submit=False):
            name = st.text_input("Report Name", value=default_name)
            metric = st.selectbox("Metric", metrics_options,
                                  index=metrics_options.index(rep.get("metric", metrics_options[0])) if rep.get(
                                      "metric") in metrics_options else 0)
            rtype = st.text_input("Type", value=rep.get("type", ""))
            desc = st.text_area("Description", value=rep.get("description", ""))
            group_by = st.text_area("Group By (one per line)",
                                    value="\n".join(rep.get("group_by", [])) if isinstance(rep.get("group_by", []),
                                                                                           list) else rep.get(
                                        "group_by",
                                        ""))
            other_fields = {}
            for k, v in rep.items():
                if k in ["metric", "type", "description", "group_by"]: continue
                if isinstance(v, list):
                    val = st.text_area(f"{k} (list, one per line)", value="\n".join(str(x) for x in v))
                    other_fields[k] = [x.strip() for x in val.split("\n") if x.strip()]
                elif isinstance(v, bool):
                    val = st.checkbox(k, value=v)
                    other_fields[k] = val
                elif isinstance(v, (int, float)):
                    val = st.number_input(k, value=v)
                    other_fields[k] = val
                elif isinstance(v, dict):
                    st.markdown(f"{'######'} **{k}**:")
                    other_fields[k] = display_dict_as_table(v, read_only=False)
                else:
                    val = st.text_input(k, value=str(v))
                    other_fields[k] = val

            submit = st.form_submit_button("Save Report")
            delete = st.form_submit_button("Delete Report") if edit_mode else None

            if submit:
                if not name:
                    st.error("Report name is required.")
                else:
                    group_by_list = [x.strip() for x in group_by.split("\n") if x.strip()]
                    new_report = {
                        "metric": metric,
                        "type": rtype,
                        "description": desc,
                        "group_by": group_by_list,
                        **other_fields
                    }
                    reports[name] = new_report
                    cfg["reports"] = reports
                    st.success(f"Report '{name}' saved.")
                    st.session_state.selected_report = name
                    st.session_state.report_copy_template = None
                    st.session_state.report_form_mode = "edit"

            if delete and edit_mode:
                if name in reports:
                    del reports[name]
                    cfg["reports"] = reports
                    st.success(f"Report '{name}' deleted.")
                    st.session_state.selected_report = None

        st.info(
            "Select a report to edit, or choose <New Report> to create a new one. "
            "Use the 'Copy definition from report' dropdown to use an existing report as a template."
        )

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
