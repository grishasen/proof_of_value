import copy
import uuid

import pandas as pd
import streamlit as st
import tomlkit

from value_dashboard.utils.py_utils import isBool, strtobool


def _parse_list(value) -> list:
    """Support manual list editing through a simple multiline text area."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    return []


def _render_dict_editor(value: dict, key_prefix: str) -> dict:
    """Edit arbitrary dict fields without constraining their schema."""
    rows = [{"Name": key, "Value": val} for key, val in value.items()]
    st.caption("Add or update key/value pairs. Empty keys are ignored when saving.")
    edited_df = st.data_editor(
        pd.DataFrame(rows, columns=["Name", "Value"]),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=key_prefix,
    )
    result = {}
    for _, row in edited_df.iterrows():
        name = row.get("Name")
        if pd.isna(name) or str(name).strip() == "":
            continue
        result[str(name)] = row.get("Value")
    return result


def _render_raw_value(label: str, value, key_prefix: str):
    """Pick a generic widget for unsupported raw TOML fields while keeping their values intact."""
    if isinstance(value, list):
        val = st.text_area(
            label,
            value="\n".join(str(item) for item in value),
            key=key_prefix,
            help="Edit one list item per line. The builder converts this back into a TOML array.",
        )
        return _parse_list(val)
    if isinstance(value, dict):
        return _render_dict_editor(value, key_prefix)
    if isinstance(value, bool) or isBool(value):
        return st.checkbox(
            label,
            value=strtobool(value),
            key=key_prefix,
            help="Toggle the raw boolean value stored in this report field.",
        )
    if isinstance(value, int):
        return int(
            st.number_input(
                label,
                value=value,
                step=1,
                key=key_prefix,
                help="Edit the numeric value exactly as it should be stored in TOML.",
            )
        )
    if isinstance(value, float):
        return st.number_input(
            label,
            value=value,
            format="%.6f",
            key=key_prefix,
            help="Edit the numeric value exactly as it should be stored in TOML.",
        )
    value = "" if value is None else str(value)
    if len(value) > 80:
        return st.text_area(
            label,
            value=value,
            key=key_prefix,
            help="Edit the raw text value stored in this report field.",
        )
    return st.text_input(
        label,
        value=value,
        key=key_prefix,
        help="Edit the raw text value stored in this report field.",
    )


def render_raw_report_editor(cfg: dict, report_name: str, report: dict, original_name: str, key_base: str):
    """Expose the underlying report fields for unsupported or hand-authored TOML reports."""
    st.info("Raw mode preserves unsupported fields and manual TOML edits.")

    working_report = copy.deepcopy(report)
    edited_report = {}

    name = st.text_input(
        "Report Name",
        value=report_name,
        key=f"{key_base}_raw_name",
        help="Unique report key under [reports.<name>] in the TOML file.",
    )
    for key, value in working_report.items():
        edited_report[key] = _render_raw_value(key, value, f"{key_base}_raw_{key}")

    st.write("### Generated TOML")
    st.code(tomlkit.dumps({"reports": {name: edited_report}}), language="toml")

    if st.button(
            "Save Report",
            key=f"{key_base}_raw_save",
            type="primary",
            help="Write the raw field values back into the current config draft.",
    ):
        if not name:
            st.error("Report name is required.")
            return
        reports = cfg.setdefault("reports", {})
        if name != original_name and name in reports:
            st.error(f"Report '{name}' already exists.")
            return

        if original_name in reports and original_name != name:
            del reports[original_name]
        reports[name] = edited_report
        cfg["reports"] = reports
        st.session_state.rb_selected_report = name
        st.session_state.rb_draft_report = None
        st.session_state.rb_editor_token = uuid.uuid4().hex[:8]
        st.success(f"Report '{name}' saved.")
