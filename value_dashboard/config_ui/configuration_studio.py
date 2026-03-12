import ast
import os
import re
import uuid

import pandas as pd
import streamlit as st
import tomlkit

from value_dashboard.config_generator.preprocess import build_calculated_fields_config_text, compile_filter_rules
from value_dashboard.report_builder import render_report_builder
from value_dashboard.utils.config import set_config
from value_dashboard.utils.config_builder import display_dict_as_table, render_section, render_value, serialize_exprs

FILTER_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "contains", "starts with", "in", "not in", "is null",
                    "is not null"]


def _render_intro():
    st.header("Configuration Editor", divider="red")


def _build_steps(cfg: dict) -> list[tuple[str, str]]:
    step_names = [
        ("branding", "Branding"),
        ("ux", "UX"),
        ("data_runtime", "Data Runtime"),
        ("ih_defaults", "IH Defaults"),
        ("ih_filter", "IH Filtering"),
        ("ih_columns", "IH Calculated Fields"),
    ]
    if "holdings" in cfg:
        step_names.extend([
            ("holdings_defaults", "Holdings Defaults"),
            ("holdings_filter", "Holdings Filtering"),
            ("holdings_columns", "Holdings Calculated Fields"),
        ])
    step_names.extend([
        ("metrics", "Metrics"),
        ("variants", "Variants"),
        ("chat", "Chat with Data"),
        ("reports", "Reports"),
        ("save", "Save & Export"),
    ])
    return [(key, f"{idx}. {label}") for idx, (key, label) in enumerate(step_names, start=1)]


def _blank_filter_row() -> dict:
    return {"Field": "", "Operator": "==", "Value": "", "Enabled": True}


def _blank_calculated_row() -> dict:
    return {"Name": "", "Expression": "", "Enabled": True}


def _normalize_rows(frame: pd.DataFrame) -> list[dict]:
    rows = frame.to_dict("records")
    return [
        {key: ("" if pd.isna(value) else value) for key, value in row.items()}
        for row in rows
    ]


def _editor_frame(rows: list[dict], columns: list[str], blank_row_factory) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame([blank_row_factory()], columns=columns)


def _split_top_level(text: str, separator: str) -> list[str]:
    items = []
    start = 0
    depth = 0
    quote = None
    escape = False
    idx = 0
    while idx < len(text):
        char = text[idx]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
        else:
            if char in {"'", '"'}:
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            elif depth == 0 and text.startswith(separator, idx):
                items.append(text[start:idx].strip())
                idx += len(separator)
                start = idx
                continue
        idx += 1
    items.append(text[start:].strip())
    return [item for item in items if item]


def _strip_outer_parentheses(text: str) -> str:
    candidate = text.strip()
    while candidate.startswith("(") and candidate.endswith(")"):
        inner = candidate[1:-1].strip()
        depth = 0
        quote = None
        escape = False
        balanced = True
        for char in inner:
            if quote:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
            else:
                if char in {"'", '"'}:
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth < 0:
                        balanced = False
                        break
        if not balanced or depth != 0 or quote:
            break
        candidate = inner
    return candidate


def _parse_simple_filter_rules(filter_text: str) -> list[dict] | None:
    raw_text = str(filter_text or "").strip()
    if not raw_text:
        return []
    rows = []

    def _safe_literal_eval(value_text: str):
        try:
            return ast.literal_eval(value_text)
        except (SyntaxError, ValueError):
            raise ValueError

    for clause in _split_top_level(raw_text, " & "):
        expression = _strip_outer_parentheses(clause)
        match = re.fullmatch(r'pl\.col\("([^"]+)"\)\.is_null\(\)', expression)
        if match:
            rows.append({"Field": match.group(1), "Operator": "is null", "Value": "", "Enabled": True})
            continue
        match = re.fullmatch(r'pl\.col\("([^"]+)"\)\.is_not_null\(\)', expression)
        if match:
            rows.append({"Field": match.group(1), "Operator": "is not null", "Value": "", "Enabled": True})
            continue
        match = re.fullmatch(r'pl\.col\("([^"]+)"\)\.cast\(pl\.Utf8\)\.str\.contains\((.+)\)', expression)
        if match:
            try:
                parsed_value = _safe_literal_eval(match.group(2))
            except ValueError:
                return None
            rows.append({
                "Field": match.group(1),
                "Operator": "contains",
                "Value": str(parsed_value),
                "Enabled": True,
            })
            continue
        match = re.fullmatch(r'pl\.col\("([^"]+)"\)\.cast\(pl\.Utf8\)\.str\.starts_with\((.+)\)', expression)
        if match:
            try:
                parsed_value = _safe_literal_eval(match.group(2))
            except ValueError:
                return None
            rows.append({
                "Field": match.group(1),
                "Operator": "starts with",
                "Value": str(parsed_value),
                "Enabled": True,
            })
            continue
        match = re.fullmatch(r'~pl\.col\("([^"]+)"\)\.is_in\((.+)\)', expression)
        if match:
            try:
                values = _safe_literal_eval(match.group(2))
            except ValueError:
                return None
            rows.append({
                "Field": match.group(1),
                "Operator": "not in",
                "Value": ", ".join(map(str, values)),
                "Enabled": True,
            })
            continue
        match = re.fullmatch(r'pl\.col\("([^"]+)"\)\.is_in\((.+)\)', expression)
        if match:
            try:
                values = _safe_literal_eval(match.group(2))
            except ValueError:
                return None
            rows.append({
                "Field": match.group(1),
                "Operator": "in",
                "Value": ", ".join(map(str, values)),
                "Enabled": True,
            })
            continue
        comparison_match = re.fullmatch(r'pl\.col\("([^"]+)"\)\s*(==|!=|>=|<=|>|<)\s*(.+)', expression)
        if comparison_match:
            try:
                parsed_value = _safe_literal_eval(comparison_match.group(3))
            except ValueError:
                return None
            rows.append({
                "Field": comparison_match.group(1),
                "Operator": comparison_match.group(2),
                "Value": str(parsed_value),
                "Enabled": True,
            })
            continue
        return None
    return rows


def _stringify_columns_value(columns_value) -> str:
    if columns_value is None:
        return ""
    if isinstance(columns_value, str):
        return columns_value
    return str(columns_value)


def _parse_calculated_rows(columns_value) -> list[dict] | None:
    raw_text = _stringify_columns_value(columns_value).strip()
    if not raw_text:
        return []
    if not (raw_text.startswith("[") and raw_text.endswith("]")):
        return None
    body = raw_text[1:-1].strip()
    if not body:
        return []
    rows = []
    for expression_text in _split_top_level(body, ","):
        cleaned = expression_text.strip()
        alias_match = re.search(r"\.alias\((['\"])(.*?)\1\)\s*$", cleaned)
        if not alias_match:
            return None
        name = alias_match.group(2)
        if not name or not cleaned:
            return None
        rows.append({"Name": name, "Expression": cleaned, "Enabled": True})
    return rows


def _init_filter_editor_state(section_name: str, filter_value: str):
    source_key = f"configuration_editor_{section_name}_filter_source"
    mode_key = f"configuration_editor_{section_name}_filter_mode"
    rows_key = f"configuration_editor_{section_name}_filter_rows"
    raw_key = f"configuration_editor_{section_name}_raw_filter"
    source_value = str(filter_value or "")
    if st.session_state.get(source_key) == source_value:
        return
    parsed_rows = _parse_simple_filter_rules(source_value)
    if parsed_rows is None and source_value.strip():
        st.session_state[mode_key] = "Raw Polars"
        st.session_state[rows_key] = [_blank_filter_row()]
        st.session_state[raw_key] = source_value
    else:
        st.session_state[mode_key] = "Rules"
        st.session_state[rows_key] = parsed_rows or [_blank_filter_row()]
        st.session_state[raw_key] = source_value
    st.session_state[source_key] = source_value


def _init_calculated_editor_state(section_name: str, columns_value):
    source_key = f"configuration_editor_{section_name}_columns_source"
    mode_key = f"configuration_editor_{section_name}_columns_mode"
    rows_key = f"configuration_editor_{section_name}_calculated_rows"
    raw_key = f"configuration_editor_{section_name}_raw_columns"
    source_value = _stringify_columns_value(columns_value)
    if st.session_state.get(source_key) == source_value:
        return
    parsed_rows = _parse_calculated_rows(columns_value)
    if parsed_rows is None and source_value.strip():
        st.session_state[mode_key] = "Raw Expressions"
        st.session_state[rows_key] = [_blank_calculated_row()]
        st.session_state[raw_key] = source_value
    else:
        st.session_state[mode_key] = "Table"
        st.session_state[rows_key] = parsed_rows or [_blank_calculated_row()]
        st.session_state[raw_key] = source_value
    st.session_state[source_key] = source_value


def _render_simple_section_step(title: str, caption: str, section: dict, path: str) -> dict:
    with st.container(border=True):
        st.write(f"### {title}")
        st.caption(caption)
        return render_section(section, path)


def _render_runtime_card(section: dict, path: str, title: str, group_pattern_key: str) -> dict:
    updated = dict(section)
    runtime_keys = ["file_type", "file_pattern", group_pattern_key, "hive_partitioning", "streaming", "background"]
    present_runtime_keys = [key for key in runtime_keys if key in section]
    if present_runtime_keys:
        with st.container(border=True):
            st.write(f"### {title} Runtime Settings")
            st.caption("Edit the file loading, grouping, and collection settings for this data source.")
            cols = st.columns(3)
            for idx, key in enumerate(present_runtime_keys):
                with cols[idx % 3]:
                    updated[key] = render_value(key, section.get(key), path)
    return updated


def _render_additional_settings_card(
        updated_section: dict,
        original_section: dict,
        path: str,
        title: str,
        excluded_keys: set[str],
):
    remaining_keys = [key for key in original_section.keys() if key not in excluded_keys]
    if not remaining_keys:
        return
    with st.container(border=True):
        st.write(f"### {title} Additional Settings")
        st.caption("These settings are preserved and can still be edited directly.")
        for key in remaining_keys:
            updated_section[key] = render_value(key, original_section.get(key), path)


def _ensure_extensions(cfg: dict, section_name: str) -> dict:
    section = dict(cfg.get(section_name, {}))
    extensions = section.get("extensions", {})
    if not isinstance(extensions, dict):
        extensions = {}
    section["extensions"] = dict(extensions)
    cfg[section_name] = section
    return section


def _render_data_runtime_step(cfg: dict):
    st.write("### Data Runtime")
    st.caption(
        "Interaction History and Holdings runtime settings are grouped here so ingestion-level options live in one place.")

    ih = cfg.get("ih", {})
    cfg["ih"] = _render_runtime_card(ih, "ih", "Interaction History", "ih_group_pattern")
    _render_additional_settings_card(
        cfg["ih"],
        ih,
        "ih",
        "Interaction History",
        {"extensions", "file_type", "file_pattern", "ih_group_pattern", "hive_partitioning", "streaming", "background"},
    )

    if "holdings" in cfg:
        holdings = cfg.get("holdings", {})
        cfg["holdings"] = _render_runtime_card(holdings, "holdings", "Holdings", "file_group_pattern")
        _render_additional_settings_card(
            cfg["holdings"],
            holdings,
            "holdings",
            "Holdings",
            {"extensions", "file_type", "file_pattern", "file_group_pattern", "hive_partitioning", "streaming",
             "background"},
        )
    else:
        st.info("This config does not contain a `holdings` section, so Holdings runtime settings are skipped.")


def _render_defaults_step(cfg: dict, section_name: str, title: str):
    section = _ensure_extensions(cfg, section_name)
    extensions = dict(section.get("extensions", {}))
    with st.container(border=True):
        st.write(f"### {title} Defaults")
        st.caption("Defaults are applied to missing or null values before downstream processing.")
        default_values = extensions.get("default_values", {})
        if isinstance(default_values, dict):
            extensions["default_values"] = display_dict_as_table(default_values, read_only=False)
        else:
            extensions["default_values"] = render_value("default_values", default_values, f"{section_name}.extensions")
    section["extensions"] = extensions
    cfg[section_name] = section


def _render_filter_step(cfg: dict, section_name: str, title: str):
    section = _ensure_extensions(cfg, section_name)
    extensions = dict(section.get("extensions", {}))
    _init_filter_editor_state(section_name, extensions.get("filter", ""))
    with st.container(border=True):
        st.write(f"### {title} Filtering")
        st.caption("Use the table builder for simple filters or switch to raw mode for manual Polars expressions.")
        mode_key = f"configuration_editor_{section_name}_filter_mode"
        rows_key = f"configuration_editor_{section_name}_filter_rows"
        raw_key = f"configuration_editor_{section_name}_raw_filter"
        st.session_state[mode_key] = st.segmented_control(
            "Filter Mode",
            options=["Rules", "Raw Polars"],
            selection_mode="single",
            default=st.session_state[mode_key],
            key=f"{mode_key}_selector",
            help="Rules mode matches the builder in AI Configuration Studio. Raw mode preserves fully custom expressions.",
        )
        if st.session_state[mode_key] == "Rules":
            filter_rows_frame = _editor_frame(
                st.session_state[rows_key],
                ["Field", "Operator", "Value", "Enabled"],
                _blank_filter_row,
            )
            edited_filters = st.data_editor(
                filter_rows_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=f"{section_name}.extensions.filter_editor",
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="medium"),
                    "Operator": st.column_config.SelectboxColumn("Operator", options=FILTER_OPERATORS, width="small"),
                    "Value": st.column_config.TextColumn(
                        "Value",
                        help="Use comma-separated values for `in` and `not in`.",
                        width="large",
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                },
            )
            st.session_state[rows_key] = _normalize_rows(edited_filters)
            extensions["filter"] = compile_filter_rules(st.session_state[rows_key])
            st.caption("Compiled filter")
            st.code(extensions["filter"] or "pl.lit(True)", language="python", wrap_lines=True, line_numbers=True)
        else:
            st.session_state[raw_key] = st.text_area(
                "Raw Polars Filter",
                value=st.session_state[raw_key],
                key=f"{raw_key}_editor",
                height=220,
                help="This expression is written back to the TOML config as-is.",
            )
            extensions["filter"] = st.session_state[raw_key]
    section["extensions"] = extensions
    cfg[section_name] = section


def _render_columns_step(cfg: dict, section_name: str, title: str):
    section = _ensure_extensions(cfg, section_name)
    extensions = dict(section.get("extensions", {}))
    _init_calculated_editor_state(section_name, extensions.get("columns", ""))
    with st.container(border=True):
        st.write(f"### {title} Calculated Fields")
        st.caption("Use the table builder for named calculated fields or switch to raw mode for manual expressions.")
        mode_key = f"configuration_editor_{section_name}_columns_mode"
        rows_key = f"configuration_editor_{section_name}_calculated_rows"
        raw_key = f"configuration_editor_{section_name}_raw_columns"
        st.session_state[mode_key] = st.segmented_control(
            "Calculated Fields Mode",
            options=["Table", "Raw Expressions"],
            selection_mode="single",
            default=st.session_state[mode_key],
            key=f"{mode_key}_selector",
            help="Table mode matches the builder in AI Configuration Studio. Raw mode preserves manual expressions.",
        )
        if st.session_state[mode_key] == "Table":
            calculated_frame = _editor_frame(
                st.session_state[rows_key],
                ["Name", "Expression", "Enabled"],
                _blank_calculated_row,
            )
            edited_calculated = st.data_editor(
                calculated_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=f"{section_name}.extensions.columns_editor",
                column_config={
                    "Name": st.column_config.TextColumn("Name", width="small"),
                    "Expression": st.column_config.TextColumn(
                        "Expression",
                        help=(
                            "Enter the Polars expression body, or keep a full expression with `.alias(...)` when you "
                            "want to preserve an existing custom expression."
                        ),
                        width="large",
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                },
            )
            st.session_state[rows_key] = _normalize_rows(edited_calculated)
            extensions["columns"] = build_calculated_fields_config_text(st.session_state[rows_key])
            st.caption("Compiled calculated fields")
            st.code(extensions["columns"] or "[]", language="python", wrap_lines=True, line_numbers=True)
        else:
            st.session_state[raw_key] = st.text_area(
                "Raw Calculated Fields",
                value=st.session_state[raw_key],
                key=f"{raw_key}_editor",
                height=240,
                help="This value is written back to the TOML config as-is.",
            )
            extensions["columns"] = st.session_state[raw_key]
    section["extensions"] = extensions
    cfg[section_name] = section


def _render_ux_step(cfg: dict):
    ux = cfg.get("ux", {})
    updated = {}
    primary_keys = ["refresh_dashboard", "refresh_interval", "data_cache_hours", "data_profiling"]
    with st.container(border=True):
        st.write("### UX Settings")
        st.caption("Tune dashboard refresh and data-handling behavior.")
        cols = st.columns(4)
        for idx, key in enumerate(primary_keys):
            if key not in ux:
                continue
            with cols[idx // 4]:
                updated[key] = render_value(key, ux.get(key), "ux")

    remaining_keys = [key for key in ux.keys() if key not in set(primary_keys + ["chat_with_data"])]
    if remaining_keys:
        with st.container(border=True):
            st.write("### UX Additional Settings")
            st.caption("These settings are preserved and can still be edited directly.")
            for key in remaining_keys:
                updated[key] = render_value(key, ux.get(key), "ux")

    if "chat_with_data" in ux:
        updated["chat_with_data"] = ux["chat_with_data"]
    cfg["ux"] = updated


def _render_metrics_step(cfg: dict):
    metrics = cfg.get("metrics", {})
    updated_metrics = {}
    st.write("### Metrics")
    st.caption("Review metric-level grouping, filters, and field references. Scores remain read-only.")

    if "global_filters" in metrics:
        with st.container(border=True):
            st.subheader("global_filters")
            st.caption("Choose which fields should be exposed as top-level metric filters.")
            updated_metrics["global_filters"] = render_value("global_filters", metrics.get("global_filters", []),
                                                             "metrics")

    for key, value in metrics.items():
        if key == "global_filters":
            continue
        with st.container(border=True):
            st.subheader(key)
            if isinstance(value, dict):
                updated_metrics[key] = render_section(value, f"metrics.{key}")
            else:
                updated_metrics[key] = render_value(key, value, "metrics")

    if "clv" not in metrics:
        st.info("This config does not contain a `metrics.clv` section, so CLV-specific metric settings are skipped.")

    cfg["metrics"] = updated_metrics


def _render_chat_step(cfg: dict):
    st.write("### Chat with Data")
    st.caption("Manage assistant settings, the agent prompt, and metric descriptions.")

    ux = cfg.get("ux", {})
    if "chat_with_data" in ux:
        with st.container(border=True):
            st.write("### Chat Enablement")
            st.caption("This toggle controls whether the chat experience is available in the app.")
            ux["chat_with_data"] = render_value("chat_with_data", ux.get("chat_with_data", False), "ux")
            cfg["ux"] = ux

    with st.container(border=True):
        st.write("### Chat Settings")
        st.caption("Edit the assistant prompt and supporting metric descriptions.")
        chat = cfg.get("chat_with_data", {})
        cfg["chat_with_data"] = render_section(chat, "chat_with_data")


def _render_reports_step(cfg: dict):
    with st.container(border=True):
        st.write("### Reports")
        st.caption("Use the visual report builder to edit the report catalog without rewriting TOML by hand.")
        render_report_builder(cfg)


def _render_save_step(cfg: dict):
    safe_cfg = serialize_exprs(cfg)
    toml_text = tomlkit.dumps(safe_cfg)

    st.write("### Save & Export")
    st.caption("Review the final TOML and either download it or activate it in the running app.")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Metrics", len(safe_cfg.get("metrics", {})))
    summary_col2.metric("Reports", len(safe_cfg.get("reports", {})))
    summary_col3.metric("Variants", len(safe_cfg.get("variants", {})))

    st.code(toml_text, language="toml", height=520)
    action_col1, action_col2, _ = st.columns([2, 2, 4])
    action_col2.download_button(
        "Download Config",
        data=toml_text,
        file_name="config.toml",
        mime="text/plain",
        type="secondary",
    )
    if action_col1.button("Apply Config In App", type="primary", key="configuration_studio_apply"):
        os.makedirs("temp_configs", exist_ok=True)
        cfg_file_name = os.path.join("temp_configs", f"config_{uuid.uuid4().hex}.toml")
        with open(cfg_file_name, "w") as handle:
            handle.write(toml_text)
        set_config(cfg_file_name)
        st.success("Configuration activated for the current app session.")


def render_configuration_studio(cfg: dict):
    _render_intro()

    steps = _build_steps(cfg)
    step_labels = [label for _, label in steps]
    step_key_by_label = {label: key for key, label in steps}

    if "configuration_studio_step_selector" not in st.session_state:
        st.session_state["configuration_studio_step_selector"] = step_labels[0]
    if st.session_state["configuration_studio_step_selector"] not in step_labels:
        st.session_state["configuration_studio_step_selector"] = step_labels[0]

    selected_step_label = st.segmented_control(
        "Configuration Steps",
        options=step_labels,
        selection_mode="single",
        key="configuration_studio_step_selector",
        label_visibility="hidden"
    )
    selected_step_key = step_key_by_label[selected_step_label]

    if selected_step_key == "branding":
        cfg["copyright"] = _render_simple_section_step(
            "Branding",
            "Edit the copyright and version metadata displayed by the application.",
            cfg.get("copyright", {}),
            "copyright",
        )
    elif selected_step_key == "ux":
        _render_ux_step(cfg)
    elif selected_step_key == "data_runtime":
        _render_data_runtime_step(cfg)
    elif selected_step_key == "ih_defaults":
        _render_defaults_step(cfg, "ih", "Interaction History")
    elif selected_step_key == "ih_filter":
        _render_filter_step(cfg, "ih", "Interaction History")
    elif selected_step_key == "ih_columns":
        _render_columns_step(cfg, "ih", "Interaction History")
    elif selected_step_key == "holdings_defaults":
        _render_defaults_step(cfg, "holdings", "Holdings")
    elif selected_step_key == "holdings_filter":
        _render_filter_step(cfg, "holdings", "Holdings")
    elif selected_step_key == "holdings_columns":
        _render_columns_step(cfg, "holdings", "Holdings")
    elif selected_step_key == "metrics":
        _render_metrics_step(cfg)
    elif selected_step_key == "variants":
        cfg["variants"] = _render_simple_section_step(
            "Variants",
            "Update descriptive metadata and any extra runtime flags stored in the variants section.",
            cfg.get("variants", {}),
            "variants",
        )
    elif selected_step_key == "chat":
        _render_chat_step(cfg)
    elif selected_step_key == "reports":
        _render_reports_step(cfg)
    else:
        _render_save_step(cfg)
