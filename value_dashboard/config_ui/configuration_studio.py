import os
import uuid
from copy import deepcopy

import streamlit as st
import tomlkit

from value_dashboard.config_ui.preprocess_builders import render_calculated_fields_builder, \
    render_defaults_builder, render_filter_builder
from value_dashboard.report_builder import render_report_builder
from value_dashboard.utils.config import set_config
from value_dashboard.utils.config_builder import render_section, render_value, serialize_exprs


def _render_intro():
    st.header("Configuration Editor", divider="red")


def _config_signature(cfg: dict) -> str:
    return tomlkit.dumps(serialize_exprs(deepcopy(cfg)))


def _ensure_working_config(cfg: dict) -> dict:
    working_key = "configuration_editor_working_cfg"
    source_key = "configuration_editor_source_signature"
    incoming_signature = _config_signature(cfg)
    if st.session_state.get(source_key) != incoming_signature or working_key not in st.session_state:
        st.session_state[source_key] = incoming_signature
        st.session_state[working_key] = deepcopy(cfg)
    return st.session_state[working_key]


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


@st.fragment()
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
    default_values = extensions.get("default_values", {})
    if isinstance(default_values, dict):
        rows_key = f"configuration_editor_{section_name}_default_rows"
        render_defaults_builder(
            title=f"{title} Defaults",
            caption="Defaults are applied before filters. They may fill nulls or create missing columns.",
            default_values=default_values,
            rows_key=rows_key,
            source_key=f"configuration_editor_{section_name}_defaults_source",
            editor_key=f"{section_name}.extensions.default_values_editor",
            allow_custom_fields=True,
            extensions=extensions,
        )
    else:
        with st.container(border=True):
            st.write(f"### {title} Defaults")
            st.caption("Defaults are applied before filters. They may fill nulls or create missing columns.")
            extensions["default_values"] = render_value("default_values", default_values, f"{section_name}.extensions")
    section["extensions"] = extensions
    cfg[section_name] = section


def _render_filter_step(cfg: dict, section_name: str, title: str):
    section = _ensure_extensions(cfg, section_name)
    extensions = dict(section.get("extensions", {}))
    mode_key = f"configuration_editor_{section_name}_filter_mode"
    rows_key = f"configuration_editor_{section_name}_filter_rows"
    raw_key = f"configuration_editor_{section_name}_raw_filter"
    render_filter_builder(
        title=f"{title} Filtering",
        caption="Define the dataset-level filters before derived and calculated fields are added.",
        filter_value=str(extensions.get("filter", "")),
        field_options=None,
        mode_key=mode_key,
        rows_key=rows_key,
        raw_key=raw_key,
        source_key=f"configuration_editor_{section_name}_filter_source",
        editor_key=f"{section_name}.extensions.filter_editor",
        raw_editor_key=f"{raw_key}_editor",
        allow_custom_fields=True,
        extensions=extensions,
    )
    section["extensions"] = extensions
    cfg[section_name] = section


def _render_columns_step(cfg: dict, section_name: str, title: str):
    section = _ensure_extensions(cfg, section_name)
    extensions = dict(section.get("extensions", {}))
    mode_key = f"configuration_editor_{section_name}_columns_mode"
    rows_key = f"configuration_editor_{section_name}_calculated_rows"
    raw_key = f"configuration_editor_{section_name}_raw_columns"
    render_calculated_fields_builder(
        title=f"{title} Calculated Fields",
        caption="Create named calculated fields that run after defaults and filters.",
        columns_value=extensions.get("columns", ""),
        mode_key=mode_key,
        rows_key=rows_key,
        raw_key=raw_key,
        source_key=f"configuration_editor_{section_name}_columns_source",
        editor_key=f"{section_name}.extensions.columns_editor",
        raw_editor_key=f"{raw_key}_editor",
        allow_custom_fields=True,
        allow_raw_mode=True,
        extensions=extensions,
    )
    section["extensions"] = extensions
    cfg[section_name] = section


@st.fragment()
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


@st.fragment()
def _render_metrics_step(cfg: dict):
    st.write("### Metrics Review")
    st.caption("Review metric-level grouping, filters, and field references. Scores remain read-only.")

    metrics = cfg.get("metrics", {})
    updated_metrics = {}
    current_global_filters = [field for field in metrics.get("global_filters", [])]
    with st.container(border=True):
        st.subheader("global_filters")
        st.caption("Choose the fields that should appear as top-level metric filters.")
        updated_metrics["global_filters"] = st.multiselect(
            "Global Filters",
            options=current_global_filters,
            default=current_global_filters,
            key="metrics.global_filters",
            help="Choose the fields that should appear as top-level metric filters.",
            accept_new_options=True,
        )

    for key, value in metrics.items():
        if key == "global_filters":
            continue
        st.subheader(key)
        if isinstance(value, dict):
            with st.container(border=True):
                updated_metric = {}
                for metric_key, metric_value in value.items():
                    widget_path = f"config_studio.metrics.{key}"
                    if metric_key == "scores":
                        st.markdown(f"**{metric_key}**: _not editable_")
                        st.json(metric_value)
                        updated_metric[metric_key] = metric_value
                    elif metric_key == "group_by":
                        current_group_by = [field for field in metric_value]
                        updated_metric[metric_key] = st.multiselect(
                            f"{key}.group_by",
                            options=current_group_by,
                            default=current_group_by,
                            key=f"{widget_path}.group_by",
                            help="Only available fields should be used in metric group_by.",
                            accept_new_options=True
                        )
                    elif metric_key == "columns" and isinstance(metric_value, list):
                        current_columns = [field for field in metric_value]
                        updated_metric[metric_key] = st.multiselect(
                            f"{key}.columns",
                            options=current_columns,
                            default=current_columns,
                            key=f"{widget_path}.columns",
                            help="Choose approved descriptive or metric fields only.",
                        )
                    elif metric_key == "filter":
                        filter_key = f"{widget_path}.filter"
                        draft_filter = st.text_area(
                            f"{key}.filter",
                            value=str(metric_value),
                            key=f"{filter_key}_draft",
                            help=(
                                "Use a Polars expression. Any pl.col(...) references must use approved fields only."
                            ),
                            height=160,
                        )
                        updated_metric[metric_key] = draft_filter
                    else:
                        updated_metric[metric_key] = render_value(metric_key, metric_value, widget_path)
                updated_metrics[key] = updated_metric
        else:
            updated_metrics[key] = render_value(key, value, "config_editor.metrics")
    cfg["metrics"] = updated_metrics


@st.fragment()
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
        st.session_state["configuration_editor_working_cfg"] = deepcopy(cfg)
        st.session_state["configuration_editor_source_signature"] = toml_text
        st.success("Configuration activated for the current app session.")


def render_configuration_studio(cfg: dict):
    _render_intro()
    cfg = _ensure_working_config(cfg)

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

    st.session_state["configuration_editor_working_cfg"] = cfg
