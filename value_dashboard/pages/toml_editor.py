import ast
import numbers
import os
from traceback import print_stack

import pandas as pd
import streamlit as st
import tomlkit
from streamlit_tags import st_tags

from value_dashboard.pipeline import holdings
from value_dashboard.utils.config import get_config
from value_dashboard.utils.string_utils import strtobool, isBool


def clear_config_cache():
    get_config.clear()
    holdings.get_reports_data.clear()


# Function to display and edit dictionary in a structured way
def display_dict(name, data, read_only=False, level=0):
    updated_data = {}
    for key, value in data.items():
        if key == "scores":
            ro = True
        else:
            ro = read_only
        if isinstance(value, dict):
            if key == "default_values":
                st.markdown(f"{'######'} **{key}**:")
                updated_data[key] = display_dict_as_table(value, read_only=ro)
            else:
                st.markdown(f"{'##'.join(map(lambda x: x * level, 'map'))} **{key}**:")
                updated_data[key] = display_dict(key, value, ro, level + 1)
        else:
            if key == "file_type":
                file_types = ("parquet", "pega_ds_export")
                updated_data[key] = st.selectbox(
                    f"{'' * level * 2}- {name}-{key}",
                    file_types,
                    index=file_types.index(value),
                    disabled=ro,
                )
            elif isBool(value):
                updated_data[key] = st.checkbox(
                    label=f"{' ' * level * 2}- {name}-{key}",
                    value=strtobool(value),
                    disabled=ro,
                )
            elif isinstance(value, numbers.Number):
                updated_data[key] = st.number_input(
                    f"{'###' * level * 2} {name}-{key}", value=value, disabled=ro
                )
            elif isinstance(value, list):
                key_label = f"{'###' * level * 2} {name}-{key}"
                if ro:
                    updated_data[key] = value
                    st.multiselect(
                        label=key_label, options=value, default=value, disabled=ro
                    )
                else:
                    updated_data[key] = st_tags(
                        label=key_label, text="", value=value, key=key_label
                    )
            else:
                if len(str(value)) < 80:
                    updated_data[key] = st.text_input(
                        f"{' ' * level * 2}- {name}-{key}", str(value), disabled=ro
                    )
                else:
                    updated_data[key] = st.text_area(
                        label=f"{' ' * level * 2}- {name}-{key}",
                        value=str(value),
                        disabled=ro,
                    )

    return updated_data


# Function to display reports as editable table
def display_reports(metrics, reports):
    report_data = []
    metrics_list = [k for k in metrics.keys() if isinstance(metrics[k], dict)]
    report_types = set([k["type"] for k in reports.values()]).union(
        ["line", "bar_polar", "treemap", "heatmap", "gauge", "boxplot", "histogram", "scatter", "funnel"]
    )
    for key, report in reports.items():
        other_params = {
            k: v
            for k, v in report.items()
            if k not in ["metric", "type", "description", "group_by"]
        }
        report_data.append(
            [
                key,
                report.get("metric", ""),
                report.get("type", ""),
                report.get("description", ""),
                str(report.get("group_by", [])),
                str(other_params),
            ]
        )

    df = pd.DataFrame(
        report_data,
        columns=["id", "metric", "type", "description", "group_by", "others"],
    )
    edited_df = st.data_editor(
        df,
        column_config={
            "type": st.column_config.SelectboxColumn(
                "type",
                help="Select plot type",
                options=report_types,
                required=True,
            ),
            "metric": st.column_config.SelectboxColumn(
                help="Metric to be used",
                options=metrics_list,
                required=True,
            ),
        },
        num_rows="dynamic",
    )
    edited_df.set_index("id", inplace=True)
    edited_dict = edited_df[
        ["metric", "type", "description", "group_by", "others"]
    ].to_dict("index")
    for key in edited_dict.keys():
        val = edited_dict[key]
        report = val
        group_by = report.get("group_by", [])
        if group_by:
            report["group_by"] = ast.literal_eval(group_by)
    for key in edited_dict.keys():
        val = edited_dict[key]
        report = val
        others = report.pop("others", None)
        if others:
            others = ast.literal_eval(others)
            for k_o in others.keys():
                report[k_o] = others[k_o]
    return edited_dict


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


with st.sidebar:
    st.button("Clear config cache 🗑️", on_click=lambda: clear_config_cache())

tabs = ["📄 Configuration", "📝 Readme"]
conf, readme = st.tabs(tabs)

with conf:
    st.title("Application Configuration Editor")

    try:
        config = get_config()
    except:
        print_stack()
        st.error("Error reading TOML file. Please validate config file.")
        st.stop()
    # Display and edit copyright section
    st.subheader(
        "Branding Section",
        help="These settings provide versioning and release information about the application.",
    )
    config["copyright"] = display_dict(
        "copyright", config.get("copyright", {}), read_only=False
    )

    # Display and edit UX section
    st.subheader(
        "UX Section",
        help="""These settings control the behavior of the application's user interface.
    
    **refresh_dashboard**: A boolean-like string that indicates whether the dashboard should automatically refresh. Possible values are "true" or "false".
    
    **refresh_interval**: The time interval (in milliseconds) for refreshing the dashboard automatically. The default value is 120000, which equals 2 minutes.""",
    )
    config["ux"] = display_dict("ux", config.get("ux", {}))

    # Display and edit IH section
    st.subheader(
        "IH Section",
        help="""These settings define how input data files (usually Interaction History exports) are processed and recognized by the
    application.
    - **file_type**: The expected type of input data files. The default setting is "parquet", indicating that files should
      be in Apache Parquet format.
    - **file_pattern**: A glob pattern used to locate data files within the directory structure. For example: "**/*
      .parquet", which recursively searches for all files with a .parquet extension.
    - **ih_group_pattern**: A regular expression pattern used to extract date or identifier information from file names.
      E.g. "ih_(\\d{8})", which captures date in YYYYMMDD format.
    - **streaming**: Process the polars query in batches to handle larger-than-memory data. If set to False (default), the
      entire query is processed in a single batch. Should be changed to `true` if dataset files are larger than few GBs.
    - **background**: Run the polars query in the background and return execution. Currently, all initial load frames are
      lazy frames, collected asynchronously.
    - **extensions**: Additional file handling extensions or configurations that might be added later.
    """,
    )
    config["ih"] = display_dict("ih", config.get("ih", {}))

    # Display Metrics section (Read-Only)
    st.subheader(
        "Metrics Section",
        help="""The application currently supports various metrics. Please refer to README on metrics' details. 
    
    **Scores** parameter cannot be changed!!!""",
    )
    config["metrics"] = display_dict(
        "metrics", config.get("metrics", {}), read_only=False
    )

    # Display Reports section (Editable)
    st.subheader(
        "Reports Section",
        help="""The [reports] section in the configuration file allows for the definition of various analytical reports. 
    Each report is configured to display specific metrics and visualizations based on the application's requirements. 
    These configurations can be added or modified without changing the underlying code, providing flexibility in reporting.
    
    Each report in the configuration file shares a set of common properties that establish its metric, type, description, grouping, and visual attributes. 
    These properties provide a consistent structure for defining various reports and ensure that data is presented effectively.""",
    )
    metrics = config.get("metrics", {})
    config["reports"] = display_reports(metrics, config.get("reports", {}))

    # Display Variants section
    st.subheader(
        "Variants Section",
        help="Provides metadata and contextual information about the dashboard configuration. "
             "This section is informational and does not directly impact the functionality of the dashboard.",
    )
    config["variants"] = display_dict(
        "variants", config.get("variants", {}), read_only=False
    )

    # Display and edit chat with data section
    st.subheader(
        "Chat with data Section",
        help="The `[chat_with_data]` section used to configure integration with a chatbot for questions on the data and visualizations beyond ad-hoc reports and queries configured for dashboard.",
    )
    config["chat_with_data"] = display_dict(
        "chat_with_data", config.get("chat_with_data", {})
    )

    try:
        text = tomlkit.dumps(config)
        st.download_button(
            "Download config file", data=text, file_name="config.toml", type="primary"
        )
    except:
        print_stack()
    # pprint.pprint(config)

with readme:
    with open(os.path.join(os.path.dirname(__file__), "../../README.md"), "r") as f:
        readme_line = f.readlines()
        readme_buffer = []

    for line in readme_line:
        readme_buffer.append(line)

    st.markdown("".join(readme_buffer))
