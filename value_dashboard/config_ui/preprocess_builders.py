import ast
import re

import polars as pl
import streamlit as st

from value_dashboard.config_generator.preprocess import build_calculated_fields_config_text, compile_filter_rules

FILTER_OPERATORS = [
    "==", "!=", ">", ">=", "<", "<=", "contains", "starts with", "in", "not in", "is null", "is not null"
]


def blank_default_row() -> dict:
    return {"Field": "", "Default Value": "", "Enabled": True}


def blank_filter_row() -> dict:
    return {"Field": "", "Operator": "==", "Value": "", "Enabled": True}


def blank_calculated_row() -> dict:
    return {"Name": "", "Expression": "", "Enabled": True}


def default_rows_from_dict(default_values: dict) -> list[dict]:
    return [
        {"Field": key, "Default Value": value, "Enabled": True}
        for key, value in default_values.items()
    ] or [blank_default_row()]


def build_default_values_map(default_rows: list[dict]) -> dict:
    result = {}
    for row in default_rows:
        if not row.get("Enabled", True):
            continue
        field_name = str(row.get("Field", "")).strip()
        if not field_name:
            continue
        result[field_name] = row.get("Default Value", "")
    return result


def _frame_records(frame) -> list[dict]:
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    return frame.to_dict("records")


def _is_missing_editor_value(value) -> bool:
    return value is None or value != value


def normalize_rows(frame) -> list[dict]:
    rows = _frame_records(frame)
    return [
        {key: ("" if _is_missing_editor_value(value) else value) for key, value in row.items()}
        for row in rows
    ]


def editor_frame(rows: list[dict], columns: list[str], blank_row_factory) -> pl.DataFrame:
    editor_rows = rows or [blank_row_factory()]
    return pl.DataFrame({
        column: [
            (
                bool(row.get(column, False))
                if column == "Enabled"
                else ("" if _is_missing_editor_value(row.get(column, "")) else str(row.get(column, "")))
            )
            for row in editor_rows
        ]
        for column in columns
    })


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


def parse_simple_filter_rules(filter_text: str) -> list[dict] | None:
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


def stringify_columns_value(columns_value) -> str:
    if columns_value is None:
        return ""
    if isinstance(columns_value, str):
        return columns_value
    return str(columns_value)


def parse_calculated_rows(columns_value) -> list[dict] | None:
    raw_text = stringify_columns_value(columns_value).strip()
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


def _init_defaults_editor_state(source_key: str, rows_key: str, default_values):
    if not isinstance(default_values, dict):
        st.session_state.pop(rows_key, None)
        st.session_state[source_key] = None
        return
    source_value = tuple(
        (str(key), str(value))
        for key, value in sorted(default_values.items(), key=lambda item: str(item[0]).casefold())
    )
    if st.session_state.get(source_key) == source_value and rows_key in st.session_state:
        return
    st.session_state[rows_key] = default_rows_from_dict(default_values)
    st.session_state[source_key] = source_value


def _init_filter_editor_state(source_key: str, mode_key: str, rows_key: str, raw_key: str, filter_value: str):
    source_value = str(filter_value or "")
    if st.session_state.get(source_key) == source_value and rows_key in st.session_state:
        return
    parsed_rows = parse_simple_filter_rules(source_value)
    if parsed_rows is None and source_value.strip():
        st.session_state[mode_key] = "Raw Polars"
        st.session_state[rows_key] = [blank_filter_row()]
        st.session_state[raw_key] = source_value
    else:
        st.session_state[mode_key] = "Rules"
        st.session_state[rows_key] = parsed_rows or [blank_filter_row()]
        st.session_state[raw_key] = source_value
    st.session_state[source_key] = source_value


def _init_calculated_editor_state(source_key: str, mode_key: str, rows_key: str, raw_key: str, columns_value):
    source_value = stringify_columns_value(columns_value)
    if st.session_state.get(source_key) == source_value and rows_key in st.session_state:
        return
    parsed_rows = parse_calculated_rows(columns_value)
    if parsed_rows is None and source_value.strip():
        st.session_state[mode_key] = "Raw Expressions"
        st.session_state[rows_key] = [blank_calculated_row()]
        st.session_state[raw_key] = source_value
    else:
        st.session_state[mode_key] = "Table"
        st.session_state[rows_key] = parsed_rows or [blank_calculated_row()]
        st.session_state[raw_key] = source_value
    st.session_state[source_key] = source_value


def _append_suggested_row(rows_key: str, blank_row_factory, target_key: str, field_name: str):
    rows = list(st.session_state.get(rows_key) or [])
    rows.append({**blank_row_factory(), target_key: field_name})
    st.session_state[rows_key] = rows


def _render_suggested_field_picker(
        title: str,
        suggestions: list[str],
        rows_key: str,
        blank_row_factory,
        target_key: str,
        widget_key_prefix: str,
):
    if not suggestions:
        return
    select_col, button_col = st.columns([0.75, 0.25], vertical_alignment="bottom")
    selected_field = select_col.selectbox(
        title,
        options=suggestions,
        help="Pick a known field and insert it into the editor, or type a custom field name directly in the table.",
        key=f"{widget_key_prefix}_suggested_field",
    )
    if button_col.button("Insert Field", key=f"{widget_key_prefix}_insert_field"):
        _append_suggested_row(rows_key, blank_row_factory, target_key, selected_field)
        st.rerun()


@st.fragment()
def render_defaults_builder(
        *,
        title: str,
        caption: str,
        default_values,
        rows_key: str,
        source_key: str,
        editor_key: str,
        field_suggestions: list[str] | None = None,
        allow_custom_fields: bool = False,
        extensions: dict,
):
    if isinstance(default_values, dict):
        _init_defaults_editor_state(source_key, rows_key, default_values)
    with st.container(border=True):
        st.write(f"### {title}")
        st.caption(caption)
        if allow_custom_fields:
            _render_suggested_field_picker(
                "Suggested Fields",
                field_suggestions or [],
                rows_key,
                blank_default_row,
                "Field",
                editor_key,
            )
        if isinstance(default_values, dict):
            default_frame = editor_frame(
                st.session_state.get(rows_key, default_rows_from_dict(default_values)),
                ["Field", "Default Value", "Enabled"],
                blank_default_row,
            )
            edited_defaults = st.data_editor(
                default_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=editor_key,
                column_config={
                    "Field": st.column_config.TextColumn(
                        "Field",
                        help="Column to fill or create.",
                        width="medium",
                    ),
                    "Default Value": st.column_config.TextColumn(
                        "Default Value",
                        help="Literal value. Examples: N/A, 0.0, true, 1e-10.",
                        width="medium",
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                },
            )
            st.session_state[rows_key] = normalize_rows(edited_defaults)
            st.session_state[source_key] = tuple(
                (str(key), str(value))
                for key, value in sorted(build_default_values_map(st.session_state[rows_key]).items(),
                                         key=lambda item: str(item[0]).casefold())
            )
            extensions["default_values"] = build_default_values_map(st.session_state[rows_key])


@st.fragment()
def render_filter_builder(
        *,
        title: str,
        caption: str,
        filter_value: str,
        field_options: list[str] | None,
        mode_key: str,
        rows_key: str,
        raw_key: str,
        source_key: str,
        editor_key: str,
        raw_editor_key: str,
        field_suggestions: list[str] | None = None,
        allow_custom_fields: bool = False,
        extensions: dict,
):
    _init_filter_editor_state(source_key, mode_key, rows_key, raw_key, filter_value)
    with st.container(border=True):
        st.write(f"### {title}")
        st.caption(caption)
        st.session_state[mode_key] = st.segmented_control(
            "Filter Mode",
            options=["Rules", "Raw Polars"],
            selection_mode="single",
            default=st.session_state[mode_key],
            key=f"{editor_key}_mode",
            help="Rules are easier to author; raw mode gives full control over the final Polars filter expression.",
        )
        if st.session_state[mode_key] == "Rules":
            if allow_custom_fields:
                _render_suggested_field_picker(
                    "Suggested Fields",
                    field_suggestions or [],
                    rows_key,
                    blank_filter_row,
                    "Field",
                    editor_key,
                )
            filter_rows_frame = editor_frame(
                st.session_state[rows_key],
                ["Field", "Operator", "Value", "Enabled"],
                blank_filter_row,
            )
            field_column = (
                st.column_config.TextColumn(
                    "Field",
                    help="Type any field name, or insert a suggested field above.",
                    width="medium",
                )
                if allow_custom_fields or not field_options
                else st.column_config.SelectboxColumn(
                    "Field",
                    options=field_options,
                    required=False,
                    width="medium",
                )
            )
            edited_filters = st.data_editor(
                filter_rows_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=editor_key,
                column_config={
                    "Field": field_column,
                    "Operator": st.column_config.SelectboxColumn(
                        "Operator",
                        options=FILTER_OPERATORS,
                        required=False,
                        width="small",
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value",
                        help="Comma-separated values for in/not in.",
                        width="large",
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                },
            )
            st.session_state[rows_key] = normalize_rows(edited_filters)
            compiled_filter = compile_filter_rules(st.session_state[rows_key])
            st.session_state[source_key] = compiled_filter
            extensions["filter"] = (
                compile_filter_rules(st.session_state[rows_key])
                if st.session_state[mode_key] == "Rules"
                else st.session_state[raw_key]
            )
            st.caption("Compiled filter")
            st.code(compiled_filter or "pl.lit(True)", language="python", wrap_lines=True, line_numbers=True)
        else:
            st.session_state[raw_key] = st.text_area(
                "Raw Polars Filter",
                value=st.session_state[raw_key],
                key=raw_editor_key,
                height=220,
                help="Full Polars predicate expression written back to config as-is.",
            )
            st.session_state[source_key] = st.session_state[raw_key]
            extensions["filter"] = (
                compile_filter_rules(st.session_state[rows_key])
                if st.session_state[mode_key] == "Rules"
                else st.session_state[raw_key]
            )
            st.caption('Example: pl.col("Outcome").is_in(["Pending", "Impression", "Clicked"])')


@st.fragment()
def render_calculated_fields_builder(
        *,
        title: str,
        caption: str,
        columns_value,
        mode_key: str,
        rows_key: str,
        raw_key: str,
        source_key: str,
        editor_key: str,
        raw_editor_key: str,
        field_suggestions: list[str] | None = None,
        allow_custom_fields: bool = False,
        allow_raw_mode: bool = True,
        extensions: dict,
):
    _init_calculated_editor_state(source_key, mode_key, rows_key, raw_key, columns_value)
    with st.container(border=True):
        title_col, _, example_col = st.columns([0.4, 0.3, 0.3], vertical_alignment="center")
        with title_col:
            st.write(f"### {title}")
        with example_col:
            with st.popover("Examples", icon=":material/flare:"):
                st.code(
                    """pl.when(pl.col("CustomerID").str.slice(0, 1) == "C").then(pl.lit("Customers known")).otherwise(pl.lit("Device/Anonymous")).alias("CustomerType")""",
                    language="python",
                )
                st.caption("Enter the expression body. The builder will alias it to the field name automatically.")
        st.caption(caption)
        if allow_raw_mode:
            st.session_state[mode_key] = st.segmented_control(
                "Calculated Fields Mode",
                options=["Table", "Raw Expressions"],
                selection_mode="single",
                default=st.session_state[mode_key],
                key=f"{editor_key}_mode",
                help="Table mode is easier to author; raw mode preserves manual expressions exactly.",
            )
        else:
            st.session_state[mode_key] = "Table"
        if st.session_state[mode_key] == "Table":
            if allow_custom_fields:
                _render_suggested_field_picker(
                    "Suggested Field Names",
                    field_suggestions or [],
                    rows_key,
                    blank_calculated_row,
                    "Name",
                    editor_key,
                )
            calculated_frame = editor_frame(
                st.session_state[rows_key],
                ["Name", "Expression", "Enabled"],
                blank_calculated_row,
            )
            edited_calculated = st.data_editor(
                calculated_frame,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=editor_key,
                column_config={
                    "Name": st.column_config.TextColumn(
                        "Name",
                        help=(
                            "Type any generated field name, or insert a suggested field name above when you want to "
                            "override or derive from a known field."
                        ),
                        width="small",
                    ),
                    "Expression": st.column_config.TextColumn(
                        "Expression",
                        help=(
                            "Polars expression body using pl and np. Do not add .alias(...) unless you want to "
                            "fully control the expression."
                        ),
                        width="large",
                    ),
                    "Enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                },
            )
            st.session_state[rows_key] = normalize_rows(edited_calculated)
            compiled_calc = build_calculated_fields_config_text(st.session_state[rows_key])
            st.session_state[source_key] = compiled_calc
            extensions["columns"] = (
                build_calculated_fields_config_text(st.session_state[rows_key])
                if st.session_state[mode_key] == "Table"
                else st.session_state[raw_key]
            )
            st.caption("Compiled calculated fields")
            st.code(compiled_calc or "[]", language="python", wrap_lines=True, line_numbers=True)
        else:
            st.session_state[raw_key] = st.text_area(
                "Raw Calculated Fields",
                value=st.session_state[raw_key],
                key=raw_editor_key,
                height=240,
                help="This value is written back to config as-is.",
            )
            st.session_state[source_key] = st.session_state[raw_key]
            extensions["columns"] = st.session_state[raw_key]
