import ast
import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import streamlit as st

from value_dashboard.metrics.constants import DECISION_TIME, DROP_IH_COLUMNS, OUTCOME_TIME
from value_dashboard.utils.file_utils import read_dataset_export
from value_dashboard.utils.polars_utils import schema_with_unique_counts
from value_dashboard.utils.py_utils import capitalize


def detect_ih_file_settings(file_name: str) -> tuple[str, str]:
    """Infer IH file settings from the uploaded sample name."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".parquet":
        return "parquet", "**/*.parquet"
    return "pega_ds_export", "**/*.zip"


@st.cache_data(show_spinner=False)
def load_ih_sample(file_name: str, file_bytes: bytes) -> pl.DataFrame:
    """Load a single uploaded IH sample and normalize column naming like the IH pipeline."""
    with tempfile.TemporaryDirectory(prefix="config_studio_ih_") as temp_dir:
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as handle:
            handle.write(file_bytes)
        sample = read_dataset_export(file_names=file_name, src_folder=temp_dir, lazy=False)

    dframe_columns = sample.collect_schema().names()
    leave_cols = list(set(dframe_columns).difference(set(DROP_IH_COLUMNS)))
    capitalized = capitalize(leave_cols)
    rename_map = dict(zip(leave_cols, capitalized))
    return sample.select(leave_cols).rename(rename_map)


def _safe_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _eval_polars_expression(expression_text: str):
    return eval(expression_text, {"pl": pl, "np": np})


def compile_filter_rules(filter_rows: list[dict]) -> str:
    """Compile simple rule rows into a raw Polars filter expression string."""
    compiled = []
    for row in filter_rows:
        if not row.get("Enabled", True):
            continue
        field_name = str(row.get("Field", "")).strip()
        operator = str(row.get("Operator", "")).strip()
        raw_value = row.get("Value", "")
        if not field_name or not operator:
            continue
        if operator == "is null":
            compiled.append(f'pl.col("{field_name}").is_null()')
            continue
        if operator == "is not null":
            compiled.append(f'pl.col("{field_name}").is_not_null()')
            continue

        parsed_value = _safe_literal(raw_value)
        value_repr = repr(parsed_value)
        if operator == "==":
            compiled.append(f'pl.col("{field_name}") == {value_repr}')
        elif operator == "!=":
            compiled.append(f'pl.col("{field_name}") != {value_repr}')
        elif operator == ">":
            compiled.append(f'pl.col("{field_name}") > {value_repr}')
        elif operator == ">=":
            compiled.append(f'pl.col("{field_name}") >= {value_repr}')
        elif operator == "<":
            compiled.append(f'pl.col("{field_name}") < {value_repr}')
        elif operator == "<=":
            compiled.append(f'pl.col("{field_name}") <= {value_repr}')
        elif operator == "contains":
            compiled.append(f'pl.col("{field_name}").cast(pl.Utf8).str.contains({value_repr})')
        elif operator == "starts with":
            compiled.append(f'pl.col("{field_name}").cast(pl.Utf8).str.starts_with({value_repr})')
        elif operator == "in":
            values = [_safe_literal(item) for item in str(raw_value).split(",") if str(item).strip()]
            compiled.append(f'pl.col("{field_name}").is_in({repr(values)})')
        elif operator == "not in":
            values = [_safe_literal(item) for item in str(raw_value).split(",") if str(item).strip()]
            compiled.append(f'~pl.col("{field_name}").is_in({repr(values)})')
    return " & ".join(f"({expression})" for expression in compiled)


def compile_calculated_fields(field_rows: list[dict]) -> tuple[list[Any], str]:
    """Turn calculated field rows into executable expressions and config text."""
    expressions = []
    text_rows = []
    for row in field_rows:
        if not row.get("Enabled", True):
            continue
        field_name = str(row.get("Name", "")).strip()
        expression_text = str(row.get("Expression", "")).strip()
        if not field_name or not expression_text:
            continue
        if ".alias(" in expression_text:
            compiled_text = expression_text
        else:
            compiled_text = f"({expression_text}).alias({field_name!r})"
        expressions.append(_eval_polars_expression(compiled_text))
        text_rows.append(compiled_text)
    return expressions, str(text_rows)


def _apply_default_values(dataframe: pl.DataFrame, default_values: dict[str, Any]) -> pl.DataFrame:
    result = dataframe
    for column_name, raw_value in default_values.items():
        if column_name is None or str(column_name).strip() == "":
            continue
        value = _safe_literal(raw_value)
        if column_name not in result.columns:
            result = result.with_columns(pl.lit(value).alias(column_name))
        else:
            result = result.with_columns(pl.col(column_name).fill_null(value).alias(column_name))
    return result


def _alias_time_columns(dataframe: pl.DataFrame, outcome_time_col: str, decision_time_col: str) -> pl.DataFrame:
    expressions = []
    if outcome_time_col and outcome_time_col in dataframe.columns and outcome_time_col != OUTCOME_TIME:
        expressions.append(pl.col(outcome_time_col).alias(OUTCOME_TIME))
    if decision_time_col and decision_time_col in dataframe.columns and decision_time_col != DECISION_TIME:
        expressions.append(pl.col(decision_time_col).alias(DECISION_TIME))
    if expressions:
        dataframe = dataframe.with_columns(expressions)
    return dataframe


def _derive_time_fields(dataframe: pl.DataFrame) -> pl.DataFrame:
    if OUTCOME_TIME not in dataframe.columns or DECISION_TIME not in dataframe.columns:
        return dataframe
    return dataframe.with_columns([
        pl.col(OUTCOME_TIME).str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z", strict=False),
        pl.col(DECISION_TIME).str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z", strict=False),
    ]).with_columns([
        pl.col(OUTCOME_TIME).dt.date().alias("Day"),
        pl.col(OUTCOME_TIME).dt.strftime("%Y-%m").alias("Month"),
        pl.col(OUTCOME_TIME).dt.year().cast(pl.Int16).alias("Year"),
        (pl.col(OUTCOME_TIME).dt.year().cast(pl.Utf8) + "_Q" +
         pl.col(OUTCOME_TIME).dt.quarter().cast(pl.Utf8)).alias("Quarter"),
        (pl.col(OUTCOME_TIME) - pl.col(DECISION_TIME)).dt.total_seconds().alias("ResponseTime"),
    ])


@st.cache_data(show_spinner=False)
def apply_ih_preprocessing(
        file_name: str,
        file_bytes: bytes,
        outcome_time_col: str,
        decision_time_col: str,
        default_values: dict[str, Any],
        filter_expression: str,
        calculated_field_rows: list[dict],
) -> tuple[pl.DataFrame, list[str], str]:
    """Apply generator-side IH preprocessing with the same semantics as the runtime pipeline."""
    dataframe = load_ih_sample(file_name, file_bytes)
    dataframe = _apply_default_values(dataframe, default_values)
    dataframe = _alias_time_columns(dataframe, outcome_time_col, decision_time_col)

    if filter_expression.strip():
        dataframe = dataframe.filter(_eval_polars_expression(filter_expression))

    dataframe = _derive_time_fields(dataframe)
    calculated_expressions, calculated_text = compile_calculated_fields(calculated_field_rows)
    if calculated_expressions:
        dataframe = dataframe.with_columns(calculated_expressions)
    return dataframe, dataframe.columns, calculated_text


def build_schema_preview(dataframe: pl.DataFrame, selected_fields: list[str] | None = None) -> pl.DataFrame:
    """Return a schema profile for the full or field-approved working dataframe."""
    if selected_fields:
        preview_frame = dataframe.select([column for column in selected_fields if column in dataframe.columns])
    else:
        preview_frame = dataframe
    return schema_with_unique_counts(preview_frame).sort("Column")


def build_ih_config(
        template_config: dict,
        file_name: str,
        default_values: dict[str, Any],
        filter_expression: str,
        calculated_fields_text: str,
) -> dict:
    """Build the IH section for the generated config from the approved preprocessing choices."""
    file_type, file_pattern = detect_ih_file_settings(file_name)
    ih_config = copy.deepcopy(template_config["ih"])
    ih_config["file_type"] = file_type
    ih_config["file_pattern"] = file_pattern
    ih_config.setdefault("extensions", {})
    ih_config["extensions"]["default_values"] = default_values
    ih_config["extensions"]["filter"] = filter_expression
    ih_config["extensions"]["columns"] = calculated_fields_text
    return ih_config
