from __future__ import annotations

import ast
import traceback
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from plotly.basedatatypes import BaseFigure

from value_dashboard.ai.data_context import DataChatDataset
from value_dashboard.ai.responses import AgentResponse, DataFrameResponse, PlotlyResponse, TextResponse


class CodeValidationError(ValueError):
    """Raised when generated code does not meet the local execution policy."""


class CodeExecutionError(RuntimeError):
    """Raised when generated code fails during execution."""


@dataclass
class CodeExecutionResult:
    response: AgentResponse
    code: str


class GeneratedCodeValidator:
    """AST guardrails for generated analysis code.

    This is not a security sandbox. It is a local policy layer that keeps generated
    analysis code focused on dataframe work and Plotly chart construction.
    """

    ALLOWED_IMPORT_ROOTS = {
        "math",
        "numpy",
        "pandas",
        "plotly",
        "polars",
        "statistics",
    }
    BLOCKED_NAMES = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "exit",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "vars",
    }
    BLOCKED_ATTRIBUTES = {
        "read_csv",
        "read_excel",
        "read_json",
        "read_parquet",
        "to_csv",
        "to_excel",
        "to_json",
        "to_parquet",
        "to_pickle",
        "write_html",
        "write_image",
        "write_json",
    }

    def validate(self, code: str) -> None:
        tree = ast.parse(code)
        has_result_assignment = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._validate_import(node)
            elif isinstance(node, ast.Name) and node.id in self.BLOCKED_NAMES:
                raise CodeValidationError(f"Use of `{node.id}` is not allowed.")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in self.BLOCKED_ATTRIBUTES:
                    raise CodeValidationError(f"Use of attribute `{node.attr}` is not allowed.")
            elif isinstance(node, ast.Assign):
                has_result_assignment = has_result_assignment or any(
                    isinstance(target, ast.Name) and target.id == "result"
                    for target in node.targets
                )
            elif isinstance(node, ast.AnnAssign):
                has_result_assignment = has_result_assignment or (
                        isinstance(node.target, ast.Name) and node.target.id == "result"
                )

        if not has_result_assignment:
            raise CodeValidationError("Generated code must assign a `result` variable.")

    def _validate_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
        else:
            module_names = [node.module or ""]

        for module_name in module_names:
            root = module_name.split(".", 1)[0]
            if root not in self.ALLOWED_IMPORT_ROOTS:
                raise CodeValidationError(f"Import `{module_name}` is not allowed.")


class GeneratedCodeExecutor:
    """Execute generated dataframe analysis code and normalize its result."""

    def __init__(self, datasets: list[DataChatDataset]):
        self._datasets = datasets
        self._validator = GeneratedCodeValidator()

    def execute(self, code: str) -> CodeExecutionResult:
        self._validator.validate(code)
        env = self._build_environment()
        try:
            exec(code, env, env)
        except Exception as exc:
            error = traceback.format_exc()
            raise CodeExecutionError(error) from exc

        if "result" not in env:
            raise CodeExecutionError("Generated code did not assign `result`.")

        response = self._parse_result(env["result"])
        response.code = code
        return CodeExecutionResult(response=response, code=code)

    def _build_environment(self) -> dict[str, Any]:
        dataset_map = {
            dataset.name: dataset.dataframe.copy(deep=False)
            for dataset in self._datasets
        }

        def execute_sql_query(query: str) -> pd.DataFrame:
            connection = duckdb.connect(database=":memory:")
            try:
                for name, dataframe in dataset_map.items():
                    connection.register(name, dataframe)
                return connection.execute(query).df()
            finally:
                connection.close()

        safe_builtins = {
            "__import__": __import__,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        return {
            "__builtins__": safe_builtins,
            "datasets": dataset_map,
            "execute_sql_query": execute_sql_query,
            "go": go,
            "np": np,
            "pd": pd,
            "pio": pio,
            "pl": pl,
            "px": px,
        }

    def _parse_result(self, result: Any) -> AgentResponse:
        if not isinstance(result, dict):
            if isinstance(result, BaseFigure):
                return PlotlyResponse(value=result)
            if isinstance(result, pl.DataFrame):
                return DataFrameResponse(value=result.to_pandas())
            if isinstance(result, (pd.DataFrame, pd.Series, dict)):
                return DataFrameResponse(value=result)
            return TextResponse(value=str(result))

        result_type = str(result.get("type", "string")).lower()
        value = result.get("value")
        summary = str(result.get("summary", "") or "")

        if result_type in {"plot", "chart", "figure", "plotly"} or isinstance(value, BaseFigure):
            if not isinstance(value, BaseFigure):
                raise CodeExecutionError("Plotly result must provide a Plotly figure in `value`.")
            return PlotlyResponse(value=value, summary=summary)
        if result_type in {"data", "dataframe", "table"}:
            if isinstance(value, pl.DataFrame):
                value = value.to_pandas()
            if isinstance(value, (list, tuple)):
                value = pd.DataFrame(value)
            return DataFrameResponse(value=value, summary=summary)
        if result_type in {"number", "string", "text"}:
            return TextResponse(value=str(value), summary=summary)

        raise CodeExecutionError(f"Unsupported result type `{result_type}`.")
