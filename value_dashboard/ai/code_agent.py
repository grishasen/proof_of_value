from __future__ import annotations

import re

from value_dashboard.ai.code_executor import CodeExecutionError, GeneratedCodeExecutor
from value_dashboard.ai.data_context import DataChatDataset
from value_dashboard.ai.litellm_client import LiteLLMClient
from value_dashboard.ai.responses import AgentResponse


class DataCodeAgent:
    """LiteLLM-backed data chat agent that generates and executes Python code."""

    def __init__(
            self,
            datasets: list[DataChatDataset],
            llm: LiteLLMClient,
            description: str,
            memory_size: int = 10,
            max_retries: int = 2,
    ):
        self.datasets = datasets
        self.llm = llm
        self.description = description
        self.memory_size = memory_size
        self.max_retries = max_retries
        self.messages: list[dict[str, str]] = []
        self.last_generated_code = ""
        self.last_prompt = ""
        self._executor = GeneratedCodeExecutor(datasets)

    def start_new_conversation(self) -> None:
        self.messages = []
        self.last_generated_code = ""
        self.last_prompt = ""

    def set_history(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages[-self.memory_size:]

    def chat(self, query: str) -> AgentResponse:
        self.start_new_conversation()
        return self._process_query(query)

    def follow_up(self, query: str) -> AgentResponse:
        return self._process_query(query)

    def _process_query(self, query: str) -> AgentResponse:
        prompt = self._build_prompt(query)
        code = self._generate_code(prompt)

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                result = self._executor.execute(code)
                self.last_generated_code = result.code
                self.messages.append({"role": "user", "content": query})
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": result.response.summary or self._response_preview(result.response),
                    }
                )
                return result.response
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.max_retries:
                    raise
                code = self._repair_code(query=query, code=code, error=last_error)

        raise CodeExecutionError(last_error)

    def _generate_code(self, prompt: str) -> str:
        self.last_prompt = prompt
        response = self.llm.complete_text(
            prompt,
            system_prompt="You generate Python dataframe analysis code only.",
        )
        return self._extract_code(response)

    def _repair_code(self, query: str, code: str, error: str) -> str:
        repair_prompt = f"""
The previous generated code failed.

User question:
{query}

Failing code:
```python
{code}
```

Error:
{error}

Return a corrected complete Python code block only. Preserve the same result contract.
"""
        response = self.llm.complete_text(
            repair_prompt,
            system_prompt="You fix Python dataframe analysis code. Return code only.",
        )
        return self._extract_code(response)

    def _build_prompt(self, query: str) -> str:
        datasets = "\n".join(dataset.prompt_description() for dataset in self.datasets)
        history = self._history_block()
        return f"""
{self.description}

You are running inside the CDH Value Dashboard.

Available objects:
- `datasets`: dict[str, pandas.DataFrame], keyed by exact dataset name.
- `execute_sql_query(sql: str)`: executes DuckDB SQL over all datasets and returns a pandas DataFrame.
- Libraries already available: pandas as `pd`, polars as `pl`, numpy as `np`, plotly.express as `px`,
  plotly.graph_objects as `go`, plotly.io as `pio`.

Important execution rules:
1. Return Python code only. No markdown outside a code block, no explanation.
2. Do not read or write files. Do not use network calls. Do not use OS/process APIs.
3. Do not call Streamlit APIs. The app will render returned Plotly figures.
4. Use the existing dataframes in `datasets`; do not create synthetic data unless it is derived from them.
5. Prefer `execute_sql_query` for joins, filtering, grouping, aggregation, and sorting.
6. When calculating rates, aggregate numerator and denominator first, then divide.
7. For a chart, create a Plotly figure and return it as `result["value"]`.
8. Always assign `result` at the end.

Result contract:
```python
result = {{
    "type": "string" | "dataframe" | "plotly",
    "value": value,
    "summary": "short plain-English summary"
}}
```

Dataset catalog:
{datasets}

Recent conversation:
{history}

User question:
{query}

Generate complete Python code now.
"""

    def _history_block(self) -> str:
        if not self.messages:
            return "No previous conversation."
        lines = []
        for message in self.messages[-self.memory_size:]:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_code(response: str) -> str:
        fenced = re.search(
            r"```(?:python|py)?\s*(.*?)```",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        code = fenced.group(1) if fenced else response
        code = code.strip()
        if code.lower().startswith("python\n"):
            code = code[7:].strip()
        return code

    @staticmethod
    def _response_preview(response: AgentResponse) -> str:
        if response.response_type == "plotly":
            return "Generated a Plotly chart."
        if response.response_type == "dataframe":
            return "Generated a dataframe response."
        return str(response.value)
