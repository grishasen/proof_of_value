from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from plotly.basedatatypes import BaseFigure

ResponseType = Literal["string", "dataframe", "plotly"]


@dataclass
class AgentResponse:
    value: Any
    response_type: ResponseType
    summary: str = ""
    code: str = ""


@dataclass
class TextResponse(AgentResponse):
    value: str
    response_type: ResponseType = "string"


@dataclass
class DataFrameResponse(AgentResponse):
    value: pd.DataFrame | pd.Series | dict
    response_type: ResponseType = "dataframe"


@dataclass
class PlotlyResponse(AgentResponse):
    value: BaseFigure
    response_type: ResponseType = "plotly"
