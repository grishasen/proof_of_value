from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from plotly.basedatatypes import BaseFigure

ResponseType = Literal["string", "dataframe", "plotly"]


@dataclass
class AgentResponse:
    """Base response payload returned by the data chat agent."""
    value: Any
    response_type: ResponseType
    summary: str = ""
    code: str = ""


@dataclass
class TextResponse(AgentResponse):
    """Represent a plain-text response from the data chat agent."""
    value: str
    response_type: ResponseType = "string"


@dataclass
class DataFrameResponse(AgentResponse):
    """Represent a dataframe response from the data chat agent."""
    value: pd.DataFrame | pd.Series | dict
    response_type: ResponseType = "dataframe"


@dataclass
class PlotlyResponse(AgentResponse):
    """Represent a Plotly figure response from the data chat agent."""
    value: BaseFigure
    response_type: ResponseType = "plotly"
