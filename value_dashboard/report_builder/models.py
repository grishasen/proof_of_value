from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportBuilderState:
    name: str = ""
    description: str = ""
    metric: str = ""
    type: str = ""
    chart_key: str = ""
    group_by: List[str] = field(default_factory=list)
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    facet_row: Optional[str] = None
    facet_column: Optional[str] = None
    value: Optional[str] = None
    size: Optional[str] = None
    property: Optional[str] = None
    score: Optional[str] = None
    stages: List[str] = field(default_factory=list)
    reference: Dict[str, Any] = field(default_factory=dict)
    r: Optional[str] = None
    theta: Optional[str] = None
    animation_frame: Optional[str] = None
    animation_group: Optional[str] = None
    log_x: bool = False
    log_y: bool = False
    showlegend: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)
    mode: str = "visual"
    reason: str = ""
