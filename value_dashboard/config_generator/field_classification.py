from typing import Any

import polars as pl

from value_dashboard.metrics.constants import REQ_IH_COLUMNS
from value_dashboard.utils.common_constants import SCHEMA_PREVIEW_COLUMN

FIELD_TAGS_COLUMN = "Field Tags"

BUSINESS_FIELD_HINTS = {
    "action",
    "channel",
    "category",
    "control",
    "customer",
    "group",
    "issue",
    "model",
    "name",
    "outcome",
    "placement",
    "product",
    "segment",
    "source",
    "status",
    "treatment",
    "type",
    "variant",
}
ID_FIELD_HINTS = ("id", "key", "guid", "uuid")
MEASURE_FIELD_HINTS = {
    "amount",
    "count",
    "cost",
    "duration",
    "finalpropensity",
    "lift",
    "priority",
    "propensity",
    "rate",
    "response",
    "revenue",
    "score",
    "value",
    "weight",
}
TECHNICAL_FIELD_HINTS = {
    "application",
    "component",
    "division",
    "fact",
    "internal",
    "operator",
    "organization",
    "partition",
    "stream",
    "unit",
    "version",
    "work",
}
TIME_FIELD_HINTS = ("date", "datetime", "day", "month", "quarter", "time", "timestamp", "year")
NUMERIC_TYPE_HINTS = ("int", "float", "decimal", "number")
TIME_TYPE_HINTS = ("date", "datetime", "time")


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_any(text: str, hints: set[str] | tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _is_numeric_dtype(data_type: str) -> bool:
    return _has_any(data_type.casefold(), NUMERIC_TYPE_HINTS)


def _is_time_dtype(data_type: str) -> bool:
    return _has_any(data_type.casefold(), TIME_TYPE_HINTS)


def _is_high_cardinality(unique_count: int | None, row_count: int | None) -> bool:
    if unique_count is None:
        return False
    if unique_count >= 50:
        return True
    if row_count and row_count > 0:
        return unique_count >= 20 and (unique_count / row_count) >= 0.2
    return False


def classify_field(
        field_name: str,
        *,
        data_type: str = "",
        unique_count: int | None = None,
        row_count: int | None = None,
        required_fields: set[str] | None = None,
) -> list[str]:
    field_lower = field_name.casefold()
    required = required_fields if required_fields is not None else set(REQ_IH_COLUMNS)
    tags = []

    is_required = field_name in required
    is_time = _has_any(field_lower, TIME_FIELD_HINTS) or _is_time_dtype(data_type)
    is_likely_id = (
            field_lower.endswith(ID_FIELD_HINTS)
            or _has_any(field_lower, (" id", "_id", "-id"))
            or field_lower in {"id", "subjectid", "customerid", "interactionid"}
    )
    is_technical = (
            field_lower.startswith(("px", "py"))
            or _has_any(field_lower, TECHNICAL_FIELD_HINTS)
    )
    has_measure_hint = _has_any(field_lower, MEASURE_FIELD_HINTS)
    is_numeric_measure = (
            _is_numeric_dtype(data_type)
            and not is_likely_id
            and not is_time
    ) or has_measure_hint
    is_high_cardinality = _is_high_cardinality(unique_count, row_count)
    is_business_dimension = (
            not is_technical
            and not is_likely_id
            and not is_numeric_measure
            and not is_time
            and (
                _has_any(field_lower, BUSINESS_FIELD_HINTS)
                or not is_high_cardinality
            )
    )

    if is_required:
        tags.append("Required")
    if is_time:
        tags.append("Time")
    if is_business_dimension:
        tags.append("Business dimension")
    if is_numeric_measure:
        tags.append("Numeric measure")
    if is_likely_id:
        tags.append("Likely ID")
    if is_technical:
        tags.append("Technical")
    if is_high_cardinality:
        tags.append("High-cardinality")
    if not tags:
        tags.append("Review")
    return tags


def add_field_classification(schema_preview: pl.DataFrame, row_count: int | None = None) -> pl.DataFrame:
    if schema_preview.is_empty() or SCHEMA_PREVIEW_COLUMN not in schema_preview.columns:
        return schema_preview

    records = []
    for row in schema_preview.to_dicts():
        tags = classify_field(
            str(row.get(SCHEMA_PREVIEW_COLUMN, "")),
            data_type=str(row.get("Data Type", "")),
            unique_count=_as_int(row.get("Unique Count")),
            row_count=row_count,
        )
        records.append({**row, FIELD_TAGS_COLUMN: ", ".join(tags)})
    return pl.from_dicts(records)
