"""Shared constants reused across UI, pipeline, and config editing flows."""

FILTER_OPERATORS = (
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "contains",
    "starts with",
    "in",
    "not in",
    "is null",
    "is not null",
)

IH_FILE_TYPES = ("parquet", "pega_ds_export")
CONFIG_FILE_TYPES = IH_FILE_TYPES + ("gzip",)

SCHEMA_PREVIEW_COLUMN = "Column"
SCHEMA_PREVIEW_MOST_OCCURRING = "Most occurring"
SCHEMA_PREVIEW_VALUES = "Values"
AI_SCHEMA_EXAMPLE_COLUMNS = (
    SCHEMA_PREVIEW_MOST_OCCURRING,
    SCHEMA_PREVIEW_VALUES,
)

IH_FOLDER_SESSION_KEY = "ihfolder"
HOLDINGS_FOLDER_SESSION_KEY = "holdingsfolder"
