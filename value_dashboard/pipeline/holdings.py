import asyncio
import glob
import os
import re
import time
import typing
from collections import defaultdict
from datetime import timedelta

import numpy as np
import polars as pl
import streamlit as st
from polars import LazyFrame, DataFrame

from value_dashboard.pipeline.datatools import collect_clv_metrics_data
from value_dashboard.utils.config import get_config
from value_dashboard.utils.file_utils import read_dataset_export, detect_delimiter
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.py_utils import capitalize
from value_dashboard.utils.py_utils import strtobool
from value_dashboard.utils.timer import timed

HOLDINGS_FOLDER = "holdingsfolder"
logger = get_logger(__name__)
data_cache_hours = 24
if 'data_cache_hours' in get_config()['ux'].keys():
    data_cache_hours = get_config()['ux']['data_cache_hours']
logger.debug(f"Data will be cached for {data_cache_hours} hours.")
logger.debug(f"Numpy version {np.__version__}.")


def read_holdings_file_group(files: typing.Iterable,
                             filetype: str,
                             streaming: bool,
                             config: dict,
                             hive_partitioning: bool) -> LazyFrame | None:
    """
    Read and preprocess a group of product holdings files into a single Polars LazyFrame.

    This function scans (lazily) one or more input files—supporting Parquet, CSV, Excel,
    and Pega dataset export—and performs schema normalization, optional default value
    filling, global filtering, and calculated column additions defined in the config.
    It returns a concatenated LazyFrame (diagonal concat) that can be subsequently
    collected with streaming or auto engine.

    Parameters
    ----------
    files : Iterable
        An iterable of file paths or directories (when `hive_partitioning=True`) that
        belong to the same logical group (e.g., by date).
    filetype : str
        One of {'parquet', 'csv', 'xlsx', 'pega_ds_export'}. Determines the reader used.
    streaming : bool
        Whether to use Polars streaming engine when collecting the final result
        (`engine="streaming"` vs `"auto"`).
    config : dict
        Configuration dict with at least:
        - `config["holdings"]["extensions"]["default_values"]` : dict[str, Any], optional
          Default values for missing or null columns.
        - `config["holdings"]["extensions"]["filter"]` : str, optional
          A Python expression (string) evaluated to a Polars expression used to filter rows.
        - `config["holdings"]["extensions"]["columns"]` : str, optional
          A Python expression (string) evaluated to a sequence of Polars expressions
          to add/derive columns.
    hive_partitioning : bool
        If True, `pl.scan_parquet(..., hive_partitioning=True)` is used and elements in
        `files` are expected to be partition directories.

    Returns
    -------
    LazyFrame or None
        A Polars LazyFrame representing the concatenated and preprocessed holdings group,
        or `None` if the input list is empty.

    Raises
    ------
    Exception
        If `filetype` is not one of the supported types.
    Exception
        Any exception raised by underlying readers or malformed `eval` expressions.

    Notes
    -----
    - Column names are capitalized uniformly using `capitalize(...)` and then renamed.
    - When `default_values` are provided, the function adds missing columns with defaults
      and fills nulls in existing columns.
    - The `filter` and `columns` config entries are evaluated via `eval`. Ensure that
      the execution context is controlled and trusted.
    - Uses diagonal concatenation (`pl.concat(..., how="diagonal")`) to accommodate
      partially overlapping schemas.
    - Logging with `logger.debug` measures I/O and preprocessing times.

    See Also
    --------
    pl.scan_parquet : Lazy scan for Parquet.
    pl.scan_csv : Lazy scan for CSV.
    pl.read_excel : Eager Excel read; converted to lazy via `.lazy()`.

    """
    product_holdings_data_list = []
    start: float = time.time()
    for file in files:
        if filetype == 'parquet':
            product_holdings = pl.scan_parquet(file, cache=False, hive_partitioning=hive_partitioning,
                                               missing_columns='insert')
        elif filetype == 'pega_ds_export':
            product_holdings = read_dataset_export(file, lazy=True)
        elif filetype == 'csv':
            product_holdings = pl.scan_csv(
                source=file,
                separator=detect_delimiter(file),
                infer_schema_length=10000,
                try_parse_dates=True,
                cache=False
            )
        elif filetype == 'xlsx':
            product_holdings = pl.read_excel(source=file).lazy()
        else:
            raise Exception("File type not supported")

        logger.debug(f"Holdings data unpacking and load: {(time.time() - start) * 10 ** 3:.03f}ms")

        dframe_columns = product_holdings.collect_schema().names()
        cols = capitalize(dframe_columns)
        product_holdings = product_holdings.rename(dict(map(lambda i, j: (i, j), dframe_columns, cols)))
        product_holdings.collect_schema()

        if 'default_values' in config["holdings"]["extensions"].keys():
            default_values = config["holdings"]["extensions"]["default_values"]
            for new_col in default_values.keys():
                if new_col not in cols:
                    product_holdings = product_holdings.with_columns(pl.lit(default_values.get(new_col)).alias(new_col))
                else:
                    product_holdings = product_holdings.with_columns(
                        pl.col(new_col).fill_null(default_values.get(new_col)))

        global_ih_filter = config["holdings"]["extensions"]["filter"]
        if global_ih_filter:
            ih_filter_expr = eval(global_ih_filter)
            product_holdings = product_holdings.filter(ih_filter_expr)

        product_holdings_data_list.append(product_holdings)

    if not product_holdings_data_list:
        return
    product_holdings_group = pl.concat(product_holdings_data_list, how="diagonal")

    add_columns = config["holdings"]["extensions"]["columns"]
    if add_columns:
        add_columns_expr = eval(add_columns)
        product_holdings_group = product_holdings_group.with_columns(add_columns_expr)

    product_holdings_group = product_holdings_group.collect(engine="streaming" if streaming else "auto").lazy()
    logger.debug(f"Product holdings pre-processing took: {(time.time() - start) * 10 ** 3:.03f}ms")
    return product_holdings_group


@st.cache_data(show_spinner=False, ttl=timedelta(hours=data_cache_hours))
def load_holdings_data() -> typing.Dict[str, pl.DataFrame]:
    """
    Load, group, preprocess, and compute CLV metrics for product holdings.

    This function:
    1) Validates the configured holdings folder from Streamlit session state,
    2) Discovers files by pattern and groups them (e.g., by date) using a regex,
    3) Reads each group via `read_holdings_file_group(...)`,
    4) Computes CLV metrics data (possibly in background) and aggregates them into memory,
    5) Returns a dictionary of metric name -> Polars DataFrame.

    Results are cached by Streamlit (`@st.cache_data`) for `data_cache_hours`.

    Returns
    -------
    Dict[str, pl.DataFrame]
        A dictionary mapping metric keys (e.g., 'clv_totals', 'clv_segments', ...) to
        Polars eager DataFrames holding the computed metrics.

    Side Effects
    ------------
    - Uses Streamlit UI components: warnings, errors, `st.stop()`, progress bars, and metrics.
    - Logs diagnostic information (file type, streaming/background flags, timings, sizes).
    - Mutates `metrics` filter strings in config by replacing them with evaluated expressions.

    Notes
    -----
    - If no files are found for the configured `file_pattern`, it falls back to searching
      for `**/*.json` and switches `filetype='pega_ds_export'`.
    - File grouping depends on `config["holdings"]["file_group_pattern"]`. If it fails,
      the file basename is used as the group key.
    - `eval` is used to transform filter and column expressions; ensure trusted inputs.
    - `collect_clv_metrics_data(...)` is expected to fill `mdata` in place.
    - Uses an explicit asyncio loop for any async/background work.

    Raises
    ------
    StreamlitAPIException
        If the configured folder is missing or invalid (triggers `st.stop()` after showing UI feedback).
    Exception
        Any exception that arises during I/O, regex grouping, or metric collection.

    """
    load_start = time.time()
    if HOLDINGS_FOLDER not in st.session_state:
        st.warning("Please configure your product holdings files in the `Data import` tab.")
        st.stop()
    folder = st.session_state[HOLDINGS_FOLDER]
    if not os.path.isdir(folder):
        st.error(f"Folder {folder} not available anymore. Please update data import parameters.")
        st.stop()
    file_groups = defaultdict(set)
    config = get_config()
    filetype = config["holdings"]["file_type"]
    logger.debug("File type: " + filetype)
    streaming = False
    if "streaming" in config['holdings'].keys():
        streaming = strtobool(config['holdings']['streaming'])
    logger.debug("Use polars streaming dataframe collect: " + str(streaming))
    background = False
    if "background" in config['holdings'].keys():
        background = strtobool(config['holdings']['background'])
    logger.debug("Use polars background dataframe collect: " + str(background))
    hive_partitioning = False
    if "hive_partitioning" in config['holdings'].keys():
        hive_partitioning = strtobool(config['holdings']['hive_partitioning'])
    logger.debug("Use hive partitioning: " + str(hive_partitioning))

    metrics = config["metrics"]
    for metric in metrics:
        params = metrics[metric]
        if isinstance(params, dict):
            if "filter" in params.keys():
                filter_exp_cmp = params["filter"]
                if isinstance(filter_exp_cmp, str):
                    if filter_exp_cmp:
                        params["filter"] = eval(filter_exp_cmp)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    files = [file for file in glob.iglob(folder + config["holdings"]["file_pattern"], recursive=True)]
    if not files:
        files = [file for file in glob.iglob(folder + '**/*.json', recursive=True)]
        filetype = 'pega_ds_export'

    for file in files:
        try:
            filedate = re.findall(config["holdings"]["file_group_pattern"], os.path.abspath(file))[0]
            if hive_partitioning:
                file_groups[filedate].add(os.path.dirname(file))
            else:
                file_groups[filedate].add(os.path.abspath(file))
        except Exception as e:
            file_groups[os.path.basename(file)].add(os.path.abspath(file))

    file_groups = dict(sorted(file_groups.items(), reverse=True))
    size = len(file_groups.items())
    mdata = {}
    progress_bar = st.progress(0)
    container = st.empty()
    i = 0
    for key, files in file_groups.items():
        logger.debug(f"Processing product holdings group: {key}")
        start = time.time()
        i = i + 1
        progress_bar.progress(i / size, text=f"Processing  product holdings: {key}")
        holdings_group = read_holdings_file_group(files, filetype, streaming, config, hive_partitioning)

        if holdings_group is None:
            continue

        collect_clv_metrics_data(loop, holdings_group, mdata, streaming, background, config)
        end = time.time()
        logger.debug(f"Product holdings time taken: {(end - start) * 10 ** 3:.03f}ms")
        load_mid = time.time()
        hours, remainder = divmod(load_mid - load_start, 3600)
        minutes, seconds = divmod(remainder, 60)
        container.metric("Time", f"{hours:.0f}h:{minutes:.0f}m:{seconds:.02f}s", label_visibility="collapsed")

    progress_bar.empty()
    container.empty()
    collected_metrics_data = {}

    for metric in mdata:
        totals_frame = mdata[metric]
        totals_frame.shrink_to_fit(in_place=True)
        collected_metrics_data[metric] = totals_frame

    load_end = time.time()
    hours, remainder = divmod(load_end - load_start, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Load product holdings time: {hours:.0f}h:{minutes:.0f}m:{seconds:.02f}s")
    for metric in collected_metrics_data:
        frame = collected_metrics_data[metric]
        logger.debug(f"{metric} dataset size is {frame.estimated_size('kb')} kb.")
    return collected_metrics_data


@st.cache_data(show_spinner=False, ttl=timedelta(hours=data_cache_hours))
def get_reports_data():
    """
    Retrieve report datasets derived from CLV metrics.

    Wraps `load_holdings_data()` and transforms the metrics into the reports payload
    via `collect_clv_reports_data(...)`. The result is cached by Streamlit for
    `data_cache_hours`.

    Returns
    -------
    Any
        The return value of `collect_clv_reports_data(load_holdings_data())`, typically
        a dictionary mapping report names to tuples of (DataFrame, report params).
    """
    return collect_clv_reports_data(load_holdings_data())


@timed
def collect_clv_reports_data(collected_metrics_data: typing.Dict[str, pl.DataFrame]) -> dict[
    typing.Any, tuple[pl.DataFrame, typing.Any]]:
    """
    Prepare CLV-based report datasets from collected metrics.

    Filters report configurations to those pointing to a 'clv*' metric and returns
    a dictionary of report key -> (Polars DataFrame, report parameters).

    Parameters
    ----------
    collected_metrics_data : Dict[str, pl.DataFrame]
        A mapping of metric name to an eager Polars DataFrame as produced by
        `load_holdings_data()` / `collect_clv_metrics_data(...)`.

    Returns
    -------
    dict[Any, tuple[pl.DataFrame, Any]]
        Dictionary where each key is the report name, and each value is a tuple of
        (report dataframe, original report parameters).

    Notes
    -----
    - The function reads `get_config()["reports"]` to determine available reports
      and their associated metrics.
    - Only reports where `params['metric']` starts with `"clv"` are included.

    """
    report_params = get_config()["reports"]
    reports_data: dict[typing.Any, tuple[DataFrame, typing.Any]] = {}
    for report in report_params:
        params = report_params[report]
        if params['metric'].startswith("clv"):
            reports_data[report] = (collected_metrics_data[params['metric']], params)
    return reports_data
