import asyncio
import typing
from collections import OrderedDict

import polars as pl

from value_dashboard.metrics.clv import clv
from value_dashboard.metrics.conversion import compact_conversion_data
from value_dashboard.metrics.descriptive import compact_descriptive_data
from value_dashboard.metrics.engagement import compact_engagement_data
from value_dashboard.metrics.experiment import compact_experiment_data
from value_dashboard.metrics.ml import compact_model_ml_scores_data
from value_dashboard.utils.config import get_config
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.timer import timed

logger = get_logger(__name__)
data_cache_hours = 24
if 'data_cache_hours' in get_config()['ux'].keys():
    data_cache_hours = get_config()['ux']['data_cache_hours']
logger.debug(f"Data will be cached for {data_cache_hours} hours.")


async def run_data_collection_async(*coros_or_futures):
    """
    Run multiple asynchronous coroutines or futures concurrently and return their results.

    Args:
        *coros_or_futures: A variable number of coroutine or future objects.

    Returns:
        List of results from the completed coroutines or futures.
    """
    result = await asyncio.gather(*coros_or_futures)
    return result


@timed
def collect_ih_metrics_data(loop, ih: pl.DataFrame | pl.LazyFrame,
                            mdata: typing.Dict[
                                str, pl.LazyFrame | pl.DataFrame | pl.lazyframe.in_process.InProcessQuery],
                            streaming: bool,
                            background: bool,
                            config: dict,
                            metric_coroutines_map: dict):
    """
    Collects IH metrics data by preparing and executing metric-specific coroutines.

    Args:
        loop: The asyncio event loop.
        ih: Input data as a Polars DataFrame or LazyFrame.
        mdata: Dictionary to store collected metric data.
        streaming: Flag indicating if streaming execution is enabled.
        background: Flag indicating if background processing is enabled.
        config: Configuration dictionary containing metric parameters.
        metric_coroutines_map: Mapping of metric names to their corresponding coroutine functions.
    """
    metrics = config["metrics"]
    global_filters = metrics["global_filters"]
    coroutines = []

    # Prepare coroutines for each metric
    for metric in metric_coroutines_map.keys():
        params = metrics[metric]
        add_grp_by = list(set(global_filters) - set(params['group_by']))
        params['group_by'] += add_grp_by
        coroutines.append(
            data_collection_async(
                ih, params, metric, streaming, background, metric_coroutines_map.get(metric)
            )
        )

    # Execute and process the coroutines
    process_metrics_coroutines(coroutines, loop, mdata, streaming)


@timed
def collect_clv_metrics_data(loop, holdings: pl.DataFrame | pl.LazyFrame,
                             mdata: typing.Dict[
                                 str, pl.LazyFrame | pl.DataFrame | pl.lazyframe.in_process.InProcessQuery],
                             streaming: bool,
                             background: bool,
                             config: dict):
    """
    Collects CLV metrics data by executing relevant metric coroutines.

    Args:
        loop: The asyncio event loop.
        holdings: Holdings data as a Polars DataFrame or LazyFrame.
        mdata: Dictionary to store collected metric data.
        streaming: Flag indicating if streaming execution is enabled.
        background: Flag indicating if background processing is enabled.
        config: Configuration dictionary containing metric parameters.
    """
    metrics = config["metrics"]
    coroutines = []

    # Only collect metrics that start with "clv"
    for metric in metrics:
        params = metrics[metric]
        if metric.startswith("clv"):
            coroutines.append(data_collection_async(holdings, params, metric, streaming, background, clv))

    process_metrics_coroutines(coroutines, loop, mdata, streaming)


def process_metrics_coroutines(coroutines: typing.List, loop,
                               mdata: typing.Dict[
                                   str,
                                   pl.LazyFrame |
                                   pl.DataFrame |
                                   pl.lazyframe.in_process.InProcessQuery
                               ],
                               streaming: bool):
    """
    Executes metric coroutines and processes their results into the mdata dictionary.

    Args:
        coroutines: List of metric coroutines to execute.
        loop: The asyncio event loop.
        mdata: Dictionary to store processed metric data.
        streaming: Flag indicating if streaming execution is enabled.
    """
    result = loop.run_until_complete(run_data_collection_async(*coroutines))
    lazy_frames = OrderedDict()

    for metric, mdf in result:
        # Handle InProcessQuery by fetching and converting to DataFrame
        if isinstance(mdf, pl.lazyframe.in_process.InProcessQuery):
            df = mdf.fetch_blocking()
            schema = df.collect_schema()
            df = pl.from_arrow(df.to_arrow(), schema=schema)
            if not (metric in mdata):
                mdata[metric] = df
            else:
                mdata[metric] = pl.concat([df, mdata[metric]], how="diagonal", rechunk=True)
        elif isinstance(mdf, pl.LazyFrame):
            lazy_frames[metric] = mdf
        else:
            if not (metric in mdata):
                mdata[metric] = mdf
            else:
                schema = mdf.collect_schema()
                mdf = pl.from_arrow(mdf.to_arrow(), schema=schema)
                mdata[metric] = pl.concat([mdf, mdata[metric]], how="diagonal", rechunk=True)
    if lazy_frames:
        frames = pl.collect_all(lazy_frames.values(), engine="streaming" if streaming else "auto")
        for metric in lazy_frames.keys():
            df = frames[list(lazy_frames.keys()).index(metric)]
            schema = df.collect_schema()
            df = pl.from_arrow(df.to_arrow(), schema=schema)
            if not (metric in mdata):
                mdata[metric] = df
            else:
                mdata[metric] = pl.concat([df, mdata[metric]], how="diagonal", rechunk=True)


@timed
def collect_reports_data(collected_metrics_data: typing.Dict[str, pl.DataFrame]) -> dict[
    typing.Any, tuple[pl.DataFrame, typing.Any]]:
    """
    Collects report data based on predefined report configurations.

    Args:
        collected_metrics_data: Dictionary of collected metric data.

    Returns:
        Dictionary mapping report names to tuples of report data and parameters.
    """
    report_params = get_config()["reports"]
    reports_data: dict[typing.Any, tuple[pl.DataFrame, typing.Any]] = {}
    for report in report_params:
        params = report_params[report]
        if params['metric'].startswith("engagement"):
            report_data = collected_metrics_data[params['metric']].clone()
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("model_ml_scores"):
            report_data = collected_metrics_data[params['metric']].clone()
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("conversion"):
            report_data = collected_metrics_data[params['metric']].clone()
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("descriptive"):
            report_data = collected_metrics_data[params['metric']].clone()
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("experiment"):
            report_data = collected_metrics_data[params['metric']].clone()
            reports_data[report] = (report_data, params)
    return reports_data


@timed
def get_reports_data_by_name(report_name: str, collected_metrics_data: typing.Dict[str, pl.DataFrame]) -> tuple[
    pl.DataFrame, dict]:
    """
    Retrieves report data for a specific report name.

    Args:
        report_name: Name of the report to retrieve.
        collected_metrics_data: Dictionary of collected metric data.

    Returns:
        Tuple containing the report DataFrame and its configuration parameters.
    """
    report_params = get_config()["reports"]
    reports_data: tuple[pl.DataFrame, typing.Any] = ()
    params = report_params[report_name]
    if params['metric'].startswith("engagement"):
        report_data = collected_metrics_data[params['metric']].clone()
        reports_data = (report_data, params)
    elif params['metric'].startswith("model_ml_scores"):
        report_data = collected_metrics_data[params['metric']].clone()
        reports_data = (report_data, params)
    elif params['metric'].startswith("conversion"):
        report_data = collected_metrics_data[params['metric']].clone()
        reports_data = (report_data, params)
    elif params['metric'].startswith("descriptive"):
        report_data = collected_metrics_data[params['metric']].clone()
        reports_data = (report_data, params)
    elif params['metric'].startswith("experiment"):
        report_data = collected_metrics_data[params['metric']].clone()
        reports_data = (report_data, params)
    return reports_data


async def data_collection_async(ih: pl.DataFrame,
                                config: dict,
                                metric: str,
                                streaming: bool,
                                background: bool,
                                func: typing.Callable):
    """
    Executes a metric-specific data collection function asynchronously.

    Args:
        ih: Input data as a Polars DataFrame.
        config: Configuration dictionary for the metric.
        metric: Name of the metric.
        streaming: Flag indicating if streaming execution is enabled.
        background: Flag indicating if background processing is enabled.
        func: Callable function to execute for data collection.

    Returns:
        Tuple of metric name and the result of the data collection function.
    """
    return metric, func(ih, config, streaming, background)


@timed
def compact_data(data: pl.DataFrame, params: dict, metric: str) -> pl.DataFrame:
    """
    Compacts data based on the metric type using appropriate transformation logic.

    Args:
        data: Input Polars DataFrame to compact.
        params: Parameters for the compaction logic.
        metric: Name of the metric to determine compaction strategy.

    Returns:
        Compacted Polars DataFrame.
    """
    if metric.startswith("engagement"):
        data = compact_engagement_data(data, params)
    elif metric.startswith("model_ml_scores"):
        data = compact_model_ml_scores_data(data, params)
    elif metric.startswith("conversion"):
        data = compact_conversion_data(data, params)
    elif metric.startswith("descriptive"):
        data = compact_descriptive_data(data, params)
    elif metric.startswith("experiment"):
        data = compact_experiment_data(data, params)
    return data
