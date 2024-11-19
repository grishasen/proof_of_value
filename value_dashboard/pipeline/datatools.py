import asyncio
import logging
import typing
from collections import OrderedDict
from typing import Any

import polars as pl
from polars import DataFrame

from value_dashboard.metrics.clv import clv
from value_dashboard.metrics.conversion import compact_conversion_data
from value_dashboard.metrics.conversion import conversion
from value_dashboard.metrics.descriptive import compact_descriptive_data
from value_dashboard.metrics.descriptive import descriptive
from value_dashboard.metrics.engagement import compact_engagement_data
from value_dashboard.metrics.engagement import engagement
from value_dashboard.metrics.experiment import compact_experiment_data
from value_dashboard.metrics.experiment import experiment
from value_dashboard.metrics.ml import compact_model_ml_scores_data
from value_dashboard.metrics.ml import model_ml_scores
from value_dashboard.reports.repdata import group_model_ml_scores_data, group_experiment_data, group_engagement_data, \
    group_conversion_data, group_descriptive_data
from value_dashboard.utils.config import get_config
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.timer import timed

logger = get_logger(__name__, logging.DEBUG)
data_cache_hours = 24
if 'data_cache_hours' in get_config()['ux'].keys():
    data_cache_hours = get_config()['ux']['data_cache_hours']
logger.debug(f"Data will be cached for {data_cache_hours} hours.")


async def run_data_collection_async(*coros_or_futures):
    result = await asyncio.gather(*coros_or_futures)
    return result


@timed
def collect_ih_metrics_data(loop, ih: pl.DataFrame | pl.LazyFrame,
                            mdata: typing.Dict[
                                str,
                                pl.LazyFrame |
                                pl.DataFrame |
                                pl.lazyframe.in_process.InProcessQuery
                            ],
                            streaming: bool,
                            background: bool,
                            config: dict):
    metrics = config["metrics"]
    coroutines = []
    for metric in metrics:
        params = metrics[metric]
        if metric.startswith("engagement"):
            coroutines.append(data_collection_async(ih, params, metric, streaming, background, engagement))
        if metric.startswith("model_ml_scores"):
            coroutines.append(data_collection_async(ih, params, metric, streaming, background, model_ml_scores))
        if metric.startswith("conversion"):
            coroutines.append(data_collection_async(ih, params, metric, streaming, background, conversion))
        if metric.startswith("descriptive"):
            coroutines.append(data_collection_async(ih, params, metric, streaming, background, descriptive))
        if metric.startswith("experiment"):
            coroutines.append(data_collection_async(ih, params, metric, streaming, background, experiment))
    process_metrics_coroutines(coroutines, loop, mdata, streaming)


@timed
def collect_clv_metrics_data(loop, holdings: pl.DataFrame | pl.LazyFrame,
                             mdata: typing.Dict[
                                 str,
                                 pl.LazyFrame |
                                 pl.DataFrame |
                                 pl.lazyframe.in_process.InProcessQuery
                             ],
                             streaming: bool,
                             background: bool,
                             config: dict):
    metrics = config["metrics"]
    coroutines = []
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
    result = loop.run_until_complete(run_data_collection_async(*coroutines))
    lazy_frames = OrderedDict()
    for metric, mdf in result:
        if isinstance(mdf, pl.lazyframe.in_process.InProcessQuery):
            df = mdf.fetch_blocking()
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
                mdata[metric] = pl.concat([mdf, mdata[metric]], how="diagonal", rechunk=True)
    if lazy_frames:
        frames = pl.collect_all(lazy_frames.values(), streaming=streaming)
        for metric in lazy_frames.keys():
            df = frames[list(lazy_frames.keys()).index(metric)]
            if not (metric in mdata):
                mdata[metric] = df
            else:
                mdata[metric] = pl.concat([df, mdata[metric]], how="diagonal", rechunk=True)


@timed
def collect_reports_data(collected_metrics_data: typing.Dict[str, pl.DataFrame]) -> dict[Any, tuple[DataFrame, Any]]:
    report_params = get_config()["reports"]
    reports_data: dict[Any, tuple[DataFrame, Any]] = {}
    for report in report_params:
        params = report_params[report]
        if params['metric'].startswith("engagement"):
            report_data = group_engagement_data(collected_metrics_data[params['metric']], params)
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("model_ml_scores"):
            report_data = group_model_ml_scores_data(collected_metrics_data[params['metric']], params)
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("conversion"):
            report_data = group_conversion_data(collected_metrics_data[params['metric']], params)
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("descriptive"):
            report_data = group_descriptive_data(collected_metrics_data[params['metric']], params)
            reports_data[report] = (report_data, params)
        elif params['metric'].startswith("experiment"):
            report_data = group_experiment_data(collected_metrics_data[params['metric']], params)
            reports_data[report] = (report_data, params)
    return reports_data


async def data_collection_async(ih: pl.DataFrame,
                                config: dict,
                                metric: str,
                                streaming: bool,
                                background: bool,
                                func: typing.Callable):
    return metric, func(ih, config, streaming, background)


@timed
def compact_data(data: pl.DataFrame, params: dict, metric: str) -> pl.DataFrame:
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
