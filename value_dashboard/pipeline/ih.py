import asyncio
import glob
import io
import json
import os
import re
import time
import typing
from collections import defaultdict
from datetime import timedelta

import numpy as np
import polars as pl
import psutil
import streamlit as st
from polars import LazyFrame

from value_dashboard.datalake.df_db_proxy import PolarsDuckDBProxy
from value_dashboard.metrics.constants import INTERACTION_ID, NAME, RANK, OUTCOME
from value_dashboard.metrics.conversion import conversion
from value_dashboard.metrics.descriptive import descriptive
from value_dashboard.metrics.engagement import engagement
from value_dashboard.metrics.experiment import experiment
from value_dashboard.metrics.ml import model_ml_scores
from value_dashboard.pipeline.datatools import collect_ih_metrics_data
from value_dashboard.pipeline.datatools import compact_data
from value_dashboard.utils.config import get_config
from value_dashboard.utils.db_utils import save_file_meta, get_file_meta, drop_all_tables
from value_dashboard.utils.file_utils import read_dataset_export
from value_dashboard.utils.logger import get_logger
from value_dashboard.utils.string_utils import strtobool, capitalize
from value_dashboard.utils.timer import timed

IHFOLDER = "ihfolder"
logger = get_logger(__name__)
data_cache_hours = 24
if 'data_cache_hours' in get_config()['ux'].keys():
    data_cache_hours = get_config()['ux']['data_cache_hours']
logger.debug(f"Data will be cached for {data_cache_hours} hours.")
logger.debug(f"Numpy version {np.__version__}.")
logger.debug(f"Polars {pl.build_info()}.")
logger.debug(f"Polars threads: {pl.thread_pool_size()}.")


@timed
@st.cache_data(show_spinner=False, ttl=timedelta(hours=data_cache_hours))
def load_data() -> typing.Dict[str, pl.DataFrame]:
    load_start = time.time()
    if 'use_aggregated' in st.session_state:
        use_aggregated = st.session_state['use_aggregated']
        if use_aggregated:
            f = open(st.session_state['aggregated_path'], "r")
            aggregated = json.load(f)
            collected_metrics_data = {}
            for metric in aggregated.keys():
                frame = json.dumps(aggregated[metric])
                collected_metrics_data[metric] = pl.DataFrame.deserialize(source=io.StringIO(frame), format='json')
            f.close()
            st.session_state['drop_cache'] = False
            return collected_metrics_data

    drop_cache = st.session_state['drop_cache'] if 'drop_cache' in st.session_state else False

    db_proxy = PolarsDuckDBProxy()

    if drop_cache:
        drop_all_tables(db_proxy)

    processed_files = get_file_meta(db_proxy)

    if IHFOLDER not in st.session_state:
        st.warning("Please configure your files in the `Data import` tab.")
        st.stop()
    folder = st.session_state[IHFOLDER]
    if not os.path.isdir(folder):
        st.error(f"Folder {folder} not available anymore. Please update data import parameters.")
        st.stop()
    file_groups = defaultdict(set)
    config = get_config()
    filetype = config["ih"]["file_type"]
    logger.debug("File type: " + filetype)
    streaming = False
    if "streaming" in config['ih'].keys():
        streaming = strtobool(config['ih']['streaming'])
    logger.debug("Use polars streaming dataframe collect: " + str(streaming))
    background = False
    if "background" in config['ih'].keys():
        background = strtobool(config['ih']['background'])
    logger.debug("Use polars background dataframe collect: " + str(background))
    hive_partitioning = False
    if "hive_partitioning" in config['ih'].keys():
        hive_partitioning = strtobool(config['ih']['hive_partitioning'])
    logger.debug("Use hive partitioning: " + str(hive_partitioning))

    add_columns = config["ih"]["extensions"]["columns"]
    global_ih_filter = config["ih"]["extensions"]["filter"]

    if global_ih_filter:
        ih_filter_expr = eval(global_ih_filter)
    else:
        ih_filter_expr = True

    if add_columns:
        add_columns_expr = eval(add_columns)
    else:
        add_columns_expr = []

    metrics = config["metrics"]
    for metric in metrics:
        params = metrics[metric]
        if isinstance(params, dict):
            if "filter" in params.keys():
                filter_exp_cmp = params["filter"]
                if isinstance(filter_exp_cmp, str):
                    if filter_exp_cmp:
                        params["filter"] = eval(filter_exp_cmp)

    metric_coroutines_map = {}
    for metric in metrics:
        if metric.startswith("engagement"):
            metric_coroutines_map[metric] = engagement
        if metric.startswith("model_ml_scores"):
            metric_coroutines_map[metric] = model_ml_scores
        if metric.startswith("conversion"):
            metric_coroutines_map[metric] = conversion
        if metric.startswith("descriptive"):
            metric_coroutines_map[metric] = descriptive
        if metric.startswith("experiment"):
            metric_coroutines_map[metric] = experiment

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    process = psutil.Process(os.getpid())

    # List all files in the folder matching the pattern
    files = [file for file in glob.iglob(folder + config["ih"]["file_pattern"], recursive=True)]
    if not files:
        files = [file for file in glob.iglob(folder + '**/*.json', recursive=True)]
        filetype = 'pega_ds_export'

    for file in files:
        if (processed_files["filename"] == file).any():
            logger.info("Skipping file : " + file)
            continue
        try:
            filedate = re.findall(config["ih"]["ih_group_pattern"], os.path.abspath(file))[0]
            if hive_partitioning:
                file_groups[filedate].add(os.path.dirname(file))
            else:
                file_groups[filedate].add(os.path.abspath(file))
        except Exception as e:
            file_groups[os.path.basename(file)].add(os.path.abspath(file))

    file_groups = dict(sorted(file_groups.items(), reverse=True))
    size = len(file_groups.items())

    mdata = {}
    for metric in metrics:
        if db_proxy.is_dataframe_exist(metric):
            mdata[metric] = db_proxy.get_dataframe(metric)

    progress_bar = st.progress(0)
    container = st.empty()
    i = 0
    for key, files_in_grp in file_groups.items():
        logger.debug(f"Processing group: {key}")
        start = time.time()
        i = i + 1
        progress_bar.progress(i / size, text=f"Processing: {key}")
        ih_group = read_file_group(
            files_in_grp,
            filetype,
            streaming,
            config,
            hive_partitioning,
            add_columns_expr,
            ih_filter_expr
        )
        if ih_group is None:
            continue
        collect_ih_metrics_data(loop, ih_group, mdata, streaming, background, config, metric_coroutines_map)
        del ih_group

        if (i > 31) & (i % 31 == 1):
            ram_mb = process.memory_info().rss / (1024 * 1024)
            logger.debug(f"RSS = {ram_mb:.2f} MB")
            logger.debug(f"SWAP = {psutil.swap_memory().used / (1024 * 1024):.2f} MB")
        end = time.time()
        logger.debug(f"Time taken: {(end - start) * 10 ** 3:.03f}ms")
        load_mid = time.time()
        hours, remainder = divmod(load_mid - load_start, 3600)
        minutes, seconds = divmod(remainder, 60)
        container.metric("Time", f"{hours:.0f}h:{minutes:.0f}m:{seconds:.02f}s", label_visibility="collapsed")

    progress_bar.empty()
    container.empty()
    collected_metrics_data = {}

    for metric in mdata:
        totals_frame = compact_data(mdata[metric], config['metrics'][metric], metric)
        totals_frame.shrink_to_fit(in_place=True)
        collected_metrics_data[metric] = totals_frame
        db_proxy.drop_dataframe(metric)
        db_proxy.store_dataframe(totals_frame, metric)

    for file in files:
        save_file_meta(db_proxy, file)

    load_end = time.time()
    hours, remainder = divmod(load_end - load_start, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Load time: {hours:.0f}h:{minutes:.0f}m:{seconds:.02f}s")
    for metric in collected_metrics_data:
        frame = collected_metrics_data[metric]
        logger.debug(f"{metric} dataset size is {frame.estimated_size('kb')} kb.")
    st.session_state['drop_cache'] = False
    return collected_metrics_data


def read_file_group(files: typing.Iterable,
                    filetype: str,
                    streaming: bool,
                    config: dict,
                    hive_partitioning: bool,
                    add_columns_expr: typing.Any,
                    ih_filter_expr: typing.Any
                    ) -> LazyFrame | None:
    ih_list = []
    start: float = time.time()
    for file in files:
        if filetype == 'parquet':
            ih = pl.scan_parquet(file, cache=False, hive_partitioning=hive_partitioning, allow_missing_columns=True)
        elif filetype == 'pega_ds_export':
            ih = read_dataset_export(file, lazy=True)
        else:
            raise Exception("File type not supported")

        logger.debug(f"Data unpacking and load: {(time.time() - start) * 10 ** 3:.03f}ms")

        dframe_columns = ih.collect_schema().names()
        capitalized = capitalize(dframe_columns)
        rename_map = dict(zip(dframe_columns, capitalized))
        ih = ih.rename(rename_map)

        with_cols_list = []
        if 'default_values' in config["ih"]["extensions"].keys():
            default_values = config["ih"]["extensions"]["default_values"]
            for new_col in default_values.keys():
                if new_col not in capitalized:
                    with_cols_list.append(pl.lit(default_values.get(new_col)).alias(new_col))
                else:
                    with_cols_list.append(pl.col(new_col).fill_null(default_values.get(new_col)))
        if with_cols_list:
            ih = ih.with_columns(with_cols_list)

        ih = ih.filter(ih_filter_expr)

        ih = (
            ih.with_columns([
                pl.col('OutcomeTime').str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z").alias('OutcomeDateTime'),
                pl.col('DecisionTime').str.strptime(pl.Datetime, "%Y%m%dT%H%M%S%.3f %Z").alias('DecisionDateTime')
            ])
            .with_columns([
                pl.col("OutcomeDateTime").dt.date().alias("Day"),
                pl.col("OutcomeDateTime").dt.strftime("%Y-%m").alias("Month"),
                pl.col("OutcomeDateTime").dt.year().cast(pl.Utf8).alias("Year"),
                (pl.col("OutcomeDateTime").dt.year().cast(pl.Utf8) + "_Q" +
                 pl.col("OutcomeDateTime").dt.quarter().cast(pl.Utf8)).alias("Quarter"),
                (pl.col("OutcomeDateTime") - pl.col("DecisionDateTime")).dt.total_seconds().alias("ResponseTime")
            ])
            .unique(subset=[INTERACTION_ID, NAME, RANK, OUTCOME])
            .drop([
                "FactID", "Label", "UpdateDateTime", "OutcomeTime", "DecisionTime",
                "OutcomeDateTime", "StreamPartition", "EvaluationCriteria", "Organization",
                "Unit", "Division", "Component", "ApplicationVersion", "Strategy"
            ], strict=False)
        )

        ih_list.append(ih)

    if not ih_list:
        return
    ih_group = pl.concat(ih_list, how="diagonal")
    if add_columns_expr:
        ih_group = ih_group.with_columns(add_columns_expr)

    ih_group = ih_group.collect(engine="streaming" if streaming else "auto").lazy()
    logger.debug(f"Pre-processing: {(time.time() - start) * 10 ** 3:.03f}ms")
    return ih_group
