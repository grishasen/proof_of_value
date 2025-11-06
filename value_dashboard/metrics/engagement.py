import traceback

import polars as pl

from value_dashboard.metrics.constants import MODELCONTROLGROUP, INTERACTION_ID, RANK, OUTCOME, ACTION_ID
from value_dashboard.utils.config import get_config
from value_dashboard.utils.py_utils import stable_dedup
from value_dashboard.utils.timer import timed


@timed
def engagement(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    """
    Compute engagement metrics from Interaction History (IH).

    Filters IH to rows whose `OUTCOME` is in the union of negative and positive model
    responses, derives a binary outcome flag, deduplicates per interaction key by
    retaining the row with the maximum `Outcome_Binary`, and aggregates counts by the
    configured grouping keys plus `MODELCONTROLGROUP`.

    The resulting metrics per group include:
      - `Count` : total qualifying interaction rows in the group,
      - `Positives` : count of rows with positive outcome (1),
      - `Negatives` : `Count - Positives`.

    Parameters
    ----------
    ih : pl.LazyFrame
        Interaction History lazy frame containing, at a minimum, columns:
        `OUTCOME`, `INTERACTION_ID`, `ACTION_ID`, `RANK`, and the fields used in
        `config['group_by']`, global metric filters, and `MODELCONTROLGROUP`.
    config : dict
        Configuration dictionary with keys:
        - `group_by` : list[str]
            Base grouping columns; they are merged with global metric filters from
            `get_config()["metrics"]["global_filters"]` and with `MODELCONTROLGROUP`,
            then deduplicated via `stable_dedup(...)`.
        - `positive_model_response` : list[Any]
            Values of `OUTCOME` considered positive.
        - `negative_model_response` : list[Any]
            Values of `OUTCOME` considered negative.
        - `filter` : Any, optional
            If present and not a string, treated as a Polars predicate and applied to `ih`.
    streaming : bool, default False
        If `background=True`, controls the engine for collection (`"streaming"` vs `"auto"`).
    background : bool, default False
        If True, the aggregated result is collected and returned as an eager DataFrame;
        otherwise a LazyFrame pipeline is returned.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        - LazyFrame when `background=False`.
        - DataFrame when `background=True` (collected using the specified engine).

    Notes
    -----
    - Rows are restricted to outcomes in `negative_model_response + positive_model_response`.
    - `Outcome_Binary` is 1 for positives and 0 otherwise, and the row with the maximum
      `Outcome_Binary` per `(INTERACTION_ID, ACTION_ID, RANK)` is retained to avoid
      duplicate outcome contributions within a single interaction key.
    - Grouping keys are the deduplicated union of `config['group_by']`,
      `get_config()["metrics"]["global_filters"]`, and `MODELCONTROLGROUP`.

    Raises
    ------
    Exception
        Any exception from Polars operations is logged (traceback) and re-raised.

    """
    mand_props_grp_by = stable_dedup(
        config['group_by'] + get_config()["metrics"]["global_filters"] + [MODELCONTROLGROUP])
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']

    if "filter" in config:
        filter_exp_cmp = config["filter"]
        if not isinstance(filter_exp_cmp, str):
            ih = ih.filter(config["filter"])

    try:
        ih_analysis = (
            ih.filter(
                (pl.col(OUTCOME).is_in(negative_model_response + positive_model_response))
            )
            .with_columns([
                pl.when(pl.col(OUTCOME).is_in(positive_model_response)).
                then(1).otherwise(0).alias('Outcome_Binary')
            ])
            .filter(pl.col('Outcome_Binary') == pl.col('Outcome_Binary').max().over(INTERACTION_ID, ACTION_ID, RANK))
            .group_by(mand_props_grp_by)
            .agg([
                pl.len().alias('Count'),
                pl.sum("Outcome_Binary").alias("Positives")
            ])
            .with_columns([
                (pl.col("Count") - (pl.col("Positives"))).alias("Negatives")
            ])
        )
        if background:
            return ih_analysis.collect(background=background, engine="streaming" if streaming else "auto")
        else:
            return ih_analysis
    except Exception as e:
        traceback.print_exc()
        raise e


@timed
def compact_engagement_data(eng_data: pl.DataFrame,
                            config: dict) -> pl.DataFrame:
    """
    Compact aggregated engagement data by regrouping and summing KPIs.

    Re-aggregates engagement metrics across the deduplicated grouping keys
    (`group_by` + global metric filters + `MODELCONTROLGROUP`) by summing
    `Negatives`, `Positives`, and `Count`.

    Parameters
    ----------
    eng_data : pl.DataFrame
        Eager DataFrame produced by `engagement(...).collect(...)`, expected to contain
        the grouping columns and the metrics: `Negatives`, `Positives`, `Count`.
    config : dict
        Configuration with:
        - `group_by` : list[str]
            Base grouping columns. Will be combined with global metric filters from
            `get_config()["metrics"]["global_filters"]` and `MODELCONTROLGROUP`, then
            deduplicated using `stable_dedup(...)`.

    Returns
    -------
    pl.DataFrame
        Compacted DataFrame with one row per deduplicated group and summed metrics:
        `Negatives`, `Positives`, `Count`.

    Notes
    -----
    - Ensures consistency with the upstream grouping logic used in `engagement(...)`.
    - This helper is typically used to reduce cardinality for reporting.

    """
    grp_by = stable_dedup(config['group_by'] + get_config()["metrics"]["global_filters"] + [MODELCONTROLGROUP])
    data_copy = (
        eng_data
        .group_by(grp_by)
        .agg(pl.sum("Negatives").alias("Negatives"),
             pl.sum("Positives").alias("Positives"),
             pl.sum("Count").alias("Count"))
    )

    return data_copy
