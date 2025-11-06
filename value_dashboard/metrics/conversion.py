import traceback

import polars as pl

from value_dashboard.metrics.constants import INTERACTION_ID, RANK, OUTCOME, CUSTOMER_ID, \
    CONVERSION_EVENT_ID, ACTION_ID
from value_dashboard.metrics.constants import REVENUE_PROP_NAME
from value_dashboard.utils.config import get_config
from value_dashboard.utils.py_utils import stable_dedup
from value_dashboard.utils.timer import timed


@timed
def conversion(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    """
    Compute conversion analytics from Interaction History (IH).

    Filters IH to rows whose `OUTCOME` is in the union of negative and positive model
    responses, calculates binary outcomes, performs simple attribution (counts of
    touchpoints per customer and conversion event), and aggregates KPIs by the configured
    grouping keys.

    The resulting metrics per group include:
      - `Count` : total qualifying interaction rows in the group,
      - `Revenue` : sum of `REVENUE_PROP_NAME`,
      - `Touchpoints` : sum of touchpoints attributed to positives,
      - `Positives` : count of rows marked with positive outcome (1),
      - `Negatives` : `Count - Positives`.

    Parameters
    ----------
    ih : pl.LazyFrame
        Interaction History lazy frame containing, at a minimum, columns:
        `OUTCOME`, `INTERACTION_ID`, `ACTION_ID`, `RANK`, `CUSTOMER_ID`,
        `CONVERSION_EVENT_ID`, and `REVENUE_PROP_NAME` (or equivalent names used by the codebase).
    config : dict
        Configuration dictionary with keys:
        - `group_by` : list[str]
            Base grouping columns; will be combined with global metric filters from
            `get_config()["metrics"]["global_filters"]` and deduplicated via `stable_dedup(...)`.
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
    - Rows are first restricted to outcomes in `negative_model_response + positive_model_response`.
    - `Outcome_Binary` is 1 for positives and 0 otherwise, and the row with the maximum
      `Outcome_Binary` per `(INTERACTION_ID, ACTION_ID, RANK)` is retained to avoid
      duplicate outcome contributions within a single interaction key.
    - Touchpoint attribution:
      touchpoints are counted per `(CUSTOMER_ID, CONVERSION_EVENT_ID)` using only **positive**
      outcomes, then left-joined back to the filtered dataset and summed per group.
    - Grouping keys are the deduplicated union of `config['group_by']` and global metric filters
      from `get_config()["metrics"]["global_filters"]`.

    Raises
    ------
    Exception
        Any exception from Polars operations is logged (traceback) and re-raised.

    """
    mand_props_grp_by = stable_dedup(config['group_by'] + get_config()["metrics"]["global_filters"])
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']

    if "filter" in config:
        filter_exp_cmp = config["filter"]
        if not isinstance(filter_exp_cmp, str):
            ih = ih.filter(config["filter"])

    try:
        ih_analysis = ih.filter(
            (pl.col(OUTCOME).is_in(negative_model_response + positive_model_response))
        )
        ih_attribution = (
            ih_analysis
            .filter(pl.col(OUTCOME).is_in(positive_model_response))
            .group_by([CUSTOMER_ID, CONVERSION_EVENT_ID])
            .agg([
                pl.len().alias('Touchpoints')
            ]))
        ih_analysis = (
            ih_analysis
            .with_columns([
                pl.when(pl.col(OUTCOME).is_in(positive_model_response)).then(1).otherwise(0).alias('Outcome_Binary')
            ])
            .filter(pl.col('Outcome_Binary') == pl.col('Outcome_Binary').max().over(INTERACTION_ID, ACTION_ID, RANK))
            .join(ih_attribution, on=[CUSTOMER_ID, CONVERSION_EVENT_ID], how='left')
            .group_by(mand_props_grp_by)
            .agg([
                pl.len().alias('Count'),
                pl.sum(REVENUE_PROP_NAME),
                pl.sum('Touchpoints'),
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
def compact_conversion_data(conv_data: pl.DataFrame,
                            config: dict) -> pl.DataFrame:
    """
    Compact aggregated conversion data by (optionally) regrouping and summing KPIs.

    Filters out groups with zero negatives (keeps only rows with `Negatives > 0`)
    and, if grouping keys are provided, re-aggregates totals across those keys.

    Parameters
    ----------
    conv_data : pl.DataFrame
        Eager DataFrame produced by `conversion(...)` after collection, expected to
        contain columns: `Negatives`, `Positives`, `Revenue`, `Count`, `Touchpoints`,
        plus any grouping columns used upstream.
    config : dict
        Configuration with:
        - `group_by` : list[str]
            Base grouping columns. Will be combined with global metric filters from
            `get_config()["metrics"]["global_filters"]` and deduplicated using `stable_dedup(...)`.

    Returns
    -------
    pl.DataFrame
        Compacted DataFrame. If grouping keys are non-empty, contains one row per
        deduplicated group with summed metrics:
        `Negatives`, `Positives`, `Revenue`, `Count`, `Touchpoints`.
        Otherwise, returns the filtered (Negatives > 0) input as-is.

    Notes
    -----
    - Uses the same deduped grouping keys strategy as `conversion(...)` to ensure
      consistency with upstream aggregation.
    - Designed as a post-processing step to reduce cardinality and remove groups
      without any negative cases.

    """
    data_copy = conv_data.filter(pl.col("Negatives") > 0)

    grp_by = stable_dedup(config['group_by'] + get_config()["metrics"]["global_filters"])
    if grp_by:
        data_copy = (
            data_copy
            .group_by(grp_by)
            .agg([
                pl.col("Negatives").sum(),
                pl.col("Positives").sum(),
                pl.col("Revenue").sum(),
                pl.col("Count").sum(),
                pl.col('Touchpoints').sum()
            ]
            )
        )

    return data_copy
