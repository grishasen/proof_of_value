import traceback

import polars as pl

from value_dashboard.metrics.constants import INTERACTION_ID, OUTCOME, RANK, ACTION_ID
from value_dashboard.utils.config import get_config
from value_dashboard.utils.py_utils import stable_dedup
from value_dashboard.utils.timer import timed


@timed
def experiment(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    """
    Compute experiment KPIs (per variant) from Interaction History (IH).

    Filters IH to rows whose `OUTCOME` is in the union of negative and positive model
    responses, creates a binary outcome flag, removes duplicate outcome contributions
    within a single interaction key, and aggregates by experiment grouping keys plus
    any configured global metric filters.

    Per-group metrics include:
      - `Count` : total qualifying rows in the group,
      - `Positives` : sum of binary outcome = 1,
      - `Negatives` : `Count - Positives`.

    Parameters
    ----------
    ih : pl.LazyFrame
        Interaction History lazy frame, expected to include at least:
        `OUTCOME`, `INTERACTION_ID`, `ACTION_ID`, `RANK`, and columns referenced by
        `config['group_by']`, `config['experiment_name']`, `config['experiment_group']`.
    config : dict
        Configuration dictionary with keys:
        - `group_by` : list[str]
            Base grouping columns.
        - `experiment_name` : str
            Column name that identifies the experiment name/key.
        - `experiment_group` : str
            Column name identifying the experiment variant (e.g., control/test).
        - `positive_model_response` : list[Any]
            Values of `OUTCOME` considered positive.
        - `negative_model_response` : list[Any]
            Values of `OUTCOME` considered negative.
        - `filter` : Any, optional
            If present and not a string, treated as a Polars predicate and applied to `ih`.
    streaming : bool, default False
        If `background=True`, controls collection engine (`"streaming"` vs `"auto"`).
    background : bool, default False
        If True, collects and returns an eager DataFrame; otherwise returns a LazyFrame.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        - LazyFrame when `background=False`.
        - DataFrame when `background=True`.

    Notes
    -----
    - Grouping keys are the stable-deduplicated union of:
      `get_config()["metrics"]["global_filters"] + config['group_by'] + [experiment_name] + [experiment_group]`.
    - Binary outcome:
      `Outcome_Binary = 1` iff `OUTCOME` in `positive_model_response`, else `0`.
    - De-duplication:
      Keeps the row with the maximum `Outcome_Binary` per `(INTERACTION_ID, ACTION_ID, RANK)`
      to avoid double-counting outcomes within a single interaction key.
    - Filters out groups with `Count <= 0` before computing `Negatives`.

    Raises
    ------
    Exception
        Any exception from Polars operations; traceback is printed and re-raised.

    """
    mand_props_grp_by = stable_dedup(get_config()["metrics"]["global_filters"] +
                                     config['group_by'] + [config['experiment_name']] + [config['experiment_group']])
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
            .filter(pl.col("Count") > 0)
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
def compact_experiment_data(exp_data: pl.DataFrame,
                            config: dict) -> pl.DataFrame:
    """
    Compact experiment aggregates at a higher grouping level.

    Re-aggregates per-variant experiment KPIs (`Count`, `Positives`, `Negatives`) across
    the deduplicated grouping keys that combine base `group_by`, the experiment name,
    experiment group, and global filters. Removes any groups where either `Positives`
    or `Negatives` are zero to ensure meaningful lift calculations downstream.

    Parameters
    ----------
    exp_data : pl.DataFrame
        Eager DataFrame produced by `experiment(...).collect(...)`, expected to include:
        `Count`, `Positives`, `Negatives`, and the grouping columns.
    config : dict
        Configuration with keys:
        - `group_by` : list[str]
            Base grouping keys to retain.
        - `experiment_name` : str
            Column with experiment name/key.
        - `experiment_group` : str
            Column with experiment variant.
        (Global metric filters are read via `get_config()["metrics"]["global_filters"]`.)

    Returns
    -------
    pl.DataFrame
        Compacted DataFrame grouped by the deduplicated grouping keys with aggregated
        sums of `Count`, `Positives`, and `Negatives`. Rows with zero `Positives` or
        zero `Negatives` are filtered out.

    Notes
    -----
    - Grouping keys are combined as:
      `stable_dedup(config['group_by'] + [experiment_name] + [experiment_group] + global_filters)`.
    - This step is typically used prior to computation of experiment metrics such as
      rates, lift, and significance.

    """
    grp_by = stable_dedup(config['group_by'] + [config['experiment_name']] +
                          [config['experiment_group']] + get_config()["metrics"]["global_filters"])
    if grp_by:
        exp_data = (
            exp_data
            .group_by(grp_by)
            .agg([
                pl.col("Count").sum(),
                pl.col("Positives").sum(),
                pl.col("Negatives").sum(),
            ])
            .filter(
                (pl.col("Positives") > 0) & (pl.col("Negatives") > 0)
            )
        )

    return exp_data
