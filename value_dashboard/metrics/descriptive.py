import traceback

import polars as pl
from polars import selectors as cs
from polars_ds import weighted_mean

from value_dashboard.utils.config import get_config
from value_dashboard.utils.polars_utils import build_digest, merge_digests
from value_dashboard.utils.py_utils import strtobool, stable_dedup
from value_dashboard.utils.timer import timed

NUM_DTYPES = tuple(pl.INTEGER_DTYPES) + tuple(pl.FLOAT_DTYPES)


def _numeric_intersection(schema: dict[str, pl.DataType], wanted: list[str]) -> list[str]:
    """
    Intersect a list of desired column names with numeric columns present in a schema.

    Filters `wanted` to keep only those names that exist in the schema and whose
    `pl.DataType` is considered numeric (as defined by `NUM_DTYPES`).

    Parameters
    ----------
    schema : dict[str, pl.DataType]
        A mapping from column name to Polars data type, typically obtained from
        `LazyFrame.collect_schema()` or `DataFrame.schema`.
    wanted : list[str]
        Candidate column names to be intersected with numeric columns in `schema`.

    Returns
    -------
    list[str]
        The subset of `wanted` that are present in `schema` and have numeric types
        according to `NUM_DTYPES`.

    Notes
    -----
    - `NUM_DTYPES` must be defined elsewhere (e.g., a tuple of numeric Polars types).
    - This helper is used to select the set of columns eligible for numeric aggregations
      in descriptive statistics.
    """
    return [c for c in wanted if (c in schema) and isinstance(schema[c], NUM_DTYPES)]


@timed
def descriptive(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    """
    Compute descriptive statistics over selected columns of Interaction History (IH).

    Aggregates descriptive metrics per group for a set of columns provided in
    `config["columns"]`. Supports two modes:
    - **t-digest mode** (`use_t_digest=True`): computes count/sum/mean/var/min/max and
      stores a t-digest per numeric column (as binary) for downstream percentile queries.
    - **explicit quantiles** (`use_t_digest=False`): computes additional statistics
      such as median, skewness, and selected quantiles (p25, p75, p90, p95).

    The function returns a **LazyFrame** unless `background=True`, in which case it
    collects and returns an eager **DataFrame**.

    Parameters
    ----------
    ih : pl.LazyFrame
        Source Interaction History, already filtered/curated upstream, containing the
        columns listed under `config["columns"]` and any grouping keys.
    config : dict
        Configuration dictionary with keys:
        - `group_by` : list[str]
            Base grouping keys. Will be merged with `get_config()["metrics"]["global_filters"]`
            and deduplicated via `stable_dedup(...)`.
        - `columns` : list[str]
            Column names to describe. Numeric aggregations are applied only to those that
            are both present and numeric in the schema.
        - `use_t_digest` : bool or str, optional (default True)
            Whether to use t-digest summarization instead of explicit quantiles.
        - `filter` : Any, optional
            If present and not a string, treated as a Polars predicate and applied to `ih`.
    streaming : bool, default False
        If `background=True`, selects `"streaming"` vs `"auto"` engine for `.collect(...)`.
    background : bool, default False
        If True, collects and returns an eager DataFrame; otherwise returns a LazyFrame.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        - LazyFrame when `background=False`.
        - DataFrame when `background=True` (collected with the specified engine).

    Notes
    -----
    - Grouping keys are computed as `stable_dedup(config['group_by'] + global_filters)`.
    - Numeric columns are determined using `_numeric_intersection(schema, columns)`,
      where `schema = ih.collect_schema()`.
    - In t-digest mode:
        * Each numeric column produces a binary field `{col}_tdigest` within a struct
          named `"TDigests"`, then un-nested into top-level columns.
        * The t-digest objects are intended for later merge/percentile calculation.
      In non t-digest mode:
        * Extra statistics include `Median`, `Skew`, and quantiles (`p25`, `p75`, `p90`, `p95`).
    - The code uses `pl.map_groups` to build per-group t-digests via `build_digest(...)`;
      ensure `build_digest` returns a binary representation compatible with your downstream
      `merge_digests(...)`.

    Raises
    ------
    Exception
        Any exceptions during aggregation/collection are logged (traceback) and re-raised.

    """
    mand_props_grp_by = stable_dedup(config['group_by'] + get_config()["metrics"]["global_filters"])
    columns = config['columns']
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else True

    schema = ih.collect_schema()
    num_columns = _numeric_intersection(schema, columns)

    if "filter" in config:
        filter_exp_cmp = config["filter"]
        if not isinstance(filter_exp_cmp, str):
            ih = ih.filter(config["filter"])

    common_aggs = [
        pl.col(columns).count().name.suffix('_Count'),
        pl.col(num_columns).sum().name.suffix('_Sum'),
        pl.col(num_columns).mean().name.suffix('_Mean'),
        pl.col(num_columns).var().name.suffix('_Var'),
        pl.col(num_columns).min().name.suffix('_Min'),
        pl.col(num_columns).max().name.suffix('_Max')
    ]
    if use_t_digest:
        tdigest_struct = pl.map_groups(
            exprs=num_columns,
            function=lambda df: {f"{value}_tdigest": build_digest([df[index]]) for index, value in
                                 enumerate(num_columns)},
            return_dtype=pl.Struct([pl.Field(f"{c}_tdigest", pl.Binary) for c in num_columns]),
            returns_scalar=True,
        ).alias("TDigests")
        agg_exprs = common_aggs + [tdigest_struct]
        ih_analysis = ih.group_by(mand_props_grp_by).agg(agg_exprs).unnest("TDigests")
    else:
        extra_aggs = [
            pl.col(num_columns).median().name.suffix('_Median'),
            pl.col(num_columns).skew().name.suffix('_Skew'),
            pl.col(num_columns).quantile(0.25).name.suffix('_p25'),
            pl.col(num_columns).quantile(0.75).name.suffix('_p75'),
            pl.col(num_columns).quantile(0.90).name.suffix('_p90'),
            pl.col(num_columns).quantile(0.95).name.suffix('_p95')
        ]
        agg_exprs = common_aggs + extra_aggs
        ih_analysis = ih.group_by(mand_props_grp_by).agg(agg_exprs)

    try:
        if background:
            return ih_analysis.collect(background=background, engine="streaming" if streaming else "auto")
        else:
            return ih_analysis
    except Exception as e:
        traceback.print_exc()
        raise e


@timed
def compact_descriptive_data(data: pl.DataFrame,
                             config: dict) -> pl.DataFrame:
    """
    Compact and reconcile descriptive statistics across groups.

    Performs second-level aggregation to consolidate pre-aggregated descriptive
    statistics to a higher grouping level. Supports both:
    - **non t-digest mode**: weighted aggregation of Mean, Median, Skew, and selected
      quantiles (if they were computed upstream),
    - **t-digest mode**: merges per-group t-digests to produce a group-level digest.

    It also recomputes pooled variance using:
    `Var = (sum(n-1)*Var + sum(n*(Mean-GroupMean)^2)) / (N-1)`, per numeric column.

    Parameters
    ----------
    data : pl.DataFrame
        Eager DataFrame produced by `descriptive(...).collect(...)`, expected to include:
        - counts/means/vars/min/max with `{col}_Count`, `{col}_Mean`, `{col}_Var`, etc.
        - and optionally `{col}_Median`, `{col}_Skew`, `{col}_p25`, `{col}_p75`, `{col}_p90`, `{col}_p95`
          or `{col}_tdigest` when t-digest mode is used.
    config : dict
        Configuration with keys:
        - `group_by` : list[str]
            Grouping keys for the compacted aggregation (merged with global filters).
        - `columns` : list[str]
            Original columns of interest; used to infer which numeric summaries exist.
        - `scores` : list[str]
            Which additional scores to aggregate when not using t-digests; may include
            any of: `["p25", "p75", "p90", "p95"].
        - `use_t_digest` : bool or str, optional (default True)
            If True, merges t-digests for numeric columns using `merge_digests(...)`.

    Returns
    -------
    pl.DataFrame
        A compacted DataFrame aggregated at the deduplicated grouping level with:
        - summed totals for `*_Count`, `*_Sum`, and min/max aggregation of `*_Min`, `*_Max`,
        - weighted means (and optionally medians/skews/quantiles),
        - merged t-digests when enabled,
        - pooled variance recomputed from component groups.

    Notes
    -----
    - Grouping keys are deduplicated as `stable_dedup(config['group_by'] + global_filters)`.
    - Weighted mean uses helper `weighted_mean(x, w)`.
    - Pooled variance uses temporary columns:
      `{c}_n_minus1_variance` = `(Count - 1) * Var`, and
      `{c}_n_mean_diff_sq` = `Count * (Mean - GroupMean)^2`,
      later combined and normalized by `(Count - 1)`.
    - In t-digest mode, `{c}_tdigest` is merged by grouping via `merge_digests(...)`
      and returned as `{c}_tdigest` (binary).

    """
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else True
    columns_conf = config['columns']
    scores = config['scores']
    num_columns = [col for col in columns_conf if (col + '_Mean') in data.columns]
    grp_by = stable_dedup(config['group_by'] + get_config()["metrics"]["global_filters"])

    grouped_mean = (
        data
        .group_by(grp_by)
        .agg([pl.col(grp_by).first().name.suffix("_a")]
             +
             [
                 weighted_mean(pl.col(f'{c}_Mean'), pl.col(f'{c}_Count')).alias(f'{c}_GroupMean_a') for c in
                 num_columns
             ]
             )
        .select(cs.ends_with("_a"))
        .rename(lambda column_name: column_name.removesuffix('_a'))
    )

    copy_data = data.join(grouped_mean, on=grp_by)

    copy_data = copy_data.with_columns(
        [((pl.col(f'{c}_Count') - 1) * pl.col(f'{c}_Var')).alias(f'{c}_n_minus1_variance')
         for c in num_columns] +
        [(pl.col(f'{c}_Count') * (pl.col(f'{c}_Mean') - pl.col(f'{c}_GroupMean')) ** 2)
        .alias(f'{c}_n_mean_diff_sq')
         for c in num_columns]
    )

    common_aggs = [
        (cs.ends_with('Count').sum()).name.suffix("_a"),
        (cs.ends_with('Sum').sum()).name.suffix("_a"),
        (cs.ends_with('Min').min()).name.suffix("_a"),
        (cs.ends_with('Max').max()).name.suffix("_a"),
        pl.col(grp_by).first().name.suffix("_a")
    ]

    mean_aggs = [
        weighted_mean(pl.col(f'{c}_Mean'), pl.col(f'{c}_Count')).alias(f'{c}_Mean_a')
        for c in num_columns
    ]

    if not use_t_digest:
        extra_aggs = (
                [weighted_mean(pl.col(f'{c}_Median'), pl.col(f'{c}_Count')).alias(f'{c}_Median_a')
                 for c in num_columns] +
                [weighted_mean(pl.col(f'{c}_Skew'), pl.col(f'{c}_Count')).alias(f'{c}_Skew_a')
                 for c in num_columns]
        )
        if 'p25' in scores:
            extra_aggs += [weighted_mean(pl.col(f'{c}_p25'), pl.col(f'{c}_Count')).alias(f'{c}_p25_a')
                           for c in num_columns]
        if 'p75' in scores:
            extra_aggs += [weighted_mean(pl.col(f'{c}_p75'), pl.col(f'{c}_Count')).alias(f'{c}_p75_a')
                           for c in num_columns]
        if 'p95' in scores:
            extra_aggs += [weighted_mean(pl.col(f'{c}_p95'), pl.col(f'{c}_Count')).alias(f'{c}_p95_a')
                           for c in num_columns]
        if 'p90' in scores:
            extra_aggs += [weighted_mean(pl.col(f'{c}_p90'), pl.col(f'{c}_Count')).alias(f'{c}_p90_a')
                           for c in num_columns]
    else:
        extra_aggs = [
            pl.map_groups(
                exprs=[pl.col(f'{c}_tdigest')],
                function=lambda s: merge_digests(s),
                return_dtype=pl.Binary,
                returns_scalar=True
            ).alias(f'{c}_tdigest_a') for c in num_columns
        ]

    tail_aggs = (
            [pl.col(f'{c}_n_minus1_variance').sum().alias(f'{c}_sum_n_minus1_variance_tmp_a')
             for c in num_columns] +
            [pl.col(f'{c}_n_mean_diff_sq').sum().alias(f'{c}_sum_n_mean_diff_sq_tmp_a')
             for c in num_columns]
    )

    agg_list = common_aggs + mean_aggs + extra_aggs + tail_aggs
    result = (
        copy_data
        .group_by(grp_by)
        .agg(agg_list)
        .select(cs.ends_with("_a"))
        .rename(lambda col: col.removesuffix('_a'))
        .with_columns([
            ((pl.col(f'{c}_sum_n_minus1_variance_tmp') + pl.col(f'{c}_sum_n_mean_diff_sq_tmp'))
             / (pl.col(f'{c}_Count') - 1)).alias(f'{c}_Var')
            for c in num_columns
        ])
        .select(~cs.ends_with("_tmp"))
    )

    return result
