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
    return [c for c in wanted if (c in schema) and isinstance(schema[c], NUM_DTYPES)]


@timed
def descriptive(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
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
