import traceback

import polars as pl
from polars import selectors as cs
from polars_ds import weighted_mean

from value_dashboard.utils.polars_utils import merge_tdigests
from value_dashboard.utils.polars_utils import tdigest
from value_dashboard.utils.string_utils import strtobool
from value_dashboard.utils.timer import timed


@timed
def descriptive(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    mand_props_grp_by = config['group_by']
    columns = config['columns']
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else False
    num_columns = [col for col in columns if col in ih.select(cs.numeric()).collect_schema().names()]
    if "filter" in config.keys():
        filter_exp = config["filter"]
        ih = ih.filter(filter_exp)

    try:
        ih_analysis = (
            ih
            .group_by(mand_props_grp_by)
            .agg(
                (
                    ([
                         pl.col(columns).count().name.suffix('_Count'),
                         (cs.numeric() & cs.by_name(columns, require_all=True)).sum().name.suffix('_Sum'),
                         (cs.numeric() & cs.by_name(columns, require_all=True)).mean().name.suffix('_Mean'),
                         (cs.numeric() & cs.by_name(columns, require_all=True)).var().name.suffix('_Var')
                     ]
                     +
                     [
                         pl.map_groups(
                             exprs=[f"{c}"],
                             function=tdigest,
                             return_dtype=pl.Struct,
                             returns_scalar=True).alias(f"{c}_tdigest") for c in num_columns
                     ]
                     ) if use_t_digest else []
                )
                +
                ([
                     pl.col(columns).count().name.suffix('_Count'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).sum().name.suffix('_Sum'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).mean().name.suffix('_Mean'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).var().name.suffix('_Var'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).median().name.suffix('_Median'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).skew().name.suffix('_Skew'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).quantile(0.25).name.suffix('_p25'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).quantile(0.75).name.suffix('_p75'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).quantile(0.90).name.suffix('_p90'),
                     (cs.numeric() & cs.by_name(columns, require_all=True)).quantile(0.95).name.suffix('_p95')
                 ] if not use_t_digest else [])
            )
        )
        if background:
            return ih_analysis.collect(background=background, streaming=streaming)
        else:
            return ih_analysis
    except Exception as e:
        traceback.print_exc()
        raise e


@timed
def compact_descriptive_data(data: pl.DataFrame,
                             config: dict) -> pl.DataFrame:
    use_t_digest = strtobool(config['use_t_digest']) if 'use_t_digest' in config.keys() else False
    columns_conf = config['columns']
    scores = config['scores']
    num_columns = [col for col in columns_conf if (col + '_Mean') in data.columns]
    grp_by = config['group_by']

    grouped_mean = (
        data
        .group_by(grp_by)
        .agg(
            [
                pl.col(grp_by).first().name.suffix("_a")
            ]
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
        [
            # (n_i - 1) * variance_i
            ((pl.col(f'{c}_Count') - 1) * pl.col(f'{c}_Var')).alias(f'{c}_n_minus1_variance') for c in num_columns
        ]
        +
        [
            # n_i * (mean_i - grp_mean)^2
            (pl.col(f'{c}_Count') * (pl.col(f'{c}_Mean') - pl.col(f'{c}_GroupMean')) ** 2).alias(
                f'{c}_n_mean_diff_sq')
            for c in num_columns
        ]
    )
    if not use_t_digest:
        copy_data = (
            copy_data
            .group_by(grp_by)
            .agg(
                [
                    (cs.ends_with('Count').sum()).name.suffix("_a"),
                    (cs.ends_with('Sum').sum()).name.suffix("_a"),
                    pl.col(grp_by).first().name.suffix("_a")
                ]
                +
                [
                    weighted_mean(pl.col(f'{c}_Mean'), pl.col(f'{c}_Count')).alias(f'{c}_Mean_a') for c in num_columns
                ]
                +
                [
                    weighted_mean(pl.col(f'{c}_Median'), pl.col(f'{c}_Count')).alias(f'{c}_Median_a') for c in
                    num_columns
                ]
                +
                [
                    weighted_mean(pl.col(f'{c}_Skew'), pl.col(f'{c}_Count')).alias(f'{c}_Skew_a') for c in num_columns
                ]
                +
                (
                    [
                        weighted_mean(pl.col(f'{c}_p25'), pl.col(f'{c}_Count')).alias(f'{c}_p25_a') for c in num_columns
                    ] if 'p25' in scores else []
                )
                +
                (
                    [
                        weighted_mean(pl.col(f'{c}_p75'), pl.col(f'{c}_Count')).alias(f'{c}_p75_a') for c in num_columns
                    ] if 'p75' in scores else []
                )
                +
                (
                    [
                        weighted_mean(pl.col(f'{c}_p95'), pl.col(f'{c}_Count')).alias(f'{c}_p95_a') for c in num_columns
                    ] if 'p95' in scores else []
                )
                +
                (
                    [
                        weighted_mean(pl.col(f'{c}_p90'), pl.col(f'{c}_Count')).alias(f'{c}_p90_a') for c in num_columns
                    ] if 'p90' in scores else []
                )
                +
                [
                    pl.col(f'{c}_n_minus1_variance').sum().alias(f'{c}_sum_n_minus1_variance_tmp_a') for c in
                    num_columns
                ]
                +
                [
                    pl.col(f'{c}_n_mean_diff_sq').sum().alias(f'{c}_sum_n_mean_diff_sq_tmp_a') for c in num_columns
                ]
            )
            .select(cs.ends_with("_a"))
            .rename(lambda column_name: column_name.removesuffix('_a'))
            .with_columns(
                [
                    ((pl.col(f'{c}_sum_n_minus1_variance_tmp') + pl.col(f'{c}_sum_n_mean_diff_sq_tmp')) / (
                            pl.col(f'{c}_Count') - 1))
                    .alias(f'{c}_Var') for c in num_columns
                ]
            )
            .select(~cs.ends_with("_tmp"))
        )
    else:
        copy_data = (
            copy_data
            .group_by(grp_by)
            .agg(
                [
                    (cs.ends_with('Count').sum()).name.suffix("_a"),
                    (cs.ends_with('Sum').sum()).name.suffix("_a"),
                    pl.col(grp_by).first().name.suffix("_a")
                ]
                +
                [
                    weighted_mean(pl.col(f'{c}_Mean'), pl.col(f'{c}_Count')).alias(f'{c}_Mean_a') for c in num_columns
                ]
                +
                [
                    pl.map_groups(
                        exprs=[f"{c}_tdigest"],
                        function=merge_tdigests,
                        return_dtype=pl.Struct,
                        returns_scalar=True
                    ).alias(f"{c}_tdigest_a") for c in num_columns
                ]
                +
                [
                    pl.col(f'{c}_n_minus1_variance').sum().alias(f'{c}_sum_n_minus1_variance_tmp_a') for c in
                    num_columns
                ]
                +
                [
                    pl.col(f'{c}_n_mean_diff_sq').sum().alias(f'{c}_sum_n_mean_diff_sq_tmp_a') for c in num_columns
                ]
            )
            .select(cs.ends_with("_a"))
            .rename(lambda column_name: column_name.removesuffix('_a'))
            .with_columns(
                [
                    ((pl.col(f'{c}_sum_n_minus1_variance_tmp') + pl.col(f'{c}_sum_n_mean_diff_sq_tmp')) / (
                            pl.col(f'{c}_Count') - 1))
                    .alias(f'{c}_Var') for c in num_columns
                ]
            )
            .select(~cs.ends_with("_tmp"))
        )

    return copy_data
