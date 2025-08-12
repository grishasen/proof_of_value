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
