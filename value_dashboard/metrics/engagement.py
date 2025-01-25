import traceback

import polars as pl

from value_dashboard.metrics.constants import MODELCONTROLGROUP, INTERACTION_ID, NAME, RANK, OUTCOME
from value_dashboard.utils.string_utils import strtobool
from value_dashboard.utils.timer import timed


@timed
def engagement(ih: pl.LazyFrame, config: dict, streaming=False, background=False):
    mand_props_grp_by = config['group_by'] + [MODELCONTROLGROUP]
    negative_model_response = config['negative_model_response']
    positive_model_response = config['positive_model_response']
    negative_model_response_both_classes = strtobool(config[
                                                         'negative_model_response_both_classes']) if 'negative_model_response_both_classes' in config.keys() else False

    if "filter" in config.keys():
        filter_exp = config["filter"]
        if filter_exp:
            ih_filter_expr = eval(filter_exp)
            ih = ih.filter(ih_filter_expr)
    try:
        ih_analysis = (
            ih.filter(
                (pl.col(OUTCOME).is_in(negative_model_response + positive_model_response))
            )
            .with_columns([
                pl.when(pl.col(OUTCOME).is_in(positive_model_response)).
                then(1).otherwise(0).alias('Outcome_Binary')
            ])
            .filter(pl.col('Outcome_Binary') == pl.col('Outcome_Binary').max().over(INTERACTION_ID, NAME, RANK))

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
            return ih_analysis.collect(background=background, streaming=streaming)
        else:
            return ih_analysis
    except Exception as e:
        traceback.print_exc()
        raise e


@timed
def compact_engagement_data(eng_data: pl.DataFrame,
                            config: dict) -> pl.DataFrame:
    grp_by = config['group_by']
    grp_by = list(set(grp_by))
    data_copy = (
        eng_data
        .group_by(grp_by + [MODELCONTROLGROUP])
        .agg(pl.sum("Negatives").alias("Negatives"),
             pl.sum("Positives").alias("Positives"),
             pl.sum("Count").alias("Count"))
    )

    return data_copy
