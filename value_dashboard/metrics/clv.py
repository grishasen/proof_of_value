import datetime
import traceback

import polars as pl
from dateutil.relativedelta import relativedelta

from value_dashboard.metrics.constants import CLV_MODEL
from value_dashboard.metrics.constants import CUSTOMER_ID
from value_dashboard.metrics.constants import PURCHASED_DATE, ONE_TIME_COST, HOLDING_ID
from value_dashboard.metrics.constants import RECURRING_PERIOD, RECURRING_COST
from value_dashboard.metrics.constants import rfm_config_dict
from value_dashboard.utils.timer import timed


@timed
def clv(holdings: pl.LazyFrame, config: dict, streaming=False, background=False):
    holding_id_col = config['order_id_col'] if 'order_id_col' in config.keys() else HOLDING_ID
    lifespan = config['lifespan'] if 'lifespan' in config.keys() else 2000
    customer_id_col = config['customer_id_col'] if 'customer_id_col' in config.keys() else CUSTOMER_ID
    monetary_value_col = config['monetary_value_col'] if 'monetary_value_col' in config.keys() else ONE_TIME_COST
    purchase_date_col = config['purchase_date_col'] if 'purchase_date_col' in config.keys() else PURCHASED_DATE
    clv_model = config['model'] if 'model' in config.keys() else CLV_MODEL
    recurring_period_col = config['recurring_period'] if 'recurring_period' in config.keys() else RECURRING_PERIOD
    recurring_cost_col = config['recurring_cost'] if 'recurring_cost' in config.keys() else RECURRING_COST
    mand_props_grp_by = config['group_by'] + [customer_id_col, 'Year', 'Quarter']

    if "filter" in config:
        filter_exp_cmp = config["filter"]
        if not isinstance(filter_exp_cmp, str):
            holdings = holdings.filter(config["filter"])

    holdings = holdings.filter(pl.col(purchase_date_col) > (datetime.datetime.now() - relativedelta(years=lifespan)))
    holdings = holdings.with_columns(pl.col(monetary_value_col).cast(pl.Float64))
    try:
        data_aggr = (
            holdings
            .with_columns([
                pl.col(purchase_date_col).dt.date().alias('Day'),
                pl.col(purchase_date_col).dt.strftime('%Y-%m').alias('Month'),
                pl.col(purchase_date_col).dt.year().cast(pl.Utf8).alias('Year'),
                (pl.col(purchase_date_col).dt.year().cast(pl.Utf8) + '_Q' +
                 pl.col(purchase_date_col).dt.quarter().cast(pl.Utf8)).alias('Quarter')
            ])
            .group_by(mand_props_grp_by)
            .agg(
                [
                    pl.count().alias('Count'),
                    pl.col(holding_id_col).n_unique().alias("unique_holdings"),
                    pl.sum(monetary_value_col).alias('lifetime_value'),
                    pl.min(purchase_date_col).alias("MinPurchasedDate"),
                    pl.max(purchase_date_col).alias("MaxPurchasedDate")
                ]
                +
                (
                    [
                        (pl.col(recurring_cost_col) * pl.col(recurring_period_col)).sum().alias("recurring_costs")
                    ] if clv_model == 'contractual' else []
                )
            )
        )
        if clv_model == 'contractual':
            data_aggr = (
                data_aggr
                .with_columns(
                    [
                        (pl.col('recurring_costs') + pl.col('lifetime_value')).alias('lifetime_value_a')
                    ]
                )
                .drop('lifetime_value')
                .rename({'lifetime_value_a': 'lifetime_value'})
            )
        if background:
            return data_aggr.collect(background=background, engine="streaming" if streaming else "auto")
        else:
            return data_aggr
    except Exception as e:
        traceback.print_exc()
        raise e


_default_rfm_segment_config = {
    "Premium Customer": [
        "334",
        "443",
        "444",
        "344",
        "434",
        "433",
        "343",
        "333",
    ],
    "Repeat Customer": ["244", "234", "232", "332", "143", "233", "243", "242"],
    "Top Spender": [
        "424",
        "414",
        "144",
        "314",
        "324",
        "124",
        "224",
        "423",
        "413",
        "133",
        "323",
        "313",
        "134",
    ],
    "At Risk Customer": [
        "422",
        "223",
        "212",
        "122",
        "222",
        "132",
        "322",
        "312",
        "412",
        "123",
        "214",
    ],
    "Inactive Customer": ["411", "111", "113", "114", "112", "211", "311"],
}


def rfm_summary(holdings_aggr: pl.DataFrame, config: dict):
    """Summarize pre-aggregated dataframe for use in CLV calculations and analysis.
    Columns added per customer id:
        frequency, recency, T, monetary_value
    """
    customer_id_col = config['customer_id_col'] if 'customer_id_col' in config.keys() else CUSTOMER_ID
    mand_props_grp_by = config['group_by'] + [customer_id_col]
    rfm_segment_config = rfm_config_dict.get(
        config['rfm_segment_config'] if 'rfm_segment_config' in config.keys() else 'NA', _default_rfm_segment_config)

    observation_period_end_ts = holdings_aggr.select(pl.col("MaxPurchasedDate").max()).item()
    time_scaler = 1.0
    segments_list = [str(x) for x in list(range(1, 5))]
    segments_recency_list = [str(x) for x in list(range(4, 0, -1))]
    segment_names = {}
    for key in rfm_segment_config.keys():
        val_list = rfm_segment_config.get(key)
        for val in val_list:
            segment_names[val] = key

    summary = (
        holdings_aggr
        .group_by(mand_props_grp_by)
        .agg(
            [
                pl.col('Count').sum(),
                pl.col("unique_holdings").sum(),
                pl.col('lifetime_value').sum(),
                pl.col("MinPurchasedDate").min(),
                pl.col("MaxPurchasedDate").max()
            ])
        .with_columns(
            [
                (pl.col("unique_holdings") - 1).alias('frequency'),
                ((pl.col("MaxPurchasedDate") - pl.col("MinPurchasedDate")).dt.total_days() / time_scaler).alias(
                    "recency"),
                ((pl.lit(observation_period_end_ts) - pl.col("MinPurchasedDate")).dt.total_days() / time_scaler).alias(
                    'tenure'),
                (pl.col('lifetime_value') / pl.col("unique_holdings")).alias('monetary_value')
            ]
        )
        .with_columns(
            [
                (pl.col('tenure') - pl.col('recency')).alias('recency'),
                pl.when(pl.col('frequency') == 0).then(pl.lit(0.0)).otherwise(pl.col('monetary_value')).alias(
                    'monetary_value')
            ]
        )
        .filter(pl.col(customer_id_col).is_not_null())
        .with_columns(
            pl.col('frequency').qcut(4, labels=segments_list, allow_duplicates=True).alias('f_quartile'),
            pl.col('monetary_value').qcut(4, labels=segments_list, allow_duplicates=True).alias('m_quartile'),
            pl.col('recency').qcut(4, labels=segments_recency_list, allow_duplicates=True).alias('r_quartile')
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("r_quartile"),
                    pl.col("f_quartile"),
                    pl.col("m_quartile")
                ],
                separator="",
            ).alias("rfm_score")
        )
        .with_columns(
            pl.col("rfm_score").replace(segment_names, default="Unknown").alias("rfm_segment")
        )
        .drop(["MinPurchasedDate", "MaxPurchasedDate", "rfm_score"])
    )
    return summary
