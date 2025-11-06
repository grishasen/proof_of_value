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
    """
    Compute Customer Lifetime Value (CLV) aggregates from holdings.

    Builds time-based features from the purchase date, groups by configured dimensions
    (including customer and calendar buckets), and computes:
    - number of unique holdings,
    - aggregated lifetime value,
    - min/max purchase timestamps,
    - (optionally) recurring costs for contractual CLV.

    Depending on the configured CLV model:
    - **non-contractual**: lifetime value is the sum of one-time monetary values.
    - **contractual**: lifetime value includes recurring charges aggregated as
      `(recurring_cost * recurring_period)` in addition to the one-time values.

    Optionally returns a *collected* (eager) DataFrame if `background=True`, otherwise
    returns a **LazyFrame** pipeline (preferred for downstream composition).

    Parameters
    ----------
    holdings : pl.LazyFrame
        LazyFrame of holdings with at least columns referenced by `config`, typically:
        customer ID, order/holding ID, purchase timestamp, monetary value, and optionally
        recurring cost/period.
    config : dict
        Configuration dictionary that may include:
        - `group_by` : list[str]
            Additional grouping keys (besides customer and calendar buckets).
        - `order_id_col` : str, optional (default: `HOLDING_ID`)
        - `lifespan` : int, optional (default: 2000)
            Number of years back from now to include in the analysis window.
        - `customer_id_col` : str, optional (default: `CUSTOMER_ID`)
        - `monetary_value_col` : str, optional (default: `ONE_TIME_COST`)
        - `purchase_date_col` : str, optional (default: `PURCHASED_DATE`)
        - `model` : str, optional (default: `CLV_MODEL`)
            'contractual' or any other value for non-contractual.
        - `recurring_period` : str, optional (default: `RECURRING_PERIOD`)
        - `recurring_cost` : str, optional (default: `RECURRING_COST`)
        - `filter` : Any
            Optional pre-built Polars expression; if present and not a string, it is
            applied as a `.filter(...)` on `holdings` before aggregation.
    streaming : bool, default False
        If `background=True`, controls the engine used by `.collect(engine=...)`.
        Otherwise ignored (the function returns a LazyFrame).
    background : bool, default False
        If True, the aggregation is collected immediately and returned as an eager
        DataFrame (possibly in the background), otherwise a LazyFrame is returned.

    Returns
    -------
    pl.LazyFrame or pl.DataFrame
        - LazyFrame when `background=False`.
        - DataFrame when `background=True` (collected via `engine="streaming"` if `streaming=True`).

    Notes
    -----
    - A lifespan filter is applied: `purchase_date > now - relativedelta(years=lifespan)`.
    - Monetary values are cast to `Float64`.
    - Calendar features created: `Day`, `Month` (YYYY-MM), `Year` (utf-8), `Quarter` (YYYY_Q#).
    - Grouping keys always include: `config['group_by'] + [customer_id_col, 'Year', 'Quarter']`.
    - Contractual CLV augments `lifetime_value` with a sum of
      `(recurring_cost * recurring_period)` per group.
    - If `config['filter']` is a non-string value (e.g., a Polars predicate expression),
      it is applied to the `holdings` LazyFrame before any transformations.

    Raises
    ------
    Exception
        Propagates any exception from Polars operations or expression logic. The traceback
        is printed before re-raising.

    """
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
    """
    Summarize pre-aggregated CLV holdings to RFM metrics and segments.

    Converts aggregated holdings (per-group) into **RFM** features and segments:
    - `frequency` : number of repeat purchases (unique_holdings - 1),
    - `recency` : time since last purchase within observation window,
    - `tenure` : time from first purchase to the observation period end,
    - `monetary_value` : average revenue per purchase (LV / unique_holdings),
    - Quartiles for R, F, M (`r_quartile`, `f_quartile`, `m_quartile`),
    - Combined `rfm_seg` code (e.g., '344') mapped to `rfm_segment` by config,
    - `rfm_score` : mean of the three quartiles.

    Parameters
    ----------
    holdings_aggr : pl.DataFrame
        Eager DataFrame with at least the following columns (as produced by `clv(...)`):
        - `unique_holdings`, `lifetime_value`,
        - `MinPurchasedDate`, `MaxPurchasedDate`,
        - group-by keys including the configured customer id.
    config : dict
        Configuration dictionary that may include:
        - `group_by` : list[str]
            Additional grouping keys to preserve.
        - `customer_id_col` : str, optional (default: `CUSTOMER_ID`)
        - `rfm_segment_config` : str, optional
            Key used to select a segmentation mapping from `rfm_config_dict`;
            falls back to `_default_rfm_segment_config` when missing.

    Returns
    -------
    pl.DataFrame
        DataFrame with RFM features and segments per `group_by + [customer_id_col]`,
        sorted descending by grouping keys. Columns include:
        `customers_count`, `unique_holdings`, `lifetime_value`, `frequency`,
        `recency`, `tenure`, `monetary_value`, quartiles (`r_quartile`, `f_quartile`,
        `m_quartile`), `rfm_segment`, `rfm_score`, and original grouping columns.

    Notes
    -----
    - `observation_period_end_ts` is inferred as the **max** of `MaxPurchasedDate` in the input.
    - `recency` is redefined as `tenure - recency` after initial computation so that larger
      values reflect more recent behavior (aligned to the chosen quartile labeling).
    - Quartiles are computed with `qcut(4)` and labeled as strings `'1'..'4'`. For recency,
      labels are reversed (`'4'..'1'`) so that **more recent** -> **higher** quartile.
    - Segment mapping is built from a dict of RFM codes (e.g., `'344'`) to human-readable
      names, defaulting to `"Unknown"` for unseen codes.
    - Drops intermediate columns: `MinPurchasedDate`, `MaxPurchasedDate`, and the raw `rfm_seg`.

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
                pl.col(customer_id_col).n_unique().alias('customers_count'),
                pl.col("unique_holdings").sum().round(2),
                pl.col('lifetime_value').sum().round(2),
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
            ).alias("rfm_seg")
        )
        .with_columns(
            pl.col("rfm_seg").replace(segment_names, default="Unknown").alias("rfm_segment")
        )
        .with_columns(
            pl.mean_horizontal(
                [
                    pl.col("r_quartile").cast(pl.String).str.to_decimal(scale=4),
                    pl.col("f_quartile").cast(pl.String).str.to_decimal(scale=4),
                    pl.col("m_quartile").cast(pl.String).str.to_decimal(scale=4)
                ]
            ).round(2).alias("rfm_score")
        )
        .sort(mand_props_grp_by, descending=True)
        .drop(["MinPurchasedDate", "MaxPurchasedDate", "rfm_seg"])
    )
    return summary
