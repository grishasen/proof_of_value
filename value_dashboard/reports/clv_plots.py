from typing import Dict, Any, Iterable

import numpy as np
import plotly.graph_objs as go
import polars_ds.sample_and_split as pds
from lifetimes import BetaGeoFitter, ParetoNBDFitter, GammaGammaFitter
from plotly.subplots import make_subplots

from value_dashboard.metrics.constants import CUSTOMER_ID
from value_dashboard.reports.shared_plot_utils import *
from value_dashboard.utils.config import get_config


@timed
def clv_histogram_plot(data: Union[pl.DataFrame, pd.DataFrame],
                       config: dict) -> pd.DataFrame:
    """
    Render an interactive CLV histogram with optional normalization, aggregation, and faceting.

    The function computes and filters the report dataset, then displays a Plotly histogram in
    Streamlit with user-selectable X/Y axes, normalization (``histnorm``), aggregation (``histfunc``),
    and cumulative counts. Faceting by row/column and color grouping are supported via keys in
    ``config``.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source data. Can be Polars or pandas; will be converted as needed for plotting.
    config : dict
        Plot configuration. Expected keys include:
        - ``x``: str, one of ``{'frequency','recency','monetary_value','tenure'}`` (required)
        - ``y``: Optional[str], one of ``{None,'lifetime_value','unique_holdings'}``
        - ``facet_row``: Optional[str], categorical column for faceting by rows
        - ``facet_column``: Optional[str], categorical column for faceting by columns
        - ``color``: Optional[str], categorical column for color groups
        - ``description``: str, chart title/description

    Returns
    -------
    pandas.DataFrame
        The filtered dataset used to build the histogram.

    Raises
    ------
    Streamlit StopException
        If the filtered dataset is empty, the function warns and stops the Streamlit run.

    Notes
    -----
    - Histogram normalization supports: ``''``, ``'percent'``, ``'probability'``,
      ``'density'``, ``'probability density'``.
    - Figure height adapts automatically to the number of facet rows if ``facet_row`` is set.
    """
    report_data = calculate_reports_data(data, config).to_pandas()
    rep_filtered_data = filter_dataframe(align_column_types(report_data), case=False)
    if rep_filtered_data.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        options_x = ['frequency', 'recency', 'monetary_value', 'tenure']
        config['x'] = st.selectbox(
            label='X-Axis',
            options=options_x,
            index=options_x.index(config['x']),
            help="Select X-Axis value."
        )
    with c2:
        options_histnorm = ['', 'percent', 'probability', 'density', 'probability density']
        histnorm = st.selectbox(
            label='Normalization',
            options=options_histnorm,
            index=options_histnorm.index(''),
            help="Select histnorm option for the plot."
        )
    with c3:
        options_y = [None, 'lifetime_value', 'unique_holdings']
        y = st.selectbox(
            label='Y-Axis',
            options=options_y,
            index=options_y.index(None),
            help="Select Y-Axis value."
        )
    with c4:
        options_histfunc = ['count', 'sum', 'avg', 'min', 'max']
        histfunc = st.selectbox(
            label='Y-Axis Aggregation',
            options=options_histfunc,
            index=options_histfunc.index('count'),
            help="Select histfunc option for the plot."
        )
    with c5:
        cumulative = st.radio(
            'Cumulative',
            (False, True),
            horizontal=True
        )

    if 'facet_row' in config.keys():
        height = max(640, 300 * len(rep_filtered_data[config['facet_row']].unique()))
    else:
        height = 640

    fig = px.histogram(
        rep_filtered_data,
        x=config['x'],
        y=y,
        histnorm=histnorm,
        histfunc=histfunc,
        cumulative=cumulative,
        facet_col=config['facet_column'] if 'facet_column' in config.keys() else None,
        facet_row=config['facet_row'] if 'facet_row' in config.keys() else None,
        color=config['color'] if 'color' in config.keys() else None,
        title=config['description'],
        height=height,
        text_auto=True,
        marginal="box",
        # barmode='group'
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return rep_filtered_data


@timed
def clv_polarbar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                      config: dict) -> pd.DataFrame:
    """
    Display a polar bar chart for CLV metrics grouped by categorical dimensions.

    The function aggregates customer-level metrics by selected angular (theta) and color
    categories, then renders a Plotly ``bar_polar`` chart in Streamlit. It also shows a
    top-level metrics card row via ``clv_totals_cards_subplot``.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source dataset containing CLV metrics and segment/group columns.
    config : dict
        Plot configuration. Expected keys include:
        - ``r``: str, radial metric (``'lifetime_value'``, ``'unique_holdings'``, or ``'monetary_value'``)
        - ``theta``: str, angular categorical axis (e.g., ``'rfm_segment'`` or any in ``group_by``)
        - ``color``: str, category used for color grouping
        - ``group_by``: list[str], dimensions to group by
        - ``description``: str, chart title
        - ``showlegend``: str convertible to bool

    Returns
    -------
    pandas.DataFrame
        Aggregated and filtered dataset shown in the polar chart.

    Raises
    ------
    Streamlit StopException
        If there is no data after filtering.

    Notes
    -----
    - The chart theme follows the Streamlit theme (dark vs none) for better visual integration.
    - When no grouping is provided, raw data are passed through to the plot.
    """
    clv_totals_cards_subplot(data, config)
    data = calculate_reports_data(data, config).to_pandas()
    c1, c2, c3 = st.columns(3)
    with c1:
        options_r = ['lifetime_value', 'unique_holdings', 'monetary_value']
        config['r'] = st.selectbox(
            label='Radial-Axis',
            options=options_r,
            index=options_r.index(config['r']),
            help="Select Radial-Axis value."
        )
    with c2:
        options_theta = list(set(['rfm_segment'] + config['group_by']))
        config['theta'] = st.selectbox(
            label='Angular axis in polar coordinates',
            options=options_theta,
            index=options_theta.index(config['theta']),
            help="Select  angular axis in polar coordinates."
        )
    with c3:
        options_color = list(set(['rfm_segment', 'f_quartile', 'r_quartile', 'm_quartile'] + config['group_by']))
        config['color'] = st.selectbox(
            label='Colour',
            options=options_color,
            index=options_color.index(config['color']),
            help="Select colour value."
        )

    grp_by = []
    if not config['theta'] in grp_by:
        grp_by.append(config['theta'])
    if not config['color'] in grp_by:
        grp_by.append(config['color'])

    if grp_by:
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        report_data = (
            data
            .group_by(grp_by)
            .agg(
                pl.sum("customers_count").alias("Customers count"),
                pl.sum("lifetime_value").alias("lifetime_value"),
                pl.sum("unique_holdings").alias("unique_holdings"),
                pl.mean("monetary_value").alias("monetary_value"),
                pl.mean("frequency").alias("Avg frequency"),
                pl.mean("rfm_score").alias("Avg rfm score")
            )
            .sort(grp_by)
        )
    else:
        report_data = data
    report_data = report_data.to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)

    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    theme = st.context.theme.type
    if theme is None:
        template = 'none'
    else:
        if theme == 'dark':
            template = 'plotly_dark'
        else:
            template = 'none'
    fig = px.bar_polar(ih_analysis,
                       r=config["r"],
                       theta=config["theta"],
                       color=config["color"],
                       barmode="group",
                       template=template,
                       title=config['description']
                       )
    fig.update_layout(
        polar_hole=0.25,
        height=700,
        # width=1400,
        margin=dict(b=25, t=50, l=0, r=0),
        showlegend=strtobool(config["showlegend"])
    )
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def clv_treemap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                     config: dict) -> pd.DataFrame:
    """
    Plot a treemap of customer counts with CLV/RFM context.

    Aggregates metrics across ``rfm_segment`` and additional ``group_by`` dimensions and
    renders a Plotly treemap where area encodes customer count and color encodes average
    RFM score.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source dataset with RFM/CLV metrics.
    config : dict
        Configuration for grouping and labeling. Expected keys include:
        - ``group_by``: list[str], dimensions to group by (in addition to ``'rfm_segment'``)
        - ``description``: str, chart title

    Returns
    -------
    pandas.DataFrame
        Aggregated and filtered dataset used by the treemap.

    Raises
    ------
    Streamlit StopException
        If the filtered dataset is empty.

    Notes
    -----
    - Color scale ranges from red (low) to green (high) for the average RFM score.
    - A summary metrics row is displayed above the chart.
    """
    clv_totals_cards_subplot(data, config)
    data = calculate_reports_data(data, config)
    grp_by = ['rfm_segment'] + config['group_by']
    if grp_by:
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        report_data = (
            data
            .group_by(grp_by)
            .agg(
                pl.sum("customers_count").round(2).alias("Customers count"),
                pl.sum("lifetime_value").round(2).alias("lifetime_value"),
                pl.sum("unique_holdings").round(2).alias("unique_holdings"),
                pl.mean("monetary_value").round(2).alias("monetary_value"),
                pl.mean("frequency").round(2).alias("Avg frequency"),
                pl.mean("rfm_score").round(2).alias("Avg rfm score")
            )
            .sort(grp_by)
        )
    else:
        report_data = data
    report_data = report_data.to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)

    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    fig = px.treemap(ih_analysis, path=[px.Constant(" ")] + grp_by, values='Customers count',
                     color="Avg rfm score",
                     color_continuous_scale=["#D61F1F", "#E03C32", "#FFD301", "#639754", "#006B3D"],
                     title=config['description'],
                     hover_data=['lifetime_value', 'unique_holdings', 'monetary_value', "Avg frequency",
                                 "Avg rfm score"],
                     height=640,
                     )
    fig.update_traces(textinfo="label+text+value+percent root", root_color="lightgrey")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def clv_exposure_plot(data: Union[pl.DataFrame, pd.DataFrame],
                      config: dict) -> pd.DataFrame:
    """
    Visualize customer exposure (recency vs. tenure) as line segments with markers.

    The function prepares and filters the CLV dataset, samples up to 100 customers for
    readability, and renders an exposure plot where each customer’s recency–tenure span
    is represented by two colored segments.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source customer-level dataset.
    config : dict
        Report configuration passed to ``calculate_reports_data``.
        Unused keys are safely ignored.

    Returns
    -------
    pandas.DataFrame
        The (sampled) filtered dataset used for plotting.

    Raises
    ------
    Streamlit StopException
        If no data remain after filtering.

    See Also
    --------
    clv_plot_customer_exposure : Low-level plotting routine used to render the figure.
    """
    data = calculate_reports_data(data, config).to_pandas()
    clv_analysis = pl.from_pandas(filter_dataframe(align_column_types(data), case=False))

    if clv_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    clv_analysis = pds.sample(clv_analysis, 100).sort(['recency', 'tenure'])
    fig = clv_plot_customer_exposure(clv_analysis, linewidth=0.5, size=0.75)
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return clv_analysis.to_pandas()


@timed
def clv_correlation_plot(data: Union[pl.DataFrame, pd.DataFrame],
                         config: dict) -> pd.DataFrame:
    """
    Compute and display pairwise correlations between two CLV-related metrics.

    Supports Pearson, Kendall, or Spearman correlation. Optionally facets the heatmap
    by a categorical column (e.g., ``'rfm_segment'`` or ``'ControlGroup'``) to compare
    correlations across groups.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source dataset including numeric CLV metrics.
    config : dict
        Configuration dictionary. Keys include:
        - ``x``: str, metric on the X-axis (from ``{'recency','frequency','monetary_value','tenure','lifetime_value'}``)
        - ``y``: str, metric on the Y-axis (same choices as ``x``)
        - ``facet_col``: Optional[str], categorical column for group-wise correlation heatmaps

    Returns
    -------
    pandas.DataFrame
        Filtered dataset used for the correlation computation and plotting.

    Raises
    ------
    Streamlit StopException
        If the filtered dataset is empty.

    Notes
    -----
    - When ``facet_col`` is provided, the output heatmap displays one 2×2 matrix per facet.
    """
    data = calculate_reports_data(data, config).to_pandas()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        options_par1 = ['recency', 'frequency', 'monetary_value', 'tenure', 'lifetime_value']
        config['x'] = st.selectbox(
            label='X-Axis',
            options=options_par1,
            index=options_par1.index(config['x']),
            help="Select X-Axis value."
        )
    with c2:
        options_par2 = ['recency', 'frequency', 'monetary_value', 'tenure', 'lifetime_value']
        config['y'] = st.selectbox(
            label='Y-Axis',
            options=options_par2,
            index=options_par2.index(config['y']),
            help="Select Y-Axis value."
        )
    with c3:
        options_facet_col = [None, 'rfm_segment', 'ControlGroup']
        if 'facet_col' in config.keys():
            config['facet_col'] = st.selectbox(
                label='Group By',
                options=options_facet_col,
                index=options_facet_col.index(config['facet_col']),
                help="Select Group By value."
            )
        else:
            config['facet_col'] = st.selectbox(
                label='Group By',
                options=options_facet_col,
                help="Select Group By value."
            )
    with c4:
        method = st.selectbox(
            label='Correlation method',
            options=['pearson', 'kendall', 'spearman'],
            help="""Method used to compute correlation:
- pearson : Standard correlation coefficient
- kendall : Kendall Tau correlation coefficient
- spearman : Spearman rank correlation"""
        )
    ih_analysis = filter_dataframe(align_column_types(data), case=False)

    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    if config['facet_col']:
        facets = sorted(ih_analysis[config['facet_col']].unique())
        img_sequence = []
        for facet_col in facets:
            img_sequence.append(
                ih_analysis[ih_analysis[config['facet_col']] == facet_col][[config['x'], config['y']]]
                .corr(method=method)
            )
        img_sequence = np.array(img_sequence)

    else:
        img_sequence = ih_analysis[[config['x'], config['y']]].corr(method=method)

    fig = px.imshow(
        img_sequence,
        color_continuous_scale='Viridis',
        text_auto=".4f",
        aspect='auto',
        x=[config['x'], config['y']],
        y=[config['x'], config['y']],
        facet_col=0 if config['facet_col'] else None,
        facet_col_wrap=4 if config['facet_col'] else None,
    )
    if config['facet_col']:
        for i, label in enumerate(facets):
            fig.layout.annotations[i]['text'] = label
    fig.update_layout(
        title=method.title() + " correlation between " + config['x'] + " and " + config['y']
    )
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def clv_totals_cards_subplot(clv_analysis: Union[pl.DataFrame, pd.DataFrame],
                             config: dict):
    """
    Render a row of Streamlit metric cards for CLV overview.

    Displays:
      1) Unique customers
      2) Total lifetime value
      3) Prior year average CLTV with YoY delta
      4) Current year average CLTV with YoY delta

    Parameters
    ----------
    clv_analysis : pl.DataFrame or pandas.DataFrame
        Dataset containing at least a ``lifetime_value`` column and a customer identifier.
        Must include a ``'Year'`` column for YoY metrics; if fewer than three years are present,
        only the first two cards are shown.
    config : dict
        Configuration passed into ``calculate_reports_data`` and for resolving the
        customer id column via ``get_config()['metrics'][config['metric']]['customer_id_col']``.
        If not present, a default ``CUSTOMER_ID`` is used.

    Returns
    -------
    None
        This function produces Streamlit UI side effects only.

    Notes
    -----
    - Uses Polars for fast aggregations; pandas inputs are converted internally.
    - YoY metrics are computed from the two most recent completed years and the current year.
    """
    if isinstance(clv_analysis, pd.DataFrame):
        clv_analysis = pl.from_pandas(clv_analysis)

    total_data = calculate_reports_data(clv_analysis, config)

    m_config = get_config()["metrics"][config["metric"]]
    customer_id_col = (
        m_config["customer_id_col"]
        if "customer_id_col" in m_config.keys()
        else CUSTOMER_ID
    )

    num_cols = 4
    cols = st.columns(num_cols, vertical_alignment='center')
    unique_customers = total_data.select(pl.col(customer_id_col).n_unique())
    total_value = total_data.select(pl.col("lifetime_value").sum())
    cols[0].metric(label='Unique customers', value='{:,}'.format(unique_customers.item()).replace(",", " "))
    cols[1].metric(label='Total value', value='{:,.2f}'.format(total_value.item()))

    years = clv_analysis.select("Year").unique().sort("Year")["Year"].to_list()
    if len(years) < 3:
        return
    year1, year2, cur_year = years[-3], years[-2], years[-1],

    df_last_two = clv_analysis.filter(pl.col("Year").is_in([year1, year2]))
    avg_per_year = (df_last_two.group_by("Year")
                    .agg((pl.col("lifetime_value").sum() / pl.col(customer_id_col).n_unique()).alias("avg"))
                    )
    avg_sorted = avg_per_year.sort("Year")
    avg1, avg2 = avg_sorted["avg"].to_list()
    percentage_diff = ((avg2 - avg1) / avg1)

    cols[2].metric(label=year2 + ' average CLTV', value='{:,.2f}'.format(avg2),
                   delta='{:.2%} YoY'.format(percentage_diff))

    cur_df = clv_analysis.filter(pl.col("Year").is_in([cur_year]))
    avg_per_year = (cur_df.group_by("Year")
                    .agg((pl.col("lifetime_value").sum() / pl.col(customer_id_col).n_unique()).alias("avg"))
                    )
    avg_sorted = avg_per_year.sort("Year")
    avg, = avg_sorted["avg"].to_list()
    percentage_diff = ((avg - avg2) / avg2)
    cols[3].metric(label=cur_year + ' average CLTV', value='{:,.2f}'.format(avg),
                   delta='{:.2%} YoY'.format(percentage_diff))


@timed
def clv_model_plot(data: Union[pl.DataFrame, pd.DataFrame],
                   config: dict) -> pd.DataFrame:
    """
    Fit classical CLV models and visualize expected purchases or value by segment.

    Supports:
      - **BG/NBD**: expected number of purchases over a horizon
      - **Pareto/NBD**: expected number of purchases over a horizon
      - **Gamma–Gamma (with BG/NBD)**: expected lifetime value over a horizon

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source dataset containing at least ``frequency``, ``recency``, ``tenure``,
        and ``monetary_value`` for Gamma–Gamma. Rows with zero frequency are filtered out.
    config : dict
        Configuration passed to ``calculate_reports_data`` and for summary cards.

    Returns
    -------
    pandas.DataFrame
        The dataset augmented with model predictions (column depends on the selected model).

    Raises
    ------
    ValueError
        Propagated from underlying model fitters if inputs are invalid.

    Notes
    -----
    - Prediction horizon is selected in years and converted to days (``t = 365 * years``).
    - Results are aggregated by ``rfm_segment`` for plotting.
    """
    clv_totals_cards_subplot(data, config)
    clv = calculate_reports_data(data, config).to_pandas()
    clv = clv[clv['frequency'] > 0]
    c1, c2 = st.columns(2)
    with c1:
        options_model = ['Gamma - Gamma Model', 'BG/NBD Model', 'Pareto/NBD model']
        model = st.selectbox(
            label='LTV prediction model',
            options=options_model,
            help="Select LTV prediction model."
        )

    with c2:
        lifespan = [1, 2, 3, 5, 8]
        predict_lifespan = st.selectbox(
            label='Predict LTV in years',
            options=lifespan,
            help="Select LTV prediction time."
        )
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(clv['frequency'], clv['recency'], clv['tenure'], verbose=True)
    t = 365 * predict_lifespan
    clv['expected_number_of_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, clv['frequency'],
                                                                                                  clv['recency'],
                                                                                                  clv['tenure'])
    if model == 'BG/NBD Model':
        clv_plt = clv.groupby('rfm_segment')['expected_number_of_purchases'].mean().reset_index()
        fig = px.bar(clv_plt,
                     x='rfm_segment',
                     y='expected_number_of_purchases',
                     color='rfm_segment',
                     )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        st.plotly_chart(fig, width='stretch', theme="streamlit")
    elif model == 'Pareto/NBD model':
        with st.spinner("Wait for it...", show_time=True):
            pnbmf = ParetoNBDFitter(penalizer_coef=0.001)
            pnbmf.fit(clv['frequency'], clv['recency'], clv['tenure'], verbose=True, maxiter=200)
            clv['expected_number_of_purchases'] = pnbmf.conditional_expected_number_of_purchases_up_to_time(t,
                                                                                                            clv[
                                                                                                                'frequency'],
                                                                                                            clv[
                                                                                                                'recency'],
                                                                                                            clv[
                                                                                                                'tenure'])
            clv_plt = clv.groupby('rfm_segment')['expected_number_of_purchases'].mean().reset_index()
            fig = px.bar(clv_plt,
                         x='rfm_segment',
                         y='expected_number_of_purchases',
                         color='rfm_segment',
                         )

            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
            st.plotly_chart(fig, width='stretch', theme="streamlit")
    else:
        with st.spinner("Wait for it...", show_time=True):
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(clv["frequency"], clv["monetary_value"], verbose=True)
            clv["expected_lifetime_value"] = ggf.customer_lifetime_value(
                bgf,
                clv["frequency"],
                clv["recency"],
                clv["tenure"],
                clv["monetary_value"],
                time=365 * predict_lifespan,
                freq="D",
                discount_rate=0.01,
            )
            clv_plt = clv.groupby('rfm_segment')['expected_lifetime_value'].mean().reset_index()
            fig = px.bar(clv_plt,
                         x='rfm_segment',
                         y='expected_lifetime_value',
                         color='rfm_segment',
                         )

            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
            st.plotly_chart(fig, width='stretch', theme="streamlit")

    return clv


def clv_plot_customer_exposure(
        df: pl.DataFrame,
        linewidth: float | None = None,
        size: float | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        padding: float = 0.25
) -> go.Figure:
    """
    Build a Plotly figure showing each customer's recency–tenure exposure.

    Each row corresponds to a customer. Two line segments are drawn:
    1) from 0 to recency, and 2) from recency to tenure; optional markers denote endpoints.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame with columns ``'recency'`` and ``'tenure'``.
    linewidth : float or None, optional
        Line width for exposure segments. If ``None``, Plotly defaults are used.
        Must be non-negative.
    size : float or None, optional
        Marker size for endpoints. If ``None``, Plotly defaults are used.
        Must be non-negative.
    labels : list of str or None, optional
        Custom legend labels for ``[recency, tenure]`` markers.
    colors : list of str or None, optional
        Two-item list of colors for the recency and tenure segments/markers, respectively.
        Defaults to ``['blue','orange']``.
    padding : float, default 0.25
        Extra margin added to both axes to provide visual padding. Must be non-negative.

    Returns
    -------
    plotly.graph_objects.Figure
        The exposure plot figure.

    Raises
    ------
    ValueError
        If ``padding``, ``size``, or ``linewidth`` are negative, or if ``colors`` is not length 2.

    Notes
    -----
    - Uses ``Scattergl`` for efficient rendering on large datasets.
    - Legend is hidden by default and can be toggled in the returned figure.
    """
    if padding < 0:
        raise ValueError("padding must be non-negative")

    if size is not None and size < 0:
        raise ValueError("size must be non-negative")

    if linewidth is not None and linewidth < 0:
        raise ValueError("linewidth must be non-negative")

    n = len(df)
    customer_idx = list(range(1, n + 1))

    recency = df['recency'].to_list()
    T = df['tenure'].to_list()

    if colors is None:
        colors = ["blue", "orange"]

    if len(colors) != 2:
        raise ValueError("colors must be a sequence of length 2")

    recency_color, T_color = colors
    fig = make_subplots()
    for idx, (rec, t) in enumerate(zip(recency, T)):
        fig.add_trace(
            go.Scattergl(
                x=[0, rec],
                y=[customer_idx[idx], customer_idx[idx]],
                mode='lines',
                line=dict(color=recency_color, width=linewidth)
            )
        )

    for idx, (rec, t) in enumerate(zip(recency, T)):
        fig.add_trace(
            go.Scattergl(
                x=[rec, t],
                y=[customer_idx[idx], customer_idx[idx]],
                mode='lines',
                line=dict(color=T_color, width=linewidth)
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=recency,
            y=customer_idx,
            mode='markers',
            marker=dict(color=recency_color, size=size),
            name=labels[0] if labels else 'Recency'
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=T,
            y=customer_idx,
            mode='markers',
            marker=dict(color=T_color, size=size),
            name=labels[1] if labels else 'tenure'
        )
    )

    fig.update_layout(
        title="Customer Exposure",
        xaxis_title="Time since first purchase",
        yaxis_title="Customer",
        xaxis=dict(range=[-padding, max(T) + padding]),
        yaxis=dict(range=[1 - padding, n + padding]),
        showlegend=False,
        barmode='group',
        height=640
    )

    return fig


# ---- mini helpers (no external side effects) ----
def _to_polars(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    """
    Convert a pandas or Polars DataFrame to Polars.

    Parameters
    ----------
    df : pl.DataFrame or pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pl.DataFrame
        Polars representation of ``df`` (returns unchanged if already Polars).
    """
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    """
    Validate the presence of required columns in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    required : Iterable[str]
        Column names that must be present.

    Returns
    -------
    None
        Emits a Streamlit error and halts execution if any column is missing.

    Raises
    ------
    Streamlit StopException
        Triggered after reporting missing columns to the user.

    Examples
    --------
    >>> _columns_exist(pd.DataFrame({'a':[1]}), ['a'])  # passes silently
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns for this report: {', '.join(missing)}")
        st.stop()


def _theme_template() -> str:
    """
    Resolve the Plotly template based on the current Streamlit theme.

    Returns
    -------
    str
        ``'plotly_dark'`` if the app is in dark mode, otherwise ``'none'``.
        If the theme cannot be retrieved, defaults to ``'none'``.
    """
    try:
        return "plotly_dark" if st.get_option("theme.base") == "dark" else "none"
    except Exception:
        return "none"


def _prepare_for_plot(data: Union[pl.DataFrame, pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute report data and apply standard filtering/typing for plotting.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source dataset for the report.
    config : dict
        Configuration passed to ``calculate_reports_data``; may be empty.

    Returns
    -------
    pandas.DataFrame
        Filtered pandas DataFrame ready for visualization.

    Raises
    ------
    Streamlit StopException
        If no data remain after filtering.
    """
    df_pl = _to_polars(data)
    rep_pl = calculate_reports_data(df_pl, config or {})
    pdf = rep_pl.to_pandas()
    pdf = filter_dataframe(align_column_types(pdf), case=False)
    if pdf.empty:
        st.warning("No data available.")
        st.stop()
    return pdf


# ---- main report ----
@timed
def clv_rfm_density_plot(
        data: Union[pl.DataFrame, pd.DataFrame],
        base_config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Plot Recency–Frequency (R–F) landscapes as 3D scatter or 2D density (heatmap/contour).

    Provides an interactive Streamlit UI to choose plotting mode, color metric, sample size,
    and aggregation settings. Useful for exploring customer activity patterns and how they
    relate to value metrics such as monetary value, CLV, or RFM score.

    Parameters
    ----------
    data : pl.DataFrame or pandas.DataFrame
        Source dataset containing at least ``recency`` and ``frequency`` columns, and
        optionally value metrics (e.g., ``monetary_value``, ``lifetime_value``, ``rfm_score``).
    base_config : dict or None, optional
        Base configuration passed to ``calculate_reports_data``; keys may include filters,
        groupings, or other pipeline parameters.

    Returns
    -------
    pandas.DataFrame
        The (possibly sampled) filtered dataset actually plotted.

    Raises
    ------
    Streamlit StopException
        If required columns are missing or the filtered dataset becomes empty.

    Notes
    -----
    - **Modes**
        - *3D scatter*: X=recency, Y=frequency, Z=selectable numeric; colored by chosen metric.
        - *2D density heatmap/contour*: bins R vs F with z-aggregation over the chosen metric.
    - Large datasets are downsampled to improve performance (user-selectable cap).
    """
    cfg = dict(base_config or {})
    pdf = _prepare_for_plot(data, cfg)

    core_needed = {"recency", "frequency"}
    _columns_exist(pdf, core_needed)

    numeric_cols = sorted([c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c].dtype)])

    r1c1, r1c2, r2c1, r2c2 = st.columns(4)
    with r1c1:
        mode = st.selectbox("Plot mode", ["3D scatter", "2D density heatmap", "2D density contour"])
    with r1c2:
        color_metric = st.selectbox(
            "Color by",
            options=[c for c in ["monetary_value", "lifetime_value", "rfm_score"] if c in pdf.columns] or numeric_cols,
            index=0,
            help="Metric used for color; for 2D density, aggregated per bin."
        )
    with r2c1:
        sample_n = st.slider("Max points (sample)", min_value=1_000, max_value=200_000, value=25_000, step=1_000)
    with r2c2:
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

    if mode == "3D scatter":
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            z_axis = st.selectbox(
                "Z-axis",
                options=[c for c in ["monetary_value", "lifetime_value", "rfm_score", "tenure"] if
                         c in pdf.columns] or numeric_cols,
                index=0
            )
        with r3c2:
            point_size = st.slider("Point size", 1, 8, 3)
        with r3c3:
            opacity = st.slider("Opacity", 0.2, 1.0, 0.75, 0.05)
    else:
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            histfunc = st.selectbox("Aggregation", options=["avg", "sum", "count", "max", "min"], index=0)
        with r3c2:
            nbins_x = st.slider("Bins (Recency)", 10, 200, 60, 5)
        with r3c3:
            nbins_y = st.slider("Bins (Frequency)", 10, 200, 60, 5)

    needed = {"recency", "frequency", color_metric}
    if mode == "3D scatter":
        needed.add(z_axis)
    _columns_exist(pdf, needed)

    n = len(pdf)
    if n > sample_n:
        pdf_plot = pdf.sample(n=sample_n, random_state=seed)
    else:
        pdf_plot = pdf

    title_base = "R–F value landscape"
    template = _theme_template()

    if mode == "3D scatter":
        fig = px.scatter_3d(
            pdf_plot,
            x="recency",
            y="frequency",
            z=z_axis,
            color=color_metric,
            opacity=opacity,
            template=template,
            title=f"{title_base} — 3D scatter<br><sup>Color: {color_metric} · Z: {z_axis} · n={len(pdf_plot):,}</sup>",
            height=640
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            margin=dict(t=80, l=0, r=0, b=0),
            scene=dict(
                xaxis_title="Recency",
                yaxis_title="Frequency",
                zaxis_title=z_axis,
            ),
        )

    elif mode == "2D density heatmap":
        fig = px.density_heatmap(
            pdf_plot,
            x="recency",
            y="frequency",
            z=color_metric,
            histfunc=histfunc,
            nbinsx=nbins_x,
            nbinsy=nbins_y,
            template=template,
            title=f"{title_base} — 2D heatmap<br><sup>Color: {color_metric} ({histfunc}) · n={len(pdf_plot):,}</sup>",
        )
        fig.update_layout(
            margin=dict(t=80, l=40, r=10, b=40),
            coloraxis_colorbar_title=color_metric,
        )

    else:
        fig = px.density_contour(
            pdf_plot,
            x="recency",
            y="frequency",
            z=color_metric,
            histfunc=histfunc,
            nbinsx=nbins_x,
            nbinsy=nbins_y,
            template=template,
            title=f"{title_base} — 2D contour<br><sup>Color metric for agg: {color_metric} ({histfunc}) · n={len(pdf_plot):,}</sup>",
        )
        fig.update_traces(contours_coloring="heatmap", contours_showlabels=False)
        fig.update_layout(
            margin=dict(t=80, l=40, r=10, b=40),
            coloraxis_colorbar_title=color_metric,
        )

    fig.update_xaxes(title_text="Recency")
    fig.update_yaxes(title_text="Frequency")

    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return pdf_plot
