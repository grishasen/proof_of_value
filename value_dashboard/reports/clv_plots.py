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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return rep_filtered_data


@timed
def clv_polarbar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                      config: dict) -> pd.DataFrame:
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def clv_treemap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                     config: dict) -> pd.DataFrame:
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def clv_exposure_plot(data: Union[pl.DataFrame, pd.DataFrame],
                      config: dict) -> pd.DataFrame:
    data = calculate_reports_data(data, config).to_pandas()
    clv_analysis = pl.from_pandas(filter_dataframe(align_column_types(data), case=False))

    if clv_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    clv_analysis = pds.sample(clv_analysis, 100).sort(['recency', 'tenure'])
    fig = clv_plot_customer_exposure(clv_analysis, linewidth=0.5, size=0.75)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return clv_analysis.to_pandas()


@timed
def clv_correlation_plot(data: Union[pl.DataFrame, pd.DataFrame],
                         config: dict) -> pd.DataFrame:
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def clv_totals_cards_subplot(clv_analysis: Union[pl.DataFrame, pd.DataFrame],
                             config: dict):
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
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
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
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
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
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    return clv


def clv_plot_customer_exposure(
        df: pl.DataFrame,
        linewidth: float | None = None,
        size: float | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        padding: float = 0.25
) -> go.Figure:
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
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns for this report: {', '.join(missing)}")
        st.stop()


def _theme_template() -> str:
    try:
        return "plotly_dark" if st.get_option("theme.base") == "dark" else "none"
    except Exception:
        return "none"


def _prepare_for_plot(data: Union[pl.DataFrame, pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
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
    RFM 3D / 2D density
    - 3D scatter: X=Recency, Y=Frequency, Z selectable; color by Monetary or CLV (or RFM score)
    - 2D density: R vs F with heatmap/contour; color is agg of Monetary/CLV/RFM score
    - Facet by segment/channel or any categorical
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

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return pdf_plot
