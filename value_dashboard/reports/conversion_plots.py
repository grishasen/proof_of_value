import math

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from value_dashboard.reports.shared_plot_utils import *


@timed
def conversion_rate_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict, options_panel: bool = True) -> pd.DataFrame:
    xplot_y_bool = False
    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)
    height = config.get('height', 640)

    if options_panel:
        toggle1, toggle2 = st.columns(2)
        cards_on = toggle1.toggle("Metric totals", value=True, key="Metrics" + config['description'],
                                  help="Show aggregated metric values with difference from mean")
        adv_on = toggle2.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                                help="Show advanced reporting options")

        if adv_on:
            plot_menu = get_plot_parameters_menu(config=config, is_y_axis_required=False)
            x_axis = plot_menu['x']
            y_axis = plot_menu['y']
            facet_column = plot_menu['facet_col']
            facet_row = plot_menu['facet_row']
            xplot_col = plot_menu['color']
            xplot_y_bool = plot_menu['log_y']

    grp_by = [x_axis]
    if not (facet_column == '---'):
        if not facet_column in grp_by:
            grp_by.append(facet_column)
    else:
        facet_column = None
    if not (facet_row == '---'):
        if not facet_row in grp_by:
            grp_by.append(facet_row)
    else:
        facet_row = None

    if not xplot_col in grp_by:
        grp_by.append(xplot_col)

    cp_config = config.copy()
    cp_config['group_by'] = grp_by

    ih_analysis = data.copy()
    if options_panel:
        ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
    if options_panel and cards_on:
        conversion_rate_cards_subplot(ih_analysis, cp_config)

    if len(ih_analysis[x_axis].unique()) < 30:
        fig = px.bar(ih_analysis,
                     x=x_axis,
                     y=y_axis,
                     color=xplot_col,
                     error_y='StdErr',
                     facet_col=facet_column,
                     facet_row=facet_row,
                     barmode="group",
                     title=config['description'],
                     custom_data=[xplot_col],
                     log_y=xplot_y_bool
                     )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=["type", "bar"],
                            label="Bar",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "line"],
                            label="Line",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    showactive=True,
                ),
            ]
        )
    else:
        fig = px.line(
            ih_analysis,
            x=x_axis,
            y=y_axis,
            color=xplot_col,
            title=config['description'],
            facet_col=facet_column,
            facet_row=facet_row,
            custom_data=[xplot_col],
            log_y=xplot_y_bool,
        )

    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickformat=',.2%')
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    yaxis_layout_dict = {yaxis_name + "_tickformat": ',.2%' for yaxis_name in yaxis_names}
    fig.update_layout(yaxis_layout_dict)
    if facet_row:
        height = max(640, 300 * len(ih_analysis[facet_row].unique()))
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=x_axis + ' : %{x}' + '<br>' +
                                          xplot_col + ' : %{customdata[0]}' + '<br>' +
                                          y_axis + ' : %{y:.2%}' + '<extra></extra>')
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def conversion_rate_gauge_plot(data: Union[pl.DataFrame, pd.DataFrame],
                               config: dict) -> pd.DatetimeIndex | None:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
    grp_by = config['group_by']
    ih_analysis = ih_analysis.sort_values(by=grp_by)
    ih_analysis = ih_analysis.reset_index()

    if len(grp_by) == 0:
        st.warning("Group By property for Gauge plot should not be empty.")
        st.stop()
    elif len(grp_by) > 2:
        st.warning("Gauge plot type does not support more than two grouping columns.")
        st.stop()
    elif len(grp_by) == 2:
        cols = ih_analysis[grp_by[0]].unique().shape[0]
        rows = ih_analysis[grp_by[1]].unique().shape[0]
    else:
        cols = math.isqrt(ih_analysis[grp_by[0]].unique().shape[0])
        rows = cols + 1

    reference = config['reference']
    ih_analysis['Name'] = ih_analysis[grp_by].apply(lambda r: ' '.join(r.values.astype(str)), axis=1)
    ih_analysis['CName'] = ih_analysis[grp_by].apply(lambda r: '_'.join(r.values.astype(str)), axis=1)
    fig = make_subplots(rows=rows,
                        cols=cols,
                        specs=[[{"type": "indicator"} for c in range(cols)] for t in range(rows)]
                        )
    fig.update_layout(
        height=300 * rows,
        autosize=True,
        title=config['description'],
        margin=dict(b=10, t=120, l=10, r=10))
    for index, row in ih_analysis.iterrows():
        ref_value = reference.get(row['CName'], None)
        max_value = 1.1 * ih_analysis[ih_analysis[grp_by[0]] == row[grp_by[0]]][config['value']].max()
        gauge = {
            'axis': {'range': [None, max_value], 'tickformat': ',.2%'},
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': ref_value
            }
        }
        if ref_value:
            if row[config['value']] < ref_value:
                gauge = {
                    'axis': {'range': [None, max_value], 'tickformat': ',.2%'},
                    'bar': {'color': '#EC5300' if row[config['value']] < (0.75 * ref_value) else '#EC9B00'},
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': ref_value
                    }
                }

        trace1 = go.Indicator(mode="gauge+number+delta",
                              number={'valueformat': ",.2%"},
                              value=row[config['value']],
                              delta={'reference': ref_value, 'valueformat': ",.2%"},
                              title={'text': row['Name']},
                              gauge=gauge,
                              )
        r, c = divmod(index, cols)
        fig.add_trace(
            trace1,
            row=(r + 1), col=(c + 1)
        )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    ih_analysis.drop(columns=['Name', 'CName', 'index'], inplace=True, errors='ignore')
    return ih_analysis


@timed
def conversion_rate_cards_subplot(ih_analysis: Union[pl.DataFrame, pd.DataFrame],
                                  config: dict):
    grp_by = config['group_by']
    if len(grp_by) > 1:
        grp_by = grp_by[-2:]
    else:
        if len(grp_by) > 0:
            grp_by = grp_by[-1:]
    if isinstance(ih_analysis, pd.DataFrame):
        dfg = ih_analysis.groupby(grp_by)
        ih_analysis = pl.from_pandas(ih_analysis)
    else:
        dfg = ih_analysis.to_pandas().groupby(grp_by)
    if dfg.ngroups > 18:
        grp_by = config['group_by'][-1:]
    data_copy = (
        ih_analysis
        .group_by(grp_by)
        .agg(pl.sum("Negatives").alias("Negatives"),
             pl.sum("Positives").alias("Positives"))
        .with_columns([
            (pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))).alias("ConversionRate")
        ])
        .sort(grp_by)
    )
    average = data_copy.select(((pl.col("Positives") + pl.col("Negatives")).dot(pl.col("ConversionRate"))) / (
        (pl.col("Positives") + pl.col("Negatives"))).sum()).item()
    data_copy = data_copy.to_pandas()
    num_metrics = data_copy.shape[0]
    dims = st.session_state['dashboard_dims']
    if dims:
        main_area_width = dims.get('width')
        max_num_cols = main_area_width // 120
    else:
        max_num_cols = 8
    num_cols = num_metrics if num_metrics < max_num_cols else max_num_cols
    cols = st.columns(num_cols)
    for index, row in data_copy.iterrows():
        if len(grp_by) > 1:
            kpi_name = row.iloc[0] + "  \n" + row.iloc[1]
        else:
            kpi_name = row.iloc[0]
        cols[index % max_num_cols].metric(label=kpi_name, value='{:.2%}'.format(row["ConversionRate"]),
                                          delta='{:.2%}'.format(row["ConversionRate"] - average))
        if (index + 1) % max_num_cols == 0:
            cols = st.columns(max_num_cols)


@timed
def conversion_rate_card(ih_analysis: Union[pl.DataFrame, pd.DataFrame]):
    data_copy = (
        ih_analysis
        .select(["Positives", "Negatives", "Revenue"])
        .sum()
        .with_columns([
            (pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))).alias("ConversionRate")
        ])
    )
    st.metric(label="**Conversion**", value='{:.2%}'.format(data_copy["ConversionRate"].item()), border=False,
              help=f'Conversion rate', delta='Revenue {:,.0f}'.format(data_copy["Revenue"].item()), delta_color='off')


@timed
def conversion_touchpoints_card(ih_analysis: Union[pl.DataFrame, pd.DataFrame]):
    data_copy = (
        ih_analysis
        .select(["Positives", "Touchpoints"])
        .sum()
        .with_columns([
            (pl.col("Touchpoints") / pl.col("Positives")).alias("AvgTouchpoints")
        ])
    )
    st.metric(label="**Total conversions**", value='{:,.0f}'.format(data_copy["Positives"].item()), border=False,
              help=f'Conversions and touchpoints', delta=f'Avg Touchpoints: {data_copy["AvgTouchpoints"].item():.2}',
              delta_color='off')


@timed
def conversion_revenue_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                                 config: dict) -> pd.DataFrame:
    adv_on = st.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    xplot_y_bool = False
    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)
    height = config.get('height', 640)

    if adv_on:
        plot_menu = get_plot_parameters_menu(config=config, is_y_axis_required=False)
        x_axis = plot_menu['x']
        y_axis = plot_menu['y']
        facet_column = plot_menu['facet_col']
        facet_row = plot_menu['facet_row']
        xplot_col = plot_menu['color']
        xplot_y_bool = plot_menu['log_y']

    grp_by = [x_axis]
    if not (facet_column == '---'):
        if not facet_column in grp_by:
            grp_by.append(facet_column)
    else:
        facet_column = None
    if not (facet_row == '---'):
        if not facet_row in grp_by:
            grp_by.append(facet_row)
    else:
        facet_row = None

    if not xplot_col in grp_by:
        grp_by.append(xplot_col)

    cp_config = config.copy()
    cp_config['group_by'] = grp_by

    ih_analysis = data.copy()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis

    if len(ih_analysis[x_axis].unique()) < 30:
        fig = px.bar(ih_analysis,
                     x=x_axis,
                     y=y_axis,
                     color=xplot_col,
                     facet_col=facet_column,
                     facet_row=facet_row,
                     barmode="group",
                     log_y=xplot_y_bool,
                     title=config['description']
                     )
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_yaxes(tickformat=',.2f')
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=["type", "bar"],
                            label="Bar",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "line"],
                            label="Line",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    showactive=True,
                ),
            ]
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    else:
        fig = px.line(
            ih_analysis,
            x=x_axis,
            y=y_axis,
            color=xplot_col,
            facet_col=facet_column,
            facet_row=facet_row,
            log_y=xplot_y_bool,
            title=config['description']
        )
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_yaxes(dict(tickformat=',.2'))
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            hovermode="x unified",
            autosize=True,
            minreducedheight=height,
            height=height
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis
