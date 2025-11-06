import math

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from value_dashboard.reports.repdata import calculate_engagement_scores
from value_dashboard.reports.shared_plot_utils import *


@timed
def engagement_ctr_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                             config: dict, options_panel: bool = True) -> pd.DataFrame:
    """
    Render CTR as a bar/line plot with optional confidence intervals and return the analysis DataFrame.

    The function aggregates input data using ``calculate_reports_data`` (with optional
    pre-filtering), then renders either a grouped bar chart (for fewer than 25 x-axis
    categories) with 95% confidence intervals or a line chart otherwise. It supports
    row/column faceting, color grouping, and an optional options panel to show metric
    summary cards and advanced plot controls.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Source dataset compatible with the reporting utilities and containing the
        fields referenced by ``config``.
    config : dict
        Plot and data configuration. Expected keys include:
            - ``'x'`` (str): Field for the x-axis (time or category).
            - ``'y'`` (str): Metric to plot (e.g., CTR).
            - ``'color'`` (str, optional): Field for color grouping.
            - ``'description'`` (str): Chart title.
            - ``'facet_row'`` (str, optional): Row facet field.
            - ``'facet_column'`` (str, optional): Column facet field.
            - ``'height'`` (int, optional): Base plot height in pixels (default 640).
            - ``'group_by'`` (list[str], optional): Grouping passed to reporting.
            - Any other keys required by ``calculate_reports_data``.
    options_panel : bool, default True
        If ``True``, displays:
          * A toggle for metric total cards via ``engagement_ctr_cards_subplot``.
          * Advanced options (x/color/facets/log-y) via ``get_plot_parameters_menu``.

    Returns
    -------
    pd.DataFrame
        The processed Pandas DataFrame used for plotting. If the result is empty,
        a Streamlit warning is shown and the empty DataFrame is returned.

    Notes
    -----
    - When number of unique ``x`` values < 25, a grouped bar plot with 95% CI is used;
      otherwise a line plot is rendered.
    - Confidence interval is computed as ``ConfInterval = StdErr * 1.96``.
    - Y-axis tick formatting is set to percentage (``',.2%'``).
    - Plot height scales with the number of row facet categories.
    - The function renders output via ``st.plotly_chart`` and may show warnings.

    Raises
    ------
    KeyError
        If required keys (e.g., ``'x'``, ``'y'``, ``'description'``) are missing in ``config``.
    Exception
        Propagated from downstream utilities such as ``calculate_reports_data`` and
        ``filter_dataframe``.
    """
    xplot_y_bool = False
    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)
    height = config.get('height', 640)

    if options_panel:
        toggle1, toggle2 = st.columns(2)
        cards_on = toggle1.toggle("Metric totals", value=True, key="Metric totals" + config['description'],
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

    grp_by = [config['x']]
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
        engagement_ctr_cards_subplot(ih_analysis, cp_config)

    ih_analysis['ConfInterval'] = ih_analysis['StdErr'] * 1.96
    if len(ih_analysis[x_axis].unique()) < 25:
        fig = px.bar(ih_analysis,
                     x=x_axis,
                     y=y_axis,
                     log_y=xplot_y_bool,
                     color=xplot_col,
                     error_y='ConfInterval',
                     facet_col=facet_column,
                     facet_row=facet_row,
                     barmode="group",
                     title=config['description'] + " with 95% confidence interval",
                     custom_data=[xplot_col, 'ConfInterval']
                     )
        if options_panel:
            fig.update_layout(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                hovermode="x unified",
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
            log_y=xplot_y_bool,
            color=xplot_col,
            title=config['description'],
            facet_col=facet_column,
            facet_row=facet_row,
            custom_data=[xplot_col, 'ConfInterval']
        )
    fig.update_xaxes(tickfont=dict(size=10))
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    yaxis_layout_dict = {yaxis_name + "_tickformat": ',.2%' for yaxis_name in yaxis_names}
    fig.update_layout(yaxis_layout_dict)
    if facet_row:
        height = max(height, 300 * len(ih_analysis[facet_row].unique()))
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=x_axis + ' : %{x}' + '<br>' +
                                          xplot_col + ' : %{customdata[0]}' + '<br>' +
                                          y_axis + ' : %{y:.2%}' + '<br>' +
                                          'CI' + ' : ± %{customdata[1]:.2%}' + '<extra></extra>'
                            )
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def engagement_z_score_plot(data: Union[pl.DataFrame, pd.DataFrame],
                            config: dict) -> pd.DataFrame:
    """
    Render a z-score plot (bar or line) with reference bands and return the analysis DataFrame.

    The function aggregates data using ``calculate_reports_data`` after interactive filtering,
    then renders either a grouped bar chart (for fewer than 25 x-axis categories) or a line
    chart otherwise. It supports row/column faceting, color grouping, and an advanced options
    menu to override axis, color, and log scaling. Horizontal reference bands/lines mark
    the ±1.96 thresholds.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Input dataset compatible with the reporting utilities.
    config : dict
        Plot and data configuration. Expected keys include:
            - ``'x'`` (str): Field for the x-axis.
            - ``'y'`` (str): Z-score metric to plot.
            - ``'color'`` (str): Field for color grouping.
            - ``'description'`` (str): Chart title.
            - ``'facet_row'`` (str, optional): Row facet field.
            - ``'facet_column'`` (str, optional): Column facet field.
            - Any other keys required by ``calculate_reports_data``.

    Returns
    -------
    pd.DataFrame
        The processed Pandas DataFrame used for plotting. If the result is empty,
        a Streamlit warning is shown and the empty DataFrame is returned.

    Notes
    -----
    - Y-axis tick formatting uses a compact numeric format (``',.4'``) for readability.
    - Horizontal shaded region and dashed lines are added at y ∈ {-1.96, 1.96}.
    - Plot height scales with the number of row facet categories.
    - The figure is rendered via ``st.plotly_chart``.

    Raises
    ------
    KeyError
        If required keys (e.g., ``'x'``, ``'y'``, ``'color'``, ``'description'``) are missing.
    Exception
        Propagated from downstream utilities such as ``calculate_reports_data`` and
        ``filter_dataframe``.
    """
    adv_on = st.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)

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
    if len(ih_analysis[x_axis].unique()) < 25:
        fig = px.bar(ih_analysis,
                     x=x_axis,
                     y=y_axis,
                     color=xplot_col,
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
            facet_row=facet_row,
            facet_col=facet_column,
            custom_data=[xplot_col],
            log_y=xplot_y_bool
        )
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    yaxis_layout_dict = {yaxis_name + "_tickformat": ',.4' for yaxis_name in yaxis_names}
    fig.update_layout(yaxis_layout_dict)
    height = 640
    if 'facet_row' in config.keys():
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
    fig.add_hrect(y0=-1.96, y1=1.96, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hline(y=-1.96, line_width=2, line_dash="dash", line_color="darkred")
    fig.add_hline(y=1.96, line_width=2, line_dash="dash", line_color="darkred")
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def engagement_ctr_gauge_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame | None:
    """
    Render CTR gauges by group (with thresholds) and return the analysis DataFrame.

    The function computes CTR per group, lays out one Plotly indicator gauge per group
    in a grid, and optionally colors the gauge bar based on comparison to a provided
    reference threshold per group. The number of rows/columns is derived from the
    number of unique groups (for 1 or 2 grouping columns).

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Input dataset compatible with the reporting utilities.
    config : dict
        Gauge configuration. Expected keys include:
            - ``'group_by'`` (list[str]): One or two grouping columns.
            - ``'value'`` (str): Metric column containing CTR values to display.
            - ``'reference'`` (dict[str, float]): Dictionary mapping group keys (joined by
              ``'_'``) to threshold reference values.
            - ``'description'`` (str): Figure title.
            - Any other keys required by ``calculate_reports_data``.

    Returns
    -------
    pd.DataFrame or None
        The processed Pandas DataFrame used for plotting with technical columns
        removed before returning. If the result is empty, an empty DataFrame is
        returned after a warning. The function may stop execution for invalid
        grouping configuration.

    Notes
    -----
    - Requires at least one grouping column; more than two is not supported.
    - Gauge axis uses percent tick formatting (``',.2%'``); deltas compare to
      the per-group reference value when provided.
    - The figure is rendered via ``st.plotly_chart`` and may show warnings.

    Raises
    ------
    ValueError
        If grouping configuration is invalid (raised via Streamlit stop/warning behavior).
    Exception
        Propagated from downstream utilities such as ``calculate_reports_data`` and
        ``filter_dataframe``.
    """
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
        rows = ih_analysis[grp_by[0]].unique().shape[0]
        cols = ih_analysis[grp_by[1]].unique().shape[0]
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
        gauge = {
            'axis': {'tickformat': ',.2%'},
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': ref_value
            }
        }
        if ref_value:
            if row[config['value']] < ref_value:
                gauge = {
                    'axis': {'tickformat': ',.2%'},
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
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    ih_analysis.drop(columns=['Name', 'CName', 'index'], inplace=True, errors='ignore')
    return ih_analysis


@timed
def engagement_lift_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame:
    """
    Render Lift as a bar/line plot and return the analysis DataFrame.

    The function aggregates data using ``calculate_reports_data`` after interactive filtering,
    then renders either a grouped bar chart (for fewer than 30 x-axis categories) or a line
    chart otherwise. It supports row/column faceting, color grouping, and an advanced options
    menu to override axis, color, and log scaling.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Input dataset compatible with the reporting utilities.
    config : dict
        Plot and data configuration. Expected keys include:
            - ``'x'`` (str): Field for the x-axis.
            - ``'y'`` (str): Metric to plot (e.g., Lift).
            - ``'color'`` (str): Field for color grouping.
            - ``'description'`` (str): Chart title.
            - ``'facet_row'`` (str, optional): Row facet field.
            - ``'facet_column'`` (str, optional): Column facet field.
            - Any other keys required by ``calculate_reports_data``.

    Returns
    -------
    pd.DataFrame
        The processed Pandas DataFrame used for plotting. If the result is empty,
        a Streamlit warning is shown and the empty DataFrame is returned.

    Notes
    -----
    - Y-axis tick formatting is set to percentage for Lift (``',.0%'``).
    - Plot height scales with the number of row facet categories.
    - The figure is rendered via ``st.plotly_chart``.

    Raises
    ------
    KeyError
        If required keys (e.g., ``'x'``, ``'y'``, ``'color'``, ``'description'``) are missing.
    Exception
        Propagated from downstream utilities such as ``calculate_reports_data`` and
        ``filter_dataframe``.
    """
    adv_on = st.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")

    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)

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
            custom_data=[xplot_col],
            title=config['description'],
            facet_row=facet_row,
            facet_col=facet_column,
            log_y=xplot_y_bool
        )
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickformat=',.0%')
    height = 640
    if 'facet_row' in config.keys():
        height = max(640, 300 * len(ih_analysis[config['facet_row']].unique()))
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
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def engagement_ctr_cards_subplot(ih_analysis: Union[pl.DataFrame, pd.DataFrame],
                                 config: dict):
    """
    Render CTR KPI cards (per group) within the Streamlit layout.

    The function computes CTR per group (from Positives/Negatives), calculates the
    weighted average CTR across groups, and renders KPI cards with the CTR and
    delta from the overall average. The cards are arranged into responsive columns
    based on the available main area width.

    Parameters
    ----------
    ih_analysis : pl.DataFrame or pd.DataFrame
        The pre-aggregated analysis dataset containing ``Positives`` and ``Negatives``.
    config : dict
        Configuration dictionary. Expected keys include:
            - ``'group_by'`` (list[str]): One or two columns to group by. If more than
              two are provided, the function will keep the last one (or last two) as needed.

    Returns
    -------
    None
        The function renders KPI cards directly in Streamlit.

    Notes
    -----
    - When number of groups exceeds 18, the grouping is reduced to the last grouping column.
    - CTR is computed as ``Positives / (Positives + Negatives)``.
    - The weighted average CTR uses total clicks as weights.
    - Column count adapts to ``st.session_state['dashboard_dims']['width']`` if available,
      otherwise defaults to a maximum of 8 columns.
    """
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
            (pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives"))).alias("CTR")
        ])
        .sort(grp_by)
    )

    average = data_copy.select(((pl.col("Positives") + pl.col("Negatives")).dot(pl.col("CTR"))) / (
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
    cols = st.columns(num_cols, vertical_alignment='center')
    for index, row in data_copy.iterrows():
        if len(grp_by) > 1:
            kpi_name = row.iloc[0] + "  \n" + row.iloc[1]
        else:
            kpi_name = row.iloc[0]
        cols[index % num_cols].metric(label=kpi_name, value='{:.2%}'.format(row["CTR"]),
                                      delta='{:.2%}'.format(row["CTR"] - average))
        if (index + 1) % num_cols == 0:
            cols = st.columns(num_cols, vertical_alignment='center')


def engagement_rate_card(ih_analysis: Union[pl.DataFrame, pd.DataFrame]):
    """
    Render a Streamlit KPI card for overall Click-through Rate (CTR) with uplift delta.

    The function computes overall CTR and Lift using ``calculate_engagement_scores``,
    and renders a KPI metric showing CTR (as the main value) and uplift vs. random
    action (as the delta), alongside an area-chart sparkline of CTR by month.

    Parameters
    ----------
    ih_analysis : pl.DataFrame or pd.DataFrame
        Input dataset suitable for engagement scoring.

    Returns
    -------
    None
        The metric is rendered directly in Streamlit.

    Notes
    -----
    - Sparkline series is CTR grouped by ``'Month'`` (scaled by 100 for charting).
    - Delta text includes the p-value from Lift (``Lift_P_Val``).
    """
    df = calculate_engagement_scores(ih_analysis, dict())
    data_trend = calculate_engagement_scores(ih_analysis, {'group_by': ['Month']})
    ctr = data_trend['CTR'].round(4) * 100
    st.metric(label="**Click-through Rate**", value='{:.2%}'.format(df["CTR"].item()), border=True,
              delta='{:.2%}'.format(df["Lift"].item()) + ' vs random action',
              help=f'Overall CTR and uplift vs random control group with p_val = {df["Lift_P_Val"].item()}',
              chart_data=ctr, chart_type="area")
