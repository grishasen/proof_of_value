from value_dashboard.metrics.constants import PROPENSITY, FINAL_PROPENSITY
from value_dashboard.reports.repdata import calculate_model_ml_scores
from value_dashboard.reports.shared_plot_utils import *


@timed
def model_ml_scores_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame:
    """Render a line plot of ML model scores and return the underlying analysis dataframe.

    This function prepares model evaluation data, builds a Plotly line chart using
    configuration provided in `config`, and renders it in Streamlit. It returns
    the processed analysis DataFrame used for plotting.

    The function expects the upstream data to be compatible with `calculate_reports_data`
    and the configuration to specify which columns and metrics to plot.

    Args:
        data: A Polars or Pandas DataFrame containing the raw input for the report.
              Typical content includes prediction outputs and grouping fields.
        config: A dictionary controlling the plot and data preparation. Expected keys:
            - 'x' (str): Column name for the x-axis.
            - 'y' (str or list[str]): Metric(s) to plot on the y-axis.
            - 'color' (str): Column used to color the lines.
            - 'description' (str): Title/description for the chart.
            - 'facet_row' (str, optional): Column to facet by rows.
            - 'facet_column' (str, optional): Column to facet by columns.
            - 'log_y' (bool, optional): If True, uses logarithmic y-axis. Defaults to False.
            - Any other keys required by `calculate_reports_data`.

    Returns:
        pd.DataFrame: The processed analysis DataFrame (`ih_analysis`) used to build the plot.
                      Returns an empty DataFrame if no data is available.

    Side Effects:
        - Renders a Plotly line chart in Streamlit via `st.plotly_chart`.
        - Shows a Streamlit warning if no data is available.

    Notes:
        - Y-axis tick formatting is set to percentage with two decimals.
        - Hover template shows x, color grouping value, and y as percentage.
        - Chart height adapts to the number of row facets (if any).

    Raises:
        KeyError: If required keys such as 'color' or 'description' are missing from `config`.
        Exception: Propagates exceptions from downstream utilities like `calculate_reports_data`.

    """
    ih_analysis = data.copy()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
    y_axis = config.get('y', None)
    x_axis = config.get('x', None)
    fig = px.line(
        ih_analysis,
        x=x_axis,
        y=y_axis,
        color=config['color'],
        log_y=config.get('log_y', False),
        title=config['description'],
        facet_row=config['facet_row'] if 'facet_row' in config.keys() else None,
        facet_col=config['facet_column'] if 'facet_column' in config.keys() else None,
        custom_data=[config['color']]
    )
    fig.update_xaxes(tickfont=dict(size=10))
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    yaxis_layout_dict = {yaxis_name + "_tickformat": ',.2%' for yaxis_name in yaxis_names}
    fig.update_layout(yaxis_layout_dict)
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
                                          config['color'] + ' : %{customdata[0]}' + '<br>' +
                                          y_axis + ' : %{y:.2%}' + '<extra></extra>')

    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def model_ml_scores_line_plot_roc_pr_curve(data: Union[pl.DataFrame, pd.DataFrame],
                                           config: dict) -> pd.DataFrame:
    """Render ROC/PR/Calibration/Gain/Lift visualizations and return the underlying dataframe.

    Depending on the selected metric (`config['y']`) and the user's pills/toggle choice
    in Streamlit, this function renders one of:
      - ROC curve (when y == 'roc_auc')
      - Precision-Recall curve (when y == 'average_precision')
      - Calibration plot
      - Gain plot
      - Lift plot

    If `config['y']` is neither 'roc_auc' nor 'average_precision', it falls back to the
    generic line plot (`model_ml_scores_line_plot`).

    Args:
        data: Polars or Pandas DataFrame with raw model evaluation input, including
              predicted probabilities, labels, and grouping fields.
        config: Configuration dict controlling data processing and plotting. Common keys:
            - 'metric' (str): Metric key used by `calculate_model_ml_scores`.
            - 'description' (str): Base title/description for the chart.
            - 'x' (str): Default x-axis field (may be overridden by advanced menu).
            - 'y' (str): Target metric; recognized values: 'roc_auc', 'average_precision'.
            - 'color' (str, optional): Grouping field for line color.
            - 'facet_row' (str, optional): Field for row facets.
            - 'facet_column' (str, optional): Field for column facets.
            - 'group_by' (list[str], optional): Passed to scoring function as needed.
            - 'property' (str, optional): One of [PROPENSITY, FINAL_PROPENSITY]; can be changed in UI.

    Returns:
        pd.DataFrame: The processed analysis DataFrame associated with the selected visualization:
            - For ROC/PR/Calibration/Gain/Lift: returns the corresponding exploded/derived DataFrame
              where applicable (e.g., includes 'sample_fraction', 'gain', 'lift' for Gain/Lift).
            - For fallback path: returns the DataFrame from `model_ml_scores_line_plot`.

    Side Effects:
        - Renders the selected Plotly visualization in Streamlit via `st.plotly_chart`.
        - Displays interactive selection widgets (pills/toggle/selectboxes) in the Streamlit UI.

    Notes:
        - Advanced options menu (`get_plot_parameters_menu_ml`) lets users override x/color/faceting/property.
        - ROC plots include the diagonal reference line; PR plots are bounded in [0,1].
        - Gain/Lift plots derive `sample_fraction`, `gain`, and `lift` from TPR/FPR and class prevalence.
        - When no grouping is provided, list-valued columns are collapsed to their first elements before explode.

    Raises:
        KeyError: If required keys such as 'metric' or 'description' are missing from `config`.
        Exception: Propagates exceptions from utility functions (e.g., `calculate_model_ml_scores`).

    """
    y_axis = config.get('y', None)
    if y_axis == "roc_auc":
        x = 'fpr'
        y = 'tpr'
        title = config['description'] + ": ROC Curve"
        label_x = 'False Positive Rate'
        label_y = 'True Positive Rate'
        x0 = 0
        y0 = 0
        x1 = 1
        y1 = 1
    elif y_axis == "average_precision":
        x = 'recall'
        y = 'precision'
        title = config['description'] + ": Precision-Recall Curve Curve"
        label_x = 'Recall'
        label_y = 'Precision'
        x0 = 0
        y0 = 1
        x1 = 1
        y1 = 0
    else:
        ih_analysis = model_ml_scores_line_plot(data, config)
        return ih_analysis

    pills1, toggle1 = st.columns(2)
    options = ["Curves", "Calibration", "Gain", "Lift"]
    selection = pills1.pills("Additional plots", options, selection_mode="single")
    # curves_on = toggle1.toggle("Show as curves", value=False, help="Show as curve (ROC or PR).",
    #                           key="Curves" + config['description'])
    # calibration_on = toggle3.toggle("Calibration plot", value=False, help="Show calibration plot.",
    #                                key="Calibration" + config['description'])
    adv_on = toggle1.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                            help="Show advanced reporting options")

    # if curves_on and calibration_on:
    #    st.warning('Select either curves or calibration.')
    #    st.stop()

    xplot_y_bool = False
    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)
    property = PROPENSITY

    if adv_on:
        plot_menu = get_plot_parameters_menu_ml(config=config, is_y_axis_required=False)
        x_axis = plot_menu['x']
        y_axis = plot_menu['y']
        facet_column = plot_menu['facet_col']
        facet_row = plot_menu['facet_row']
        xplot_col = plot_menu['color']
        property = plot_menu['property']

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

    if not (xplot_col == '---'):
        if not xplot_col in grp_by:
            grp_by.append(xplot_col)
    else:
        xplot_col = None

    cp_config = config.copy()
    cp_config['x'] = x_axis
    cp_config['group_by'] = grp_by
    cp_config['color'] = xplot_col
    cp_config['facet_row'] = facet_row
    cp_config['facet_column'] = facet_column
    cp_config['log_y'] = xplot_y_bool
    cp_config['property'] = property

    ih_analysis = pd.DataFrame()
    if selection == 'Curves':
        report_data = data.copy()
        report_data = filter_dataframe(align_column_types(report_data), case=False)
        cp_config = config.copy()
        cp_config['group_by'] = list(set(([facet_row] if facet_row is not None else []) + (
            [facet_column] if facet_column is not None else []) + (
                                             [xplot_col] if xplot_col is not None else [])))
        if not cp_config['group_by']:
            cp_config['group_by'] = None
        report_data = calculate_model_ml_scores(report_data, cp_config, False)
        if cp_config['group_by'] is None:
            report_data = report_data.with_columns(
                [
                    pl.col(x).list.first().alias(x),
                    pl.col(y).list.first().alias(y)
                ]
            )
        report_data = report_data.explode([x, y])
        fig = px.line(report_data,
                      x=x, y=y,
                      title=title,
                      color=xplot_col,
                      log_y=xplot_y_bool,
                      facet_col=facet_column,
                      facet_row=facet_row,
                      height=len(
                          report_data[facet_row].unique()) * 400 if facet_row is not None else 640
                      )
        if y_axis == "roc_auc":
            fig.add_shape(
                type="line", line=dict(dash='dash', color="darkred"),
                row='all', col='all', x0=x0, y0=y0, x1=x1, y1=y1
            )
        fig.update_layout(
            xaxis=dict(
                range=[0, 1]
            ),
            yaxis=dict(
                range=[0, 1]
            )
        )
        fig.for_each_xaxis(lambda x: x.update({'title': ''}))
        fig.for_each_yaxis(lambda y: y.update({'title': ''}))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=0.5,
            yref='paper',
            y=-0.1,
            text=label_x
        )
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=-0.04,
            yanchor='middle',
            yref='paper',
            y=0.5,
            textangle=90,
            text=label_y
        )

        st.plotly_chart(fig, width='stretch', theme="streamlit")
    elif selection == 'Calibration':
        x = 'calibration_proba'
        y = 'calibration_rate'
        title = config['description'] + ": Calibration Plot"
        label_x = 'Probabilities'
        label_y = 'Positives share'
        x0 = 0
        y0 = 0
        x1 = 1
        y1 = 1

        report_data = data.copy()
        report_data = filter_dataframe(align_column_types(report_data), case=False)
        cp_config = config.copy()
        cp_config['group_by'] = list(set(([facet_row] if facet_row is not None else []) + (
            [facet_column] if facet_column is not None else []) + (
                                             [xplot_col] if xplot_col is not None else [])))
        if not cp_config['group_by']:
            cp_config['group_by'] = None
        report_data = calculate_model_ml_scores(report_data, cp_config, False)
        if cp_config['group_by'] is None:
            report_data = (report_data
                           .with_columns([pl.col(x).list.first().alias(x), pl.col(y).list.first().alias(y)])
                           .sort(x, descending=False))
        report_data = report_data.explode([x, y]).sort(x, descending=False)
        fig = px.line(report_data,
                      x=x, y=y,
                      title=title,
                      color=xplot_col,
                      log_y=xplot_y_bool,
                      facet_col=facet_column,
                      facet_row=facet_row,
                      height=len(
                          report_data[facet_row].unique()) * 400 if facet_row is not None else 640
                      )
        fig.add_shape(
            type="line", line=dict(dash='dash', color="darkred"),
            row='all', col='all', x0=x0, y0=y0, x1=x1, y1=y1
        )
        fig.update_layout(
            xaxis=dict(
                range=[0, 1]
            ),
            yaxis=dict(
                range=[0, 1]
            )
        )
        fig.for_each_xaxis(lambda x: x.update({'title': ''}))
        fig.for_each_yaxis(lambda y: y.update({'title': ''}))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=0.5,
            yref='paper',
            y=-0.1,
            text=label_x
        )
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=-0.04,
            yanchor='middle',
            yref='paper',
            y=0.5,
            textangle=90,
            text=label_y
        )

        st.plotly_chart(fig, width='stretch', theme="streamlit")
    elif selection == 'Gain':
        x = 'sample_fraction'
        y = 'gain'
        title = config['description'] + ": Gain Plot"
        label_x = 'Fraction of population'
        label_y = 'Gain'
        x0 = 0
        y0 = 0
        x1 = 1
        y1 = 1

        report_data = data.copy()
        report_data = filter_dataframe(align_column_types(report_data), case=False)
        cp_config = config.copy()
        cp_config['group_by'] = list(set(([facet_row] if facet_row is not None else []) + (
            [facet_column] if facet_column is not None else []) + (
                                             [xplot_col] if xplot_col is not None else [])))
        if not cp_config['group_by']:
            cp_config['group_by'] = None
        report_data = calculate_model_ml_scores(report_data, cp_config, False)
        if cp_config['group_by'] is None:
            report_data = (report_data
                           .with_columns([pl.col(x).list.first().alias(x), pl.col(y).list.first().alias(y)])
                           .sort(x, descending=False))
        list_cols = ["tpr", "fpr"]
        report_data = (
            report_data.explode(list_cols)
            .drop("calibration_rate", strict=False)
            .with_columns([
                (pl.col("pos_fraction") * pl.col("tpr") + (1.0 - pl.col("pos_fraction")) * pl.col("fpr"))
                .alias("sample_fraction"),
                pl.col("tpr").alias("gain"),
            ])
        )
        fig = px.line(report_data,
                      x=x, y=y,
                      title=title,
                      color=xplot_col,
                      facet_col=facet_column,
                      facet_row=facet_row,
                      height=len(
                          report_data[facet_row].unique()) * 400 if facet_row is not None else 640
                      )
        fig.add_shape(
            type="line", line=dict(dash='dash', color="darkred"),
            row='all', col='all', x0=x0, y0=y0, x1=x1, y1=y1
        )
        fig.update_layout(
            xaxis=dict(
                range=[0, 1]
            ),
            yaxis=dict(
                range=[0, 1]
            )
        )
        fig.for_each_xaxis(lambda x: x.update({'title': ''}))
        fig.for_each_yaxis(lambda y: y.update({'title': ''}))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=0.5,
            yref='paper',
            y=-0.1,
            text=label_x
        )
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=-0.03,
            yanchor='middle',
            yref='paper',
            y=0.5,
            textangle=90,
            text=label_y
        )

        st.plotly_chart(fig, width='stretch', theme="streamlit")
        ih_analysis = report_data.select(
            (cp_config['group_by'] if cp_config['group_by'] else []) + ['pos_fraction', 'sample_fraction',
                                                                        'gain']).to_pandas()
    elif selection == 'Lift':
        x = 'sample_fraction'
        y = 'lift'
        title = config['description'] + ": Lift Plot"
        label_x = 'Fraction of population'
        label_y = 'Lift'
        x0 = 0
        y0 = 1
        x1 = 1
        y1 = 1

        report_data = data.copy()
        report_data = filter_dataframe(align_column_types(report_data), case=False)
        cp_config = config.copy()
        cp_config['group_by'] = list(set(([facet_row] if facet_row is not None else []) + (
            [facet_column] if facet_column is not None else []) + (
                                             [xplot_col] if xplot_col is not None else [])))
        if not cp_config['group_by']:
            cp_config['group_by'] = None
        report_data = calculate_model_ml_scores(report_data, cp_config, False)
        if cp_config['group_by'] is None:
            report_data = (report_data
                           .with_columns([pl.col(x).list.first().alias(x), pl.col(y).list.first().alias(y)])
                           .sort(x, descending=False))
        list_cols = ["tpr", "fpr"]
        report_data = (
            report_data.explode(list_cols)
            .drop("calibration_rate", strict=False)
            .with_columns([
                (pl.col("pos_fraction") * pl.col("tpr") + (1.0 - pl.col("pos_fraction")) * pl.col("fpr"))
                .alias("sample_fraction"),
                pl.col("tpr").alias("gain"),
                (pl.col("tpr") / (
                        pl.col("pos_fraction") * pl.col("tpr") + (1.0 - pl.col("pos_fraction")) * pl.col("fpr")))
                .alias("lift"),
            ])
            .with_columns([
                pl.when(pl.col("sample_fraction") > 0.000001)
                .then(pl.col("gain") / pl.col("sample_fraction"))
                .otherwise(0)
                .alias("lift")
            ])
        )
        fig = px.line(report_data,
                      x=x, y=y,
                      title=title,
                      color=xplot_col,
                      facet_col=facet_column,
                      facet_row=facet_row,
                      height=len(
                          report_data[facet_row].unique()) * 400 if facet_row is not None else 640
                      )
        fig.add_shape(
            type="line", line=dict(dash='dash', color="darkred"),
            row='all', col='all', x0=x0, y0=y0, x1=x1, y1=y1
        )
        fig.update_layout(
            xaxis=dict(
                range=[0, 1]
            ),
            yaxis=dict(
                range=[0, report_data[y].max()]
            )
        )
        fig.for_each_xaxis(lambda x: x.update({'title': ''}))
        fig.for_each_yaxis(lambda y: y.update({'title': ''}))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=0.5,
            yref='paper',
            y=-0.1,
            text=label_x
        )
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=-0.03,
            yanchor='middle',
            yref='paper',
            y=0.5,
            textangle=90,
            text=label_y
        )

        st.plotly_chart(fig, width='stretch', theme="streamlit")
        ih_analysis = report_data.select(
            (cp_config['group_by'] if cp_config['group_by'] else []) + ['pos_fraction', 'sample_fraction', 'gain',
                                                                        'lift']).to_pandas()
    else:
        ih_analysis = model_ml_scores_line_plot(data, cp_config)
    return ih_analysis


@timed
def model_ml_treemap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                          config: dict) -> pd.DataFrame:
    """Render a treemap of counts by grouped dimensions and return the analysis dataframe.

    This function aggregates report data based on `config['group_by']`, builds a
    Plotly treemap colored by `config['color']`, and renders it in Streamlit.
    It returns the filtered dataframe used to build the treemap.

    Args:
        data: Polars or Pandas DataFrame with the raw input suitable for reports.
        config: Configuration dict. Expected keys:
            - 'group_by' (list[str]): Ordered dimensions to form the treemap path.
            - 'color' (str): Column used for color scale.
            - 'description' (str): Title/description for the treemap.
            - Any other keys required by `calculate_reports_data`.

    Returns:
        pd.DataFrame: The filtered Pandas DataFrame used in the treemap. If empty,
                      a Streamlit warning is shown and execution stops.

    Side Effects:
        - Renders a Plotly treemap in Streamlit via `st.plotly_chart`.
        - Calls `st.warning` and `st.stop()` if no data is available.

    Notes:
        - The root node is labeled "ALL".
        - Text info displays label, raw value, percent of parent, and percent of root.
        - The treemap height is fixed at 640px and uses a diverging color scale (RdBu_r).

    Raises:
        KeyError: If required configuration keys like 'group_by', 'color', or 'description' are missing.
        Exception: Propagates exceptions from `calculate_reports_data`.

    """
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    fig = px.treemap(ih_analysis, path=[px.Constant("ALL")] + config['group_by'], values='Count',
                     color=config['color'],
                     color_continuous_scale=px.colors.sequential.RdBu_r,
                     title=config['description'],
                     height=640,
                     )
    fig.update_traces(textinfo="label+value+percent parent+percent root")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


def get_plot_parameters_menu_ml(config: dict, is_y_axis_required: bool = True):
    """Render a Streamlit menu to select ML plot parameters and return the selection.

    This helper exposes interactive controls for selecting plot axes, faceting, color,
    and property used in ML score visualizations. It reads defaults from `config`,
    metric metadata from `get_config()["metrics"][metric]`, and returns the user's choices.

    Args:
        config: Configuration dictionary. Expected keys:
            - 'metric' (str): The metric key to fetch metadata (group_by, scores).
            - 'x' (str, optional): Default x-axis column.
            - 'y' (str, optional): Default y-axis (metric) when `is_y_axis_required` is True.
            - 'color' (str, optional): Default color-by column.
            - 'facet_row' (str, optional): Default row facet column.
            - 'facet_column' (str, optional): Default column facet column.
        is_y_axis_required: If True, shows the Y-Axis metric selector; otherwise omitted.

    Returns:
        dict: A dictionary with the selected parameters:
            - 'x' (str): Selected x-axis column.
            - 'color' (str): Selected color-by column.
            - 'facet_row' (str): Selected row facet column or '---' for none.
            - 'facet_col' (str): Selected column facet or '---' for none.
            - 'y' (str or None): Selected y metric if required; otherwise None.
            - 'property' (str): Selected property (PROPENSITY or FINAL_PROPENSITY).

    Side Effects:
        - Renders several Streamlit selectbox widgets for interactive selection.

    Notes:
        - Available options for axes/facets are determined by the metric's `group_by` and `scores`
          configured in `get_config()`.
        - If defaults are not present in `config`, the first available option is selected by default.

    Raises:
        KeyError: If `config['metric']` is missing or if the metric metadata cannot be found.

    """
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = m_config['group_by']
    report_grp_by = sorted(report_grp_by)
    scores = m_config['scores']

    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)
    property = PROPENSITY

    cols = st.columns(6 if is_y_axis_required else 5)
    with cols[0]:
        x_axis = st.selectbox(
            label='X-Axis',
            options=report_grp_by,
            index=report_grp_by.index(x_axis) if x_axis else 0,
            help="Select X-Axis."
        )
    with cols[1]:
        xplot_col = st.selectbox(
            label='Colour By',
            options=report_grp_by,
            index=report_grp_by.index(xplot_col) if xplot_col else 0,
            help="Select color."
        )
    with cols[2]:
        options_row = ['---'] + report_grp_by
        if 'facet_row' in config.keys():
            facet_row = st.selectbox(
                label='Row Facets',
                options=options_row,
                index=options_row.index(config['facet_row']),
                help="Select data column."
            )
        else:
            facet_row = st.selectbox(
                label='Row Facets',
                options=options_row,
                help="Select data column."
            )
    with cols[3]:
        options_col = ['---'] + report_grp_by
        if 'facet_column' in config.keys():
            facet_column = st.selectbox(
                label='Column Facets',
                options=options_col,
                index=options_col.index(config['facet_column']),
                help="Select data column."
            )
        else:
            facet_column = st.selectbox(
                label='Column Facets',
                options=options_col,
                help="Select data column."
            )
    if is_y_axis_required:
        with cols[4]:
            y_axis = st.selectbox(
                label='Y-Axis',
                options=scores,
                index=scores.index(y_axis) if y_axis else 0,
                help="Select Y-Axis."
            )
    with cols[len(cols) - 1]:
        property = st.selectbox(
            label='Property',
            options=[PROPENSITY, FINAL_PROPENSITY],
            help="Select Property."
        )

    return {'x': x_axis, 'color': xplot_col, 'facet_row': facet_row, 'facet_col': facet_column, 'y': y_axis,
            'property': property}


def ml_scores_card(ih_analysis: Union[pl.DataFrame, pd.DataFrame], metric_name: str):
    """Render a Streamlit KPI card showing model ROC AUC and Average Precision trend.

    Computes summary metrics using `calculate_model_ml_scores` and displays them as a
    Streamlit metric card, with an area chart trend over time (by `Month`).

    Args:
        ih_analysis: Polars or Pandas DataFrame containing input for model scoring.
        metric_name: Metric key to evaluate (must be defined in `get_config()["metrics"]`).

    Returns:
        None

    Side Effects:
        - Renders a Streamlit KPI metric with value, delta, and a small area chart.

    Notes:
        - The displayed value is ROC AUC formatted as a percentage.
        - The delta shows Average Precision (AP) formatted as percentage.
        - The sparkline uses `Month`-grouped ROC AUC multiplied by 100 (as expected by `st.metric` chart).

    Raises:
        Exception: Propagates from `calculate_model_ml_scores` if configuration or data is invalid.

    """
    config = dict()
    config['metric'] = metric_name
    df = calculate_model_ml_scores(ih_analysis, config, True)
    data_trend = calculate_model_ml_scores(ih_analysis, {'metric': metric_name, 'group_by': ['Month']}, True)
    auc = data_trend['roc_auc'].round(4) * 100
    st.metric(label="**Model AUC**", value='{:.2%}'.format(df["roc_auc"].item()), border=True,
              delta=f"Avg Precision = {'{:.2%}'.format(df['average_precision'].item())}", delta_color='normal',
              help=f'Model ROC AUC and Average Precision', chart_data=auc, chart_type="area")


def ml_scores_pers_card(ih_analysis: Union[pl.DataFrame, pd.DataFrame], metric_name: str):
    """Render a Streamlit KPI card showing Personalization and Novelty metrics with trend.

    Computes personalization-related metrics and displays them as a Streamlit metric card,
    including an area chart trend by `Month`.

    Args:
        ih_analysis: Polars or Pandas DataFrame containing input for personalization scoring.
        metric_name: Metric key to evaluate (must be defined in `get_config()["metrics"]`).

    Returns:
        None

    Side Effects:
        - Renders a Streamlit KPI metric for Personalization with Novelty as the delta.

    Notes:
        - The main value displays `personalization` (rounded to two decimals).
        - The delta shows `novelty`.
        - The sparkline uses `Month`-grouped `personalization`.

    Raises:
        Exception: Propagates from `calculate_model_ml_scores` if configuration or data is invalid.

    """
    config = dict()
    config['metric'] = metric_name
    df = calculate_model_ml_scores(ih_analysis, config, True)
    data_trend = calculate_model_ml_scores(ih_analysis, {'metric': metric_name, 'group_by': ['Month']}, True)
    auc = data_trend['personalization'].round(2)
    st.metric(label="**Personalization**", value='{:.2}'.format(df["personalization"].item()), border=True,
              delta=f"Novelty = {'{:.2}'.format(df['novelty'].item())}", delta_color='off',
              help=f'Personalization and Novelty', chart_data=auc, chart_type="area")
