from value_dashboard.metrics.constants import PROPENSITY, FINAL_PROPENSITY
from value_dashboard.reports.repdata import calculate_model_ml_scores
from value_dashboard.reports.shared_plot_utils import *


@timed
def model_ml_scores_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame:
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

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def model_ml_scores_line_plot_roc_pr_curve(data: Union[pl.DataFrame, pd.DataFrame],
                                           config: dict) -> pd.DataFrame:
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

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
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

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
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

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
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

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        ih_analysis = report_data.select(
            (cp_config['group_by'] if cp_config['group_by'] else []) + ['pos_fraction', 'sample_fraction', 'gain',
                                                                        'lift']).to_pandas()
    else:
        ih_analysis = model_ml_scores_line_plot(data, cp_config)
    return ih_analysis


@timed
def model_ml_treemap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                          config: dict) -> pd.DataFrame:
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


def get_plot_parameters_menu_ml(config: dict, is_y_axis_required: bool = True):
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
    config = dict()
    config['metric'] = metric_name
    df = calculate_model_ml_scores(ih_analysis, config, True)
    st.metric(label="**Model AUC**", value='{:.2%}'.format(df["roc_auc"].item()), border=False,
              delta=f"Avg Precision = {'{:.2%}'.format(df['average_precision'].item())}", delta_color='off',
              help=f'Model ROC AUC and Average Precision')


def ml_scores_pers_card(ih_analysis: Union[pl.DataFrame, pd.DataFrame], metric_name: str):
    config = dict()
    config['metric'] = metric_name
    df = calculate_model_ml_scores(ih_analysis, config, True)
    st.metric(label="**Personalization**", value='{:.2}'.format(df["personalization"].item()), border=False,
              delta=f"Novelty = {'{:.2}'.format(df['novelty'].item())}", delta_color='off',
              help=f'Personalization and Novelty')
