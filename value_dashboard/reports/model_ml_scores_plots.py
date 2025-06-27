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
        st.stop()
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

    st.plotly_chart(fig, use_container_width=True)
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

    toggle1, toggle2 = st.columns(2)
    curves_on = toggle1.toggle("Show as curves", value=False, help="Show as curve (ROC or PR).",
                               key="Curves" + config['description'])

    adv_on = toggle2.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                            help="Show advanced reporting options")

    xplot_y_bool = False
    xplot_col = config.get('color', None)
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
    cp_config['x'] = x_axis
    cp_config['group_by'] = grp_by
    cp_config['color'] = xplot_col
    cp_config['facet_row'] = facet_row
    cp_config['facet_column'] = facet_column
    cp_config['log_y'] = xplot_y_bool

    ih_analysis = pd.DataFrame()
    if curves_on:
        cp_config = config.copy()
        cp_config['group_by'] = ([facet_row] if facet_row is not None else []) + (
            [facet_column] if facet_column is not None else []) + (
                                    [xplot_col] if xplot_col is not None else [])
        if not cp_config['group_by']:
            cp_config['group_by'] = None
        report_data = calculate_model_ml_scores(data, cp_config, False)
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
            y=-0.05,
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

        st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis
