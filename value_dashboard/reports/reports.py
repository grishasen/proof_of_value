import math
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import polars as pl
import streamlit as st
from lifetimes import BetaGeoFitter, ParetoNBDFitter
from lifetimes import GammaGammaFitter
from plotly.subplots import make_subplots
from polars_ds import sample_and_split as pds_sample

from value_dashboard.metrics.constants import CUSTOMER_ID
from value_dashboard.reports.repdata import calculate_reports_data, calculate_model_ml_scores
from value_dashboard.utils.config import get_config
from value_dashboard.utils.st_utils import filter_dataframe, align_column_types
from value_dashboard.utils.string_utils import strtobool
from value_dashboard.utils.timer import timed


@timed
def model_ml_scores_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame:
    ih_analysis = data.copy()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    fig = px.line(
        ih_analysis,
        x=config['x'],
        y=config['y'],
        color=config['color'],
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
        xaxis_title=config['x'],
        yaxis_title=config['y'],
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=config['x'] + ' : %{x}' + '<br>' +
                                          config['color'] + ' : %{customdata[0]}' + '<br>' +
                                          config['y'] + ' : %{y:.2%}' + '<extra></extra>')

    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def model_ml_scores_line_plot_roc_pr_curve(data: Union[pl.DataFrame, pd.DataFrame],
                                           config: dict) -> pd.DataFrame:
    if config['y'] == "roc_auc":
        x = 'fpr'
        y = 'tpr'
        title = config['description'] + ": ROC Curve"
        label_x = 'False Positive Rate'
        label_y = 'True Positive Rate'
        x0 = 0
        y0 = 0
        x1 = 1
        y1 = 1
    elif config['y'] == "average_precision":
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

    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = sorted(list(set(report_grp_by)))

    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3 = st.columns(4)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c2:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c3:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    help="Select data column."
                )

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
    cp_config['x'] = config['x']
    cp_config['group_by'] = grp_by
    cp_config['color'] = xplot_col
    cp_config['facet_row'] = facet_row
    cp_config['facet_column'] = facet_column

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
                      facet_col=facet_column,
                      facet_row=facet_row,
                      height=len(
                          report_data[facet_row].unique()) * 400 if facet_row is not None else 640
                      )
        if config['y'] == "roc_auc":
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
def engagement_ctr_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                             config: dict) -> pd.DataFrame:
    toggle1, toggle2 = st.columns(2)
    cards_on = toggle1.toggle("Metric totals", value=True, key="Metric totals" + config['description'],
                              help="Show aggregated metric values with difference from mean")

    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = sorted(list(set(report_grp_by)))

    adv_on = toggle2.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                            help="Show advanced reporting options")

    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c2:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c3:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    help="Select data column."
                )
        with c4:
            xplot_y_log = st.radio(
                'Y Axis scale',
                ('Linear', 'Logarithmic'),
                horizontal=True,
                help="Select axis scale.",
                # label_visibility='collapsed'
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

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
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    if cards_on:
        engagement_ctr_cards_subplot(ih_analysis, cp_config)

    ih_analysis['ConfInterval'] = ih_analysis['StdErr'] * 1.96
    if len(ih_analysis[config['x']].unique()) < 25:
        fig = px.bar(ih_analysis,
                     x=config['x'],
                     y=config['y'],
                     log_y=xplot_y_bool,
                     color=xplot_col,
                     error_y='ConfInterval',
                     facet_col=facet_column,
                     facet_row=facet_row,
                     barmode="group",
                     title=config['description'] + " with 95% confidence interval",
                     # category_orders={
                     #    xplot_col: sorted(ih_analysis[xplot_col].unique(), reverse=True)},
                     custom_data=[xplot_col, 'ConfInterval']
                     )
        fig.update_layout(
            xaxis_title=config['x'],
            yaxis_title=config['y'],
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
            x=config['x'],
            y=config['y'],
            log_y=xplot_y_bool,
            color=xplot_col,
            title=config['description'],
            facet_col=facet_column,
            facet_row=facet_row,
            custom_data=[xplot_col, 'ConfInterval'],
        )
    fig.update_xaxes(tickfont=dict(size=10))
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    yaxis_layout_dict = {yaxis_name + "_tickformat": ',.2%' for yaxis_name in yaxis_names}
    fig.update_layout(yaxis_layout_dict)
    height = 640
    if facet_row:
        height = max(640, 300 * len(ih_analysis[facet_row].unique()))

    fig.update_layout(
        xaxis_title=config['x'],
        yaxis_title=config['y'],
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=config['x'] + ' : %{x}' + '<br>' +
                                          xplot_col + ' : %{customdata[0]}' + '<br>' +
                                          config['y'] + ' : %{y:.2%}' + '<br>' +
                                          'CI' + ' : ± %{customdata[1]:.2%}' + '<extra></extra>'
                            )
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def engagement_z_score_plot(data: Union[pl.DataFrame, pd.DataFrame],
                            config: dict) -> pd.DataFrame:
    adv_on = st.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    report_grp_by = config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = list(set(report_grp_by))
    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c2:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c3:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    help="Select data column."
                )
        with c4:
            xplot_y_log = st.radio(
                'Y Axis scale',
                ('Linear', 'Logarithmic'),
                horizontal=True,
                help="Select axis scale.",
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

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
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    if len(ih_analysis[config['x']].unique()) < 25:
        fig = px.bar(ih_analysis,
                     x=config['x'],
                     y=config['y'],
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
            x=config['x'],
            y=config['y'],
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
        xaxis_title=config['x'],
        yaxis_title=config['y'],
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=config['x'] + ' : %{x}' + '<br>' +
                                          xplot_col + ' : %{customdata[0]}' + '<br>' +
                                          config['y'] + ' : %{y:.2%}' + '<extra></extra>')
    fig.add_hrect(y0=-1.96, y1=1.96, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hline(y=-1.96, line_width=2, line_dash="dash", line_color="darkred")
    fig.add_hline(y=1.96, line_width=2, line_dash="dash", line_color="darkred")
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def engagement_ctr_gauge_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame | None:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
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
    st.plotly_chart(fig, use_container_width=True)
    ih_analysis.drop(columns=['Name', 'CName', 'index'], inplace=True, errors='ignore')
    return ih_analysis


@timed
def conversion_rate_gauge_plot(data: Union[pl.DataFrame, pd.DataFrame],
                               config: dict) -> pd.DatetimeIndex | None:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
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
    st.plotly_chart(fig, use_container_width=True)
    ih_analysis.drop(columns=['Name', 'CName', 'index'], inplace=True, errors='ignore')
    return ih_analysis


@timed
def descriptive_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                          config: dict) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    scores = m_config["scores"]
    columns = m_config["columns"]
    columns_conf = m_config['columns']
    num_columns = [col for col in columns_conf if (col + '_Mean') in data.columns]

    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = list(set(report_grp_by))

    title = config['description']
    y = config['y']
    if y in data.columns:
        color = y
    elif 'color' in config.keys():
        color = config['color']
    else:
        color = config['facet_row']

    adv_on = st.toggle("Advanced options", value=True, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    xplot_y_bool = False
    option = config['score']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4, c5, c6 = st.columns(7)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            config['y'] = st.selectbox(
                label="## Select data property ",
                options=columns,
                index=columns.index(config['y']),
                label_visibility='visible',
                help="Select score to visualize."
            )
        with c2:
            opts = ['Count']
            for sc in scores:
                if config['y'] in num_columns:
                    opts.append(sc)
            option = st.selectbox(
                label="**" + config['y'] + "** score",
                options=opts,
                index=opts.index(config['score']) if config['score'] in opts else 0,
                # label_visibility='collapsed',
                help="Select score to visualize."
            )
        with c3:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c4:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label='Plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label='Plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c5:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label='Plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label='Plot columns',
                    options=options_col,
                    help="Select data column."
                )
        with c6:
            xplot_y_log = st.radio(
                'Y-Axis scale',
                ('Linear', 'Logarithmic'),
                horizontal=True,
                help="Select axis scale.",
                # label_visibility='collapsed'
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

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

    report_data = calculate_reports_data(data, cp_config)
    ih_analysis = report_data.to_pandas()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    if facet_row:
        height = max(640, 350 * len(ih_analysis[facet_row].unique()))
    else:
        height = 640

    def select(arg):
        if arg.title.text == config['y'] + "_" + option:
            return True
        return False

    if len(ih_analysis[config['x']].unique()) < 30:
        fig = px.bar(data_frame=ih_analysis,
                     x=config['x'],
                     y=config['y'] + "_" + option,
                     color=xplot_col,
                     facet_col=facet_column,
                     facet_row=facet_row,
                     barmode="group",
                     title=title,
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
            x=config['x'],
            y=config['y'] + "_" + option,
            color=xplot_col,
            title=title,
            facet_row=facet_row,
            facet_col=facet_column,
            log_y=xplot_y_bool
        )
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(title=option, selector=select)
    fig.update_layout(
        hovermode="x unified",
        autosize=True,
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def descriptive_box_plot(data: Union[pl.DataFrame, pd.DataFrame],
                         config: dict) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    columns_conf = m_config['columns']
    num_columns = [col for col in columns_conf if (col + '_Mean') in data.columns]

    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = list(set(report_grp_by))

    title = config['description']
    y = config['y']
    if y in data.columns:
        color = y
    elif 'color' in config.keys():
        color = config['color']
    else:
        color = config['facet_row']

    adv_on = st.toggle("Advanced options", value=True, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            config['y'] = st.selectbox(
                label="## Select data property ",
                options=num_columns,
                index=num_columns.index(config['y']),
                label_visibility='visible',
                help="Select score to visualize."
            )
        with c2:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c3:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label='Plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label='Plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c4:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label='Plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label='Plot columns',
                    options=options_col,
                    help="Select data column."
                )

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

    report_data = calculate_reports_data(data, cp_config)
    ih_analysis = report_data.to_pandas()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)

    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    if facet_row:
        height = max(640, 350 * len(ih_analysis[facet_row].unique()))
    else:
        height = 640

    num_rows = 1
    if facet_row:
        categories_row = ih_analysis[facet_row].unique()
        num_rows = len(categories_row)
    else:
        categories_row = ['']

    num_cols = 1
    if facet_column:
        categories_col = ih_analysis[facet_column].unique()
        num_cols = len(categories_col)
    else:
        categories_col = ['']

    fig = make_subplots(rows=num_rows,
                        cols=num_cols,
                        shared_xaxes=True,
                        shared_yaxes=False,
                        vertical_spacing=0.05
                        )

    row_col_map = {(row, col): (i + 1, j + 1) for i, row in enumerate(categories_row) for j, col in
                   enumerate(categories_col)}

    theme_colors = pio.templates[pio.templates.default].layout.colorway
    colors = sorted(list(ih_analysis[xplot_col].unique()))
    x_items = sorted(list(ih_analysis[config['x']].unique()))

    for item in ih_analysis.to_dict('records'):
        color_index = colors.index(item[xplot_col])
        row = item[facet_row] if facet_row else ''
        col = item[facet_column] if facet_column else ''
        x_value = item[config['x']]
        x_index = x_items.index(item[config['x']])
        color = item[xplot_col]
        q1 = item[config['y'] + '_p25']
        median = item[config['y'] + '_Median']
        q3 = item[config['y'] + '_p75']
        mean = item[config['y'] + '_Mean']
        sd = item[config['y'] + '_Std']
        lowerfence = item[config['y'] + '_p25'] - 1.5 * (item[config['y'] + '_p75'] - item[config['y'] + '_p25'])
        lowerfence1 = item[config['y'] + '_Min']
        if lowerfence1 > lowerfence:
            lowerfence = lowerfence1

        notchspan = (1.57 * (
                (item[config['y'] + '_p75'] - item[config['y'] + '_p25']) / (item[config['y'] + '_Count'] ** 0.5)))

        upperfence = (item[config['y'] + '_p75'] + 1.5 * (item[config['y'] + '_p75'] - item[config['y'] + '_p25']))
        upperfence1 = item[config['y'] + '_Max']
        if upperfence1 < upperfence:
            upperfence = upperfence1

        subplot_row, subplot_col = row_col_map[(row, col)]
        fig.add_trace(
            go.Box(
                q1=[q1],
                median=[median],
                q3=[q3],
                name=color,
                x=[x_value],
                mean=[mean],
                lowerfence=[lowerfence],
                notchspan=[notchspan],
                upperfence=[upperfence],
                marker_color=theme_colors[color_index % len(theme_colors)],
                offsetgroup=color,
                boxpoints=False,
                showlegend=((subplot_row == 1) and (subplot_col == 1) and (x_index == 0))
            ),
            row=subplot_row,
            col=subplot_col
        )

    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_layout(
        boxmode='group',
        height=height,
        title_text=title
    )
    for i, row in enumerate(categories_row):
        delta = 1 / (2 * len(categories_row))
        fig.add_annotation(
            dict(
                text=f"{row}",
                xref="paper",
                yref="paper",
                x=1.02, y=(1 - ((i + 1) / len(categories_row)) + delta),
                showarrow=False,
                font=dict(size=14),
                xanchor="right",
                yanchor="middle",
                textangle=90
            )
        )
    for j, col in enumerate(categories_col):
        fig.add_annotation(
            dict(
                text=f"{col}",
                xref="paper", yref="paper",
                x=(j / len(categories_col) + 0.5 / len(categories_col)), y=1.0,
                showarrow=False,
                font=dict(size=14),
                xanchor="center", yanchor="bottom"
            )
        )
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def descriptive_funnel(data: Union[pl.DataFrame, pd.DataFrame],
                       config: dict) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = config['group_by']
    title = config['description']
    x = config['x']
    color = config['color']
    stages = config['stages']
    facet_row = None if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = None if not 'facet_column' in config.keys() else config['facet_column']
    copy_config = config.copy()
    copy_config['group_by'] = report_grp_by + [x]
    report_data = calculate_reports_data(data, copy_config)
    for stage in stages:
        if report_data.filter(pl.col(x).is_in([stage])).height == 0:
            stages.remove(stage)

    report_data = (
        report_data
        .filter(pl.col(x).is_in(stages))
        .group_by(report_grp_by + [x])
        .agg(pl.col(x + "_Count").sum())
        .pivot(x, index=report_grp_by, values=x + "_Count")
        .sort(report_grp_by)
        .unpivot(stages, index=report_grp_by)
        .with_columns(pl.col("value").fill_null(0.0))
        .rename({"variable": "Stage"})
        .rename({"value": "Count"})
    )
    ih_analysis = report_data.to_pandas()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    if facet_row:
        height = max(640, 350 * len(ih_analysis[facet_row].unique()))
    else:
        height = 640

    fig = px.funnel(ih_analysis,
                    x='Count',
                    y='Stage',
                    color=color,
                    facet_row=facet_row,
                    facet_col=facet_column,
                    title=title,
                    height=height,
                    category_orders={
                        config['color']: ih_analysis.sort_values("Count", axis=0, ascending=False)[
                            config['color']].unique()
                    }
                    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def engagement_lift_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = sorted(list(set(report_grp_by)))
    adv_on = st.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")

    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c2:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c3:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    help="Select data column."
                )
        with c4:
            xplot_y_log = st.radio(
                'Y Axis scale',
                ('Linear', 'Logarithmic'),
                horizontal=True,
                help="Select axis scale.",
                # label_visibility='collapsed'
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

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
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    if len(ih_analysis[config['x']].unique()) < 30:
        fig = px.bar(ih_analysis,
                     x=config['x'],
                     y=config['y'],
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
            x=config['x'],
            y=config['y'],
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
        xaxis_title=config['x'],
        yaxis_title=config['y'],
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=config['x'] + ' : %{x}' + '<br>' +
                                          xplot_col + ' : %{customdata[0]}' + '<br>' +
                                          config['y'] + ' : %{y:.2%}' + '<extra></extra>')
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def conversion_rate_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                              config: dict) -> pd.DataFrame:
    report_grp_by = config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = list(set(report_grp_by))
    toggle1, toggle2 = st.columns(2)
    cards_on = toggle1.toggle("Metric totals", value=True, key="Metrics" + config['description'],
                              help="Show aggregated metric values with difference from mean")
    adv_on = toggle2.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                            help="Show advanced reporting options")
    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c2:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c3:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    help="Select data column."
                )
        with c4:
            xplot_y_log = st.radio(
                'Y Axis scale',
                ('Linear', 'Logarithmic'),
                horizontal=True,
                help="Select axis scale.",
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

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
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    if cards_on:
        conversion_rate_cards_subplot(ih_analysis, cp_config)

    if len(ih_analysis[config['x']].unique()) < 30:
        fig = px.bar(ih_analysis,
                     x=config['x'],
                     y=config['y'],
                     color=xplot_col,
                     error_y='StdErr',
                     log_y=xplot_y_bool,
                     facet_col=facet_column,
                     facet_row=facet_row,
                     barmode="group",
                     title=config['description'],
                     # category_orders={xplot_col: sorted(ih_analysis[xplot_col].unique())},
                     custom_data=[xplot_col]
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
            x=config['x'],
            y=config['y'],
            color=xplot_col,
            title=config['description'],
            log_y=xplot_y_bool,
            facet_col=facet_column,
            facet_row=facet_row,
            custom_data=[xplot_col]
        )

    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickformat=',.2%')
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    yaxis_layout_dict = {yaxis_name + "_tickformat": ',.2%' for yaxis_name in yaxis_names}
    fig.update_layout(yaxis_layout_dict)
    height = 640
    if facet_row:
        height = max(640, 300 * len(ih_analysis[facet_row].unique()))
    fig.update_layout(
        xaxis_title=config['x'],
        yaxis_title=config['y'],
        hovermode="x unified",
        height=height
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig = fig.update_traces(hovertemplate=config['x'] + ' : %{x}' + '<br>' +
                                          xplot_col + ' : %{customdata[0]}' + '<br>' +
                                          config['y'] + ' : %{y:.2%}' + '<extra></extra>')
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def conversion_revenue_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                                 config: dict) -> pd.DataFrame:
    adv_on = st.toggle("Advanced options", value=False, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    report_grp_by = config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = list(set(report_grp_by))
    xplot_y_bool = False
    color = config['color']
    xplot_col = color
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            config['x'] = st.selectbox(
                label='X-Axis',
                options=report_grp_by,
                index=report_grp_by.index(config['x']),
                help="Select X-Axis."
            )
        with c1:
            xplot_col = st.selectbox(
                label='Colour By',
                options=report_grp_by,
                index=report_grp_by.index(color),
                help="Select color."
            )
        with c2:
            options_row = ['---'] + report_grp_by
            if 'facet_row' in config.keys():
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    index=options_row.index(config['facet_row']),
                    help="Select data column."
                )
            else:
                facet_row = st.selectbox(
                    label=config['y'] + ' plot rows',
                    options=options_row,
                    help="Select data column."
                )
        with c3:
            options_col = ['---'] + report_grp_by
            if 'facet_column' in config.keys():
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    index=options_col.index(config['facet_column']),
                    help="Select data column."
                )
            else:
                facet_column = st.selectbox(
                    label=config['y'] + ' plot columns',
                    options=options_col,
                    help="Select data column."
                )
        with c4:
            xplot_y_log = st.radio(
                'Y Axis scale',
                ('Linear', 'Logarithmic'),
                horizontal=True,
                help="Select axis scale.",
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

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
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    ih_analysis = calculate_reports_data(ih_analysis, cp_config).to_pandas()
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    if len(ih_analysis[config['x']].unique()) < 30:
        fig = px.bar(ih_analysis,
                     x=config['x'],
                     y=config['y'],
                     color=xplot_col,
                     facet_col=facet_column,
                     facet_row=facet_row,
                     log_y=xplot_y_bool,
                     barmode="group",
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
            x=config['x'],
            y=config['y'],
            color=xplot_col,
            facet_col=facet_column,
            facet_row=facet_row,
            log_y=xplot_y_bool,
            title=config['description']
        )
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_yaxes(dict(tickformat=',.2'))
        fig.update_layout(
            xaxis_title=config['x'],
            yaxis_title=config['y'],
            hovermode="x unified",
            autosize=True,
            minreducedheight=640,
            height=640
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def engagement_ctr_cards_subplot(ih_analysis: Union[pl.DataFrame, pd.DataFrame],
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
def eng_conv_polarbar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                           config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    theme = st.session_state['theme']
    if theme is None:
        template = 'none'
    else:
        if theme['base'] == 'dark':
            template = 'plotly_dark'
        else:
            template = 'none'
    fig = px.bar_polar(ih_analysis,
                       r=config["r"],
                       theta=config["theta"],
                       color=config["color"],
                       barmode="group",
                       template=template,
                       title=config['description'],
                       )
    fig.update_polars(radialaxis_tickformat=',.2%')
    fig.update_layout(
        polar_hole=0.25,
        height=700,
        margin=dict(b=25, t=50, l=0, r=0),
        showlegend=strtobool(config["showlegend"])
    )
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def eng_conv_treemap_plot(data: Union[pl.DataFrame, pd.DataFrame],
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
                     hover_data=['StdErr', 'Positives', 'Negatives'],
                     height=640,
                     )
    fig.update_traces(textinfo="label+value+percent parent+percent root")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, use_container_width=True)
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


def eng_conv_ml_heatmap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                             config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    new_df = ih_analysis.pivot(index=config['y'], columns=config['x'])[config['color']].fillna(0)
    fig = px.imshow(new_df, x=new_df.columns, y=new_df.index,
                    color_continuous_scale=px.colors.sequential.RdBu_r,
                    text_auto=",.2%",
                    aspect="auto",
                    title=config['description'],
                    contrast_rescaling="minmax",
                    height=max(600, 40 * len(new_df.index))
                    )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["type", "heatmap"],
                        label="Heatmap",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "surface"],
                        label="3D Surface",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    fig = fig.update_traces(hovertemplate=config['x'] + ' : %{x}' + '<br>' +
                                          config['y'] + ' : %{y}' + '<br>' +
                                          config['color'] + ' : %{z}<extra></extra>')

    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


def eng_conv_ml_scatter_plot(data: Union[pl.DataFrame, pd.DataFrame],
                             config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    fig = px.scatter(ih_analysis,
                     title=config['description'],
                     x=config['x'], y=config['y'],
                     animation_frame=config['animation_frame'],
                     animation_group=config['animation_group'],
                     size=config['size'], color=config['color'],
                     hover_name=config['animation_group'],
                     size_max=100, log_x=strtobool(config.get('log_x', False)),
                     log_y=strtobool(config.get('log_y', False)),
                     range_y=[ih_analysis[config['y']].min(), ih_analysis[config['y']].max()],
                     range_x=[ih_analysis[config['x']].min(), ih_analysis[config['x']].max()],
                     height=640)
    fig.update_layout(scattermode="group", scattergap=0.75)

    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def experiment_z_score_bar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                                config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    grp_by = []
    if 'facet_column' in config.keys():
        grp_by.append(config['facet_column'])
    if 'facet_row' in config.keys():
        grp_by.append(config['facet_row'])
    grp_by.append(config['x'])
    ih_analysis = ih_analysis.dropna().sort_values(by=grp_by, ascending=False)
    if 'facet_row' in config.keys():
        height = max(600, 20 * len(ih_analysis[config['y']].unique()) * len(ih_analysis[config['facet_row']].unique()))
    else:
        height = max(600, 20 * len(ih_analysis[config['y']].unique()))

    fig = px.bar(ih_analysis,
                 x=config['x'],
                 y=config['y'],
                 color=config['y'],
                 facet_row=config['facet_row'] if 'facet_row' in config.keys() else None,
                 facet_col=config['facet_column'] if 'facet_column' in config.keys() else None,
                 orientation='h',
                 title=config['description'],
                 height=height,
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
                pad={"r": 10, "t": 20},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.add_vrect(x0=-1.96, x1=1.96, line_width=0, fillcolor="red", opacity=0.1)
    fig.update_layout(
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def experiment_odds_ratio_plot(data: Union[pl.DataFrame, pd.DataFrame],
                               config: dict) -> pd.DataFrame:
    def categorize_color(g_odds_ratio_ci_high, g_odds_ratio_ci_low):
        if (g_odds_ratio_ci_high < 1) & (g_odds_ratio_ci_low < 1):
            return 'Control'
        elif (g_odds_ratio_ci_high > 1) & (g_odds_ratio_ci_low > 1):
            return 'Test'
        else:
            return 'N/A'

    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()
    if config['x'].startswith("g"):
        x = 'g_odds_ratio_stat'
        ih_analysis["x_plus"] = ih_analysis["g_odds_ratio_ci_high"] - ih_analysis["g_odds_ratio_stat"]
        ih_analysis["x_minus"] = ih_analysis["g_odds_ratio_stat"] - ih_analysis["g_odds_ratio_ci_low"]
        x_plus = 'x_plus'
        x_minus = 'x_minus'
        ih_analysis['color'] = ih_analysis.apply(
            lambda lambdax: categorize_color(lambdax.g_odds_ratio_ci_high, lambdax.g_odds_ratio_ci_low), axis=1)
    else:
        x = 'chi2_odds_ratio_stat'
        ih_analysis["x_plus"] = ih_analysis["chi2_odds_ratio_ci_high"] - ih_analysis["chi2_odds_ratio_stat"]
        ih_analysis["x_minus"] = ih_analysis["chi2_odds_ratio_stat"] - ih_analysis["chi2_odds_ratio_ci_low"]
        x_plus = 'x_plus'
        x_minus = 'x_minus'
        ih_analysis['color'] = ih_analysis.apply(
            lambda lambdax: categorize_color(lambdax.chi2_odds_ratio_ci_high, lambdax.chi2_odds_ratio_ci_low), axis=1)

    grp_by = []
    if 'facet_column' in config.keys():
        grp_by.append(config['facet_column'])
    if 'facet_row' in config.keys():
        grp_by.append(config['facet_row'])
    grp_by.append(x)

    ih_analysis = ih_analysis.dropna().sort_values(by=grp_by, ascending=True)
    color_discrete_sequence = ["#e74c3c", "#f1c40f", "#2ecc71"]
    if 'facet_row' in config.keys():
        height = max(600, 20 * len(ih_analysis[config['facet_row']].unique()) * len(ih_analysis[config['y']].unique()))
    else:
        height = max(600, 20 * len(ih_analysis[config['y']].unique()))
    fig = px.scatter(ih_analysis,
                     x=x,
                     y=config['y'],
                     color=ih_analysis['color'],
                     color_discrete_sequence=color_discrete_sequence,
                     facet_row=config['facet_row'] if 'facet_row' in config.keys() else None,
                     facet_col=config['facet_column'] if 'facet_column' in config.keys() else None,
                     error_x=x_plus,
                     error_x_minus=x_minus,
                     orientation='h',
                     title=config['description'],
                     height=height
                     )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.add_vline(x=1, line_width=2, line_dash="dash", line_color="darkred")
    fig.update_layout(
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    ih_analysis.drop(columns=['color'], inplace=True, errors='ignore')
    return ih_analysis


@timed
def histogram_plot(data: Union[pl.DataFrame, pd.DataFrame],
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
    st.plotly_chart(fig, use_container_width=True)
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
                pl.sum("Count").alias("Count"),
                pl.sum("lifetime_value").alias("lifetime_value"),
                pl.sum("unique_holdings").alias("unique_holdings"),
                pl.mean("monetary_value").alias("monetary_value")
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
    theme = st.session_state['theme']
    if theme is None:
        template = 'none'
    else:
        if theme['base'] == 'dark':
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
    st.plotly_chart(fig, use_container_width=True)
    return ih_analysis


@timed
def clv_exposure_plot(data: Union[pl.DataFrame, pd.DataFrame],
                      config: dict) -> pd.DataFrame:
    data = calculate_reports_data(data, config).to_pandas()
    clv_analysis = pl.from_pandas(filter_dataframe(align_column_types(data), case=False))

    if clv_analysis.shape[0] == 0:
        st.warning("No data available.")
        st.stop()

    clv_analysis = pds_sample.sample(clv_analysis, 100).sort(['recency', 'tenure'])
    fig = plot_customer_exposure(clv_analysis, linewidth=0.5, size=0.75)
    st.plotly_chart(fig, use_container_width=True)
    return clv_analysis.to_pandas()


def plot_customer_exposure(
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
            go.Scatter(
                x=[0, rec],
                y=[customer_idx[idx], customer_idx[idx]],
                mode='lines',
                line=dict(color=recency_color, width=linewidth)
            )
        )

    for idx, (rec, t) in enumerate(zip(recency, T)):
        fig.add_trace(
            go.Scatter(
                x=[rec, t],
                y=[customer_idx[idx], customer_idx[idx]],
                mode='lines',
                line=dict(color=T_color, width=linewidth)
            )
        )
    fig.add_trace(
        go.Scatter(
            x=recency,
            y=customer_idx,
            mode='markers',
            marker=dict(color=recency_color, size=size),
            name=labels[0] if labels else 'Recency'
        )
    )
    fig.add_trace(
        go.Scatter(
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
    st.plotly_chart(fig, use_container_width=True)
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
    t = 12 * 30 * predict_lifespan
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
        st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
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
                time=12 * predict_lifespan,
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
            st.plotly_chart(fig, use_container_width=True)

    return clv


def get_figures() -> dict:
    figures = {}
    reports = get_config()["reports"]
    for report in reports:
        params = reports[report]
        if params['metric'].startswith("engagement"):
            if params['type'] == 'line':
                if params['y'] == "CTR":
                    figures[report] = engagement_ctr_line_plot
                elif params['y'] == "Lift":
                    figures[report] = engagement_lift_line_plot
                elif params['y'] == "Lift_Z_Score":
                    figures[report] = engagement_z_score_plot
                else:
                    raise Exception(params['y'] + " is not supported parameter for plot " + params['type'] +
                                    " and metric: " + params['metric'])
            elif params['type'] == 'bar_polar':
                figures[report] = eng_conv_polarbar_plot
            elif params['type'] == 'gauge':
                figures[report] = engagement_ctr_gauge_plot
            elif params['type'] == 'treemap':
                figures[report] = eng_conv_treemap_plot
            elif params['type'] == 'heatmap':
                figures[report] = eng_conv_ml_heatmap_plot
            elif params['type'] == 'scatter':
                figures[report] = eng_conv_ml_scatter_plot
            else:
                raise Exception(params['type'] + " is not supported type for plot " + params['type'] +
                                " and metric: " + params['metric'])
        elif params['metric'].startswith("model_ml_scores"):
            if params['type'] == 'heatmap':
                figures[report] = eng_conv_ml_heatmap_plot
            elif params['type'] == 'scatter':
                figures[report] = eng_conv_ml_scatter_plot
            elif params['type'] == 'treemap':
                figures[report] = model_ml_treemap_plot
            else:
                if params['y'] == "roc_auc":
                    figures[report] = model_ml_scores_line_plot_roc_pr_curve
                elif params['y'] == "average_precision":
                    figures[report] = model_ml_scores_line_plot_roc_pr_curve
                else:
                    figures[report] = model_ml_scores_line_plot
        elif params['metric'].startswith("conversion"):
            if params['type'] == 'heatmap':
                figures[report] = eng_conv_ml_heatmap_plot
            elif params['type'] == 'scatter':
                figures[report] = eng_conv_ml_scatter_plot
            elif params['type'] == 'gauge':
                figures[report] = conversion_rate_gauge_plot
            elif params['type'] == 'treemap':
                figures[report] = eng_conv_treemap_plot
            elif params['type'] == 'bar_polar':
                figures[report] = eng_conv_polarbar_plot
            else:
                if params['y'] == "ConversionRate":
                    figures[report] = conversion_rate_line_plot
                elif params['y'] == "Revenue":
                    figures[report] = conversion_revenue_line_plot
                else:
                    raise Exception(params['y'] + " is not supported parameter for plot " + params['type'] +
                                    " and metric: " + params['metric'])
        elif params['metric'].startswith("descriptive"):
            if params['type'] == 'line':
                figures[report] = descriptive_line_plot
            elif params['type'] == 'boxplot':
                figures[report] = descriptive_box_plot
            elif params['type'] == 'funnel':
                figures[report] = descriptive_funnel
            else:
                raise Exception(params['type'] + " is not supported type for metric: " + params['metric'])
        elif params['metric'].startswith("experiment"):
            if params['x'] == "z_score":
                figures[report] = experiment_z_score_bar_plot
            elif params['x'].startswith("g") | params['x'].startswith("chi2"):
                figures[report] = experiment_odds_ratio_plot
            else:
                raise Exception(params['x'] + " is not supported parameter for plot " + params['type'] +
                                " and metric: " + params['metric'])
        elif params['metric'].startswith("clv"):
            if params['type'] == 'histogram':
                figures[report] = histogram_plot
            elif params['type'] == 'bar_polar':
                figures[report] = clv_polarbar_plot
            elif params['type'] == 'exposure':
                figures[report] = clv_exposure_plot
            elif params['type'] == 'corr':
                figures[report] = clv_correlation_plot
            elif params['type'] == 'model':
                figures[report] = clv_model_plot
            else:
                raise Exception(params['type'] + " is not supported parameter for metric: " + params['metric'])
        else:
            raise Exception(params['metric'] + " is not supported metric. Check spelling. ")
    return figures
