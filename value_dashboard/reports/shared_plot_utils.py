from typing import Union

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

from value_dashboard.reports.repdata import calculate_reports_data
from value_dashboard.utils.config import get_config
from value_dashboard.utils.py_utils import strtobool
from value_dashboard.utils.st_utils import filter_dataframe, align_column_types
from value_dashboard.utils.timer import timed


@timed
def eng_conv_ml_heatmap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                             config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
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

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def eng_conv_ml_scatter_plot(data: Union[pl.DataFrame, pd.DataFrame],
                             config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
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

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def eng_conv_treemap_plot(data: Union[pl.DataFrame, pd.DataFrame],
                          config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
    fig = px.treemap(ih_analysis, path=[px.Constant("ALL")] + config['group_by'], values='Count',
                     color=config['color'],
                     color_continuous_scale=px.colors.sequential.RdBu_r,
                     title=config['description'],
                     hover_data=['StdErr', 'Positives', 'Negatives'],
                     height=640,
                     )
    fig.update_traces(textinfo="label+value+percent parent+percent root")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def eng_conv_polarbar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                           config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
    ih_analysis = filter_dataframe(align_column_types(report_data), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
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
                       title=config['description'],
                       )
    fig.update_polars(radialaxis_tickformat=',.2%')
    fig.update_layout(
        polar_hole=0.25,
        height=700,
        margin=dict(b=25, t=50, l=0, r=0),
        showlegend=strtobool(config["showlegend"])
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def default_bar_line_plot(data: Union[pl.DataFrame, pd.DataFrame],
                          config: dict) -> pd.DataFrame:
    adv_on = st.toggle("Advanced options", value=True, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    xplot_y_bool = False
    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)

    if adv_on:
        plot_menu = get_plot_parameters_menu(config=config, is_y_axis_required=True)
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
    cp_config['x'] = x_axis
    cp_config['y'] = y_axis
    cp_config['color'] = xplot_col

    if (x_axis is None) or (y_axis is None):
        st.warning("Please select X and Y Axis.")
        st.stop()

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
            color=xplot_col,
            title=config['description'],
            facet_col=facet_column,
            facet_row=facet_row,
            custom_data=[xplot_col],
            log_y=xplot_y_bool,
        )
    fig.update_xaxes(tickfont=dict(size=10))
    yaxis_names = ['yaxis'] + [axis_name for axis_name in fig.layout._subplotid_props if 'yaxis' in axis_name]
    # yaxis_layout_dict = {yaxis_name + "_tickformat": ',.2%' for yaxis_name in yaxis_names}
    # fig.update_layout(yaxis_layout_dict)
    height = 640
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
                                          y_axis + ' : %{y:.4}' + '<br>' +
                                          '<extra></extra>'
                            )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


def get_plot_parameters_menu(config: dict, is_y_axis_required: bool = True):
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = m_config['group_by']
    report_grp_by = sorted(report_grp_by)
    scores = m_config['scores']

    xplot_y_bool = False
    xplot_col = config.get('color', None)
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    x_axis = config.get('x', None)
    y_axis = config.get('y', None)

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
        xplot_y_log = st.radio(
            'Y Axis scale',
            ('Linear', 'Log'),
            horizontal=True,
            help="Select axis scale.",
        )
        if xplot_y_log == 'Linear':
            xplot_y_bool = False
        elif xplot_y_log == 'Log':
            xplot_y_bool = True

    return {'x': x_axis, 'color': xplot_col, 'facet_row': facet_row, 'facet_col': facet_column, 'y': y_axis,
            'log_y': xplot_y_bool}
