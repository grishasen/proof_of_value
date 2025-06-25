from typing import Union

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

from value_dashboard.reports.repdata import calculate_reports_data
from value_dashboard.utils.st_utils import filter_dataframe, align_column_types
from value_dashboard.utils.string_utils import strtobool
from value_dashboard.utils.timer import timed


@timed
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


@timed
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
def eng_conv_polarbar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                           config: dict) -> pd.DataFrame:
    report_data = calculate_reports_data(data, config).to_pandas()
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
