import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from value_dashboard.reports.repdata import merge_descriptive_digests
from value_dashboard.reports.shared_plot_utils import *
from value_dashboard.utils.config import get_config
from value_dashboard.utils.polars_utils import digest_to_histogram


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
    report_grp_by = sorted(list(set(report_grp_by)))

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
            columns = sorted(columns)
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
            opts = sorted(opts)
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
                ('Linear', 'Log'),
                horizontal=True,
                help="Select axis scale.",
                # label_visibility='collapsed'
            )
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Log':
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
        return ih_analysis

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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def descriptive_box_plot(data: Union[pl.DataFrame, pd.DataFrame],
                         config: dict) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    columns_conf = m_config['columns']
    num_columns = [col for col in columns_conf if (col + '_Mean') in data.columns]

    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = sorted(list(set(report_grp_by)))

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
            num_columns = sorted(num_columns)
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
        return ih_analysis

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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def descriptive_funnel(data: Union[pl.DataFrame, pd.DataFrame],
                       config: dict, options_panel: bool = True) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    report_grp_by = config['group_by']
    title = config['description']
    x = config['x']
    color = config['color']
    stages = config['stages']
    height = config.get('height', 640)
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
    if options_panel:
        ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis

    if facet_row:
        height = max(height, 350 * len(ih_analysis[facet_row].unique()))
    else:
        height = height

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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    return ih_analysis


@timed
def descriptive_hist_plot(data: Union[pl.DataFrame, pd.DataFrame],
                          config: dict) -> pd.DataFrame:
    metric = config["metric"]
    m_config = get_config()["metrics"][metric]
    columns_conf = m_config['columns']
    num_columns = [col for col in columns_conf if (col + '_Mean') in data.columns]

    report_grp_by = m_config['group_by'] + config['group_by'] + get_config()["metrics"]["global_filters"]
    report_grp_by = sorted(list(set(report_grp_by)))

    title = config['description']
    adv_on = st.toggle("Advanced options", value=True, key="Advanced options" + config['description'],
                       help="Show advanced reporting options")
    facet_row = '---' if not 'facet_row' in config.keys() else config['facet_row']
    facet_column = '---' if not 'facet_column' in config.keys() else config['facet_column']
    if adv_on:
        c1, c2, c3 = st.columns(3)
        with c1:
            num_columns = sorted(num_columns)
            config['x'] = st.selectbox(
                label="## Select data property ",
                options=num_columns,
                index=num_columns.index(config['x']),
                label_visibility='visible',
                help="Select score to visualize."
            )
        with c2:
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
        with c3:
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

    grp_by = []
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

    cp_config = config.copy()
    cp_config['group_by'] = grp_by

    report_data = merge_descriptive_digests(data, cp_config)
    ih_analysis = report_data.to_pandas()
    ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)

    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis

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

    for item in ih_analysis.to_dict('records'):
        bin_edges, bin_counts = digest_to_histogram(item[config['x'] + '_tdigest'], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        row = item[facet_row] if facet_row else ''
        col = item[facet_column] if facet_column else ''
        subplot_row, subplot_col = row_col_map[(row, col)]

        fig.add_bar(
            x=bin_centers, y=bin_counts, width=np.diff(bin_edges),
            name=' ',
            row=subplot_row,
            col=subplot_col
        )

    fig.update_xaxes(tickfont=dict(size=8))
    fig.update_layout(
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

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    report_data = calculate_reports_data(data, cp_config)
    report_data = report_data.to_pandas()
    return report_data
