from value_dashboard.reports.shared_plot_utils import *


@timed
def experiment_z_score_bar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                                config: dict, options_panel: bool = True) -> pd.DataFrame:
    """
    Render a horizontal z-score bar (or line) plot and return the analysis DataFrame.

    The function prepares experiment analysis data via ``calculate_reports_data``,
    optionally filters it using ``filter_dataframe``, and renders a horizontal bar plot
    of the measure in ``config['x']`` against categories in ``config['y']``. An
    optional UI control allows switching between bar and line trace types. A shaded
    vertical region is drawn between -1.96 and 1.96 to indicate a typical 95% z-score
    acceptance band.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Source dataset compatible with the reporting utilities and containing fields
        referenced by ``config``.
    config : dict
        Plot and data configuration. Expected keys include:
            - ``'x'`` (str): Numeric field for the x-axis (e.g., z-score/statistic).
            - ``'y'`` (str): Categorical field for the y-axis (labels).
            - ``'description'`` (str): Chart title.
            - ``'facet_row'`` (str, optional): Row facet column.
            - ``'facet_column'`` (str, optional): Column facet column.
            - ``'height'`` (int, optional): Base plot height in pixels (default 640).
            - Any other keys required by ``calculate_reports_data``.
    options_panel : bool, default True
        If ``True``, adds a Plotly update menu to toggle between bar and line traces.

    Returns
    -------
    pd.DataFrame
        The processed Pandas DataFrame used for plotting. If no data is available,
        an empty DataFrame is returned after a Streamlit warning.

    Notes
    -----
    - Bars are oriented horizontally (``orientation='h'``).
    - Plot height scales with the number of categories and row facets for readability.
    - Legend is hidden; colors are assigned per ``config['y']``.
    - The function renders output via ``st.plotly_chart`` and may display warnings
      with ``st.warning``.

    Raises
    ------
    KeyError
        If required keys (e.g., ``'x'``, ``'y'``, ``'description'``) are missing in ``config``.
    Exception
        Propagated from downstream utilities such as ``calculate_reports_data`` and
        ``filter_dataframe``.
    """
    ih_analysis = calculate_reports_data(data, config).to_pandas()
    if options_panel:
        ih_analysis = filter_dataframe(align_column_types(ih_analysis), case=False)
    height = config.get('height', 640)

    if ih_analysis.shape[0] == 0:
        st.warning("No data available.")
        return ih_analysis
    grp_by = []
    if 'facet_column' in config.keys():
        grp_by.append(config['facet_column'])
    if 'facet_row' in config.keys():
        grp_by.append(config['facet_row'])
    grp_by.append(config['x'])
    ih_analysis = ih_analysis.dropna().sort_values(by=grp_by, ascending=False)
    if 'facet_row' in config.keys():
        height = max(height,
                     20 * len(ih_analysis[config['y']].unique()) * len(ih_analysis[config['facet_row']].unique()))
    else:
        height = max(height, 10 * len(ih_analysis[config['y']].unique()))

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

    if options_panel:
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
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    return ih_analysis


@timed
def experiment_odds_ratio_plot(data: Union[pl.DataFrame, pd.DataFrame],
                               config: dict) -> pd.DataFrame:
    """
    Render an odds ratio scatter plot with asymmetric confidence intervals and return the analysis DataFrame.

    Depending on ``config['x']``, the function plots either generalized odds ratios
    (``g_*`` fields) or chi-squared-based odds ratios (``chi2_*`` fields). Points are
    color-coded by whether the confidence interval is entirely below 1.0 (Control),
    entirely above 1.0 (Test), or straddles 1.0 (N/A). The plot supports row/column
    faceting and adjusts height to the number of categories.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Input dataset compatible with ``calculate_reports_data`` and containing the
        requisite odds ratio statistics and confidence intervals.
    config : dict
        Plot and data configuration. Expected keys include:
            - ``'x'`` (str): Selector to choose the odds ratio family. If it starts
              with ``'g'``, uses generalized odds ratio fields:
              ``g_odds_ratio_stat``, ``g_odds_ratio_ci_low``, ``g_odds_ratio_ci_high``;
              otherwise uses chi-squared variants:
              ``chi2_odds_ratio_stat``, ``chi2_odds_ratio_ci_low``, ``chi2_odds_ratio_ci_high``.
            - ``'y'`` (str): Categorical field for the y-axis.
            - ``'description'`` (str): Chart title.
            - ``'facet_row'`` (str, optional): Row facet column.
            - ``'facet_column'`` (str, optional): Column facet column.
            - Any other keys required by ``calculate_reports_data``.

    Returns
    -------
    pd.DataFrame
        The processed Pandas DataFrame used to draw the plot. The temporary
        ``'color'`` column used for visual encoding is dropped before returning.
        If no data is available, an empty DataFrame is returned after a Streamlit warning.

    Notes
    -----
    - X reference line at 1.0 indicates no effect for odds ratios.
    - Asymmetric error bars are computed as:
      ``error_x = ci_high - stat`` and ``error_x_minus = stat - ci_low``.
    - Color coding of points:
        * CI entirely < 1.0  → ``'Control'``
        * CI entirely > 1.0  → ``'Test'``
        * Otherwise          → ``'N/A'``
    - Legend is hidden; figure height scales with category and facet counts.
    - The figure is rendered via ``st.plotly_chart`` and may surface warnings via ``st.warning``.

    Raises
    ------
    KeyError
        If required keys (e.g., ``'x'``, ``'y'``, ``'description'``) are missing in ``config``.
    Exception
        Propagated from downstream utilities such as ``calculate_reports_data`` and
        ``filter_dataframe``.
    """

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
        return ih_analysis
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
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    ih_analysis.drop(columns=['color'], inplace=True, errors='ignore')
    return ih_analysis
