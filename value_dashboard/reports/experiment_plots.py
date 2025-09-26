from value_dashboard.reports.shared_plot_utils import *


@timed
def experiment_z_score_bar_plot(data: Union[pl.DataFrame, pd.DataFrame],
                                config: dict, options_panel: bool = True) -> pd.DataFrame:
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
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
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    ih_analysis.drop(columns=['color'], inplace=True, errors='ignore')
    return ih_analysis
