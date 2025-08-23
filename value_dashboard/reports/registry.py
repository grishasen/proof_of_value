from value_dashboard.reports.clv_plots import *
from value_dashboard.reports.conversion_plots import *
from value_dashboard.reports.descriptive_plots import *
from value_dashboard.reports.engagement_plots import *
from value_dashboard.reports.experiment_plots import *
from value_dashboard.reports.model_ml_scores_plots import *
from value_dashboard.utils.config import get_config


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
                figures[report] = default_bar_line_plot
        elif params['metric'].startswith("model_ml_scores"):
            if params['type'] == 'heatmap':
                figures[report] = eng_conv_ml_heatmap_plot
            elif params['type'] == 'scatter':
                figures[report] = eng_conv_ml_scatter_plot
            elif params['type'] == 'treemap':
                figures[report] = model_ml_treemap_plot
            else:
                if 'y' in params.keys():
                    if params['y'] == "roc_auc":
                        figures[report] = model_ml_scores_line_plot_roc_pr_curve
                    elif params['y'] == "average_precision":
                        figures[report] = model_ml_scores_line_plot_roc_pr_curve
                    else:
                        figures[report] = default_bar_line_plot
                else:
                    figures[report] = default_bar_line_plot
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
                if 'y' in params.keys():
                    if params['y'] == "ConversionRate":
                        figures[report] = conversion_rate_line_plot
                    elif params['y'] == "Revenue":
                        figures[report] = conversion_revenue_line_plot
                    else:
                        figures[report] = default_bar_line_plot
                else:
                    figures[report] = default_bar_line_plot
        elif params['metric'].startswith("descriptive"):
            if params['type'] == 'line':
                figures[report] = descriptive_line_plot
            elif params['type'] == 'boxplot':
                figures[report] = descriptive_box_plot
            elif params['type'] == 'funnel':
                figures[report] = descriptive_funnel
            elif params['type'] == 'histogram':
                figures[report] = descriptive_hist_plot
            else:
                figures[report] = default_bar_line_plot
        elif params['metric'].startswith("experiment"):
            if 'x' in params.keys():
                if params['x'] == "z_score":
                    figures[report] = experiment_z_score_bar_plot
                elif params['x'].startswith("g") | params['x'].startswith("chi2"):
                    figures[report] = experiment_odds_ratio_plot
                else:
                    figures[report] = default_bar_line_plot
            else:
                figures[report] = default_bar_line_plot
        elif params['metric'].startswith("clv"):
            if params['type'] == 'histogram':
                figures[report] = clv_histogram_plot
            elif params['type'] == 'treemap':
                figures[report] = clv_treemap_plot
            elif params['type'] == 'exposure':
                figures[report] = clv_exposure_plot
            elif params['type'] == 'corr':
                figures[report] = clv_correlation_plot
            elif params['type'] == 'model':
                figures[report] = clv_model_plot
            elif params['type'] == 'rfm_density':
                figures[report] = clv_rfm_density_plot
            else:
                raise Exception(params['type'] + " is not supported parameter for metric: " + params['metric'])
        else:
            raise Exception(params['metric'] + " is not supported metric. Check spelling. ")
    return figures
