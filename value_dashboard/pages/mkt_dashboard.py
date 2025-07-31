import polars as pl
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters

from value_dashboard.pipeline.ih import load_data
from value_dashboard.reports.conversion_plots import conversion_rate_card, conversion_touchpoints_card, \
    conversion_rate_line_plot
from value_dashboard.reports.descriptive_plots import descriptive_funnel
from value_dashboard.reports.engagement_plots import engagement_rate_card, engagement_ctr_line_plot
from value_dashboard.reports.experiment_plots import experiment_z_score_bar_plot
from value_dashboard.reports.model_ml_scores_plots import ml_scores_card, ml_scores_pers_card
from value_dashboard.utils.config import get_config

if "data_loaded" not in st.session_state:
    st.warning("Please configure your files in the `data import` tab.")
    st.stop()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
metrics = get_config()["metrics"]
for metric in metrics:
    params = metrics[metric]
    if metric.startswith("conversion"):
        df_conversion = load_data()[metric].clone()
        conversion_metric = metric
    if metric.startswith("engagement"):
        df_engagement = load_data()[metric].clone()
        engagement_metric = metric
    if metric.startswith("model_ml_scores"):
        df_ml = load_data()[metric].clone()
        ml_metric = metric
    if metric.startswith("experiment"):
        df_experiments = load_data()[metric].clone()
        experiments_metric = metric
    if metric.startswith("descriptive"):
        df_descriptive = load_data()[metric].clone()
        descriptive_metric = metric

with st.sidebar:
    dynamic_filters = DynamicFilters(df_engagement.to_pandas(),
                                     filters=get_config()["metrics"]["global_filters"])
    st.write("Filter data 👇")
    dynamic_filters.display_filters()

df_engagement = pl.from_pandas(dynamic_filters.filter_df())
dynamic_filters.df = df_conversion.to_pandas()
df_conversion = pl.from_pandas(dynamic_filters.filter_df())
dynamic_filters.df = df_ml.to_pandas()
df_ml = pl.from_pandas(dynamic_filters.filter_df())
dynamic_filters.df = df_experiments.to_pandas()
df_experiments = pl.from_pandas(dynamic_filters.filter_df())
dynamic_filters.df = df_descriptive.to_pandas()
df_descriptive = pl.from_pandas(dynamic_filters.filter_df())

with kpi1:
    conversion_rate_card(df_conversion)
with kpi2:
    engagement_rate_card(df_engagement)
with kpi3:
    conversion_touchpoints_card(df_conversion)
with kpi4:
    ml_scores_card(df_ml, ml_metric)
with kpi5:
    ml_scores_pers_card(df_ml, ml_metric)

mid_col1, mid_col2 = st.columns(2)
plot_height = 420
with mid_col1:
    config = dict()
    config['metric'] = engagement_metric
    config['type'] = 'line'
    config['description'] = 'Engagement Rate Over Time'
    config['group_by'] = ['Day', 'CustomerType', 'Channel']
    config['x'] = 'Day'
    config['y'] = 'CTR'
    config['color'] = "Channel"
    config['height'] = plot_height
    engagement_ctr_line_plot(df_engagement.to_pandas(), config, options_panel=False)

with mid_col2:
    config = dict()
    config['metric'] = descriptive_metric
    config['type'] = 'funnel'
    config['description'] = 'Feedback Loop Funnel'
    config['group_by'] = ['Issue']
    config['stages'] = ['Impression', 'Clicked', 'Conversion']
    config['x'] = 'Outcome'
    config['color'] = 'Issue'
    config['height'] = plot_height
    descriptive_funnel(df_descriptive.to_pandas(), config, options_panel=False)

bot_col1, bot_col2 = st.columns(2)
with bot_col1:
    config = dict()
    config['metric'] = conversion_metric
    config['type'] = 'line'
    config['description'] = 'Conversion Rate by Channel'
    config['group_by'] = ['Day', 'Channel']
    config['x'] = 'Day'
    config['y'] = 'ConversionRate'
    config['color'] = "Channel"
    config['height'] = plot_height
    conversion_rate_line_plot(df_conversion.to_pandas(), config, options_panel=False)

with bot_col2:
    config = dict()
    config['metric'] = experiments_metric
    config['type'] = 'line'
    config['description'] = 'Business Experiments'
    config['group_by'] = ['ExperimentName']
    config['x'] = 'z_score'
    config['y'] = 'ExperimentName'
    config['height'] = plot_height
    experiment_z_score_bar_plot(df_experiments.to_pandas(), config, options_panel=False)
