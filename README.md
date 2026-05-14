# CDH Value Dashboard Application

## Overview

The CDH Value Dashboard Application is an open-source prototype designed to provide customizable metrics and insights
for linking marketing actions to business outcomes, such as customer lifetime value (CLV) and conversion rates. The
dashboard utilizes technologies like Streamlit, Polars, Plotly, and DuckDB for efficient data processing and interactive
visualizations, supporting decision-making through clear data visualization.

Go to [Wiki page](https://github.com/grishasen/proof_of_value/wiki) for additional info.

## Features

- **Data Import**: Load Interaction History and Product Holdings data from folders or uploads.
- **Interactive Dashboard**: Visualize KPIs, configurable reports, and filtered data interactively.
- **Configuration Editor**: Review, validate, edit, apply, and download TOML configuration from the app.
- **Chat With Data**: Ask questions over loaded metrics data when optional AI dependencies are installed.
- **AI Configuration Studio**: Generate a config draft from an Interaction History sample, review AI changes, validate
  reports, and repair AI-owned sections before export.

## How To Use

1. **Data Import**:
    - Navigate to the "Data Import" page.
    - For Interaction History, load raw data from a folder or upload ZIP, Parquet, or JSON files. You can also switch
      off `Import raw data` and upload a pre-aggregated metrics JSON file.
    - For Product Holdings / CLV, load data from a folder or upload ZIP, Parquet, JSON, CSV, or XLSX files. The selected
      config `file_type` must match the uploaded data format.
    - **For demo**:
        - IH reporting: switch off `Import raw data` toggle. Upload JSON file available in `data` folder (unzip
          archive).
        - CLV analytics: import product holdings zip file directly.
    - Once the data is imported, it will be processed and prepared for visualization.

2. **Dashboard**:
    - Navigate to the "Dashboard" page.
    - Review top-level KPIs and prebuilt charts with global filters.
    - Use "Reports and Analysis" for configurable report views and the underlying aggregated data.
    - Use "Chat with data" with your own API key after installing the optional AI dependencies.

3. **Configuration Editor**:
    - Navigate to the "Configuration Editor" page to edit the active TOML config.
    - Use the Config Health and Review Progress panels before applying or downloading changes.
    - Use the Reports step to review report validation status and edit reports in visual or raw mode.

4. **AI Configuration Studio**:
    - Install optional AI dependencies with `uv sync --extra ai`.
    - Upload an Interaction History sample and confirm required time fields, defaults, filters, and calculated fields.
    - Approve the fields and sample values that may be sent to AI. The privacy summary shows exactly what is included
      in the prompt.
    - Generate an AI draft, review the metric/report/variant changes, then accept only the metrics and reports to keep.
    - Refresh reports after metric edits, review report validation, and use AI Repair for blocking metric/report issues.
    - Save or apply only after validation passes.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) as the main dependency manager.

### Prerequisites

- Python >=3.11, <=3.14
- `uv`

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install From Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/grishasen/proof_of_value.git
   cd proof_of_value
   ```

2. **Create the project environment and install base dependencies**:
   ```bash
   uv sync
   ```

3. **Install optional AI dependencies**:
   ```bash
   uv sync --extra ai
   ```
   This installs `litellm` for AI-assisted config generation and chat with data.

4. **Prepare a custom dashboard config if needed**:
   ```bash
   cp value_dashboard/config/config_template.toml value_dashboard/config/config.toml
   vi value_dashboard/config/config.toml
   ```

   AI provider settings are configured separately in `value_dashboard/config/llm_config.toml`.

5. **Run the application**:
   ```bash
   uv run cdhdashboard run
   ```
   The app uses `value_dashboard/config/config_template.toml` by default. To run with a specific config:
   ```bash
   uv run cdhdashboard run --config value_dashboard/config/config_demo.toml
   ```
   Streamlit options can be passed before app options:
   ```bash
   uv run cdhdashboard run --server.port 8502 --config value_dashboard/config/config_demo.toml
   ```

### Install As a Python Package

Base package:

```bash
uv pip install cdhdashboard
```

Package with AI extras:

```bash
uv pip install "cdhdashboard[ai]"
```

To verify the installation:

```bash
cdhdashboard run --config <config_file_path.toml>
```

### Build Distributions

Build the source distribution and wheel:

```bash
uv build
```

This creates the package archives in `dist/`.

## File Structure

- **vd_app.py**: The main entry point of the application.
- **value_dashboard/pages/home.py**: Application description.
- **value_dashboard/pages/data_import.py**: Handles data import functionality.
- **value_dashboard/pages/mkt_dashboard.py**: Contains the KPI dashboard for Interaction History metrics.
- **value_dashboard/pages/ih_analysis.py**: Contains configurable IH report views and aggregated data tables.
- **value_dashboard/pages/clv_analysis.py**: Contains the dashboard for Product Holdings data and CLV-related metrics.
- **value_dashboard/pages/chat_with_data.py**: Chat with loaded engagement, conversion, experiment, and CLV data.
- **value_dashboard/pages/configuration_editor.py**: Visual configuration editor for the active TOML config.
- **value_dashboard/pages/ai_configuration_studio.py**: AI-assisted config generator with field approval, privacy
  controls, draft diff review, validation, report refresh, and repair workflow.
- **value_dashboard/config_generator/**: Shared config editor, preprocessing, validation, diff, and AI generator helpers.
- **value_dashboard/metrics/**: Calculation of various metrics supported by the application.
- **value_dashboard/pipeline/**: Data loading and processing steps.
- **value_dashboard/reports/**: Plots and data visualization functions.
- **value_dashboard/utils/**: Utility functions for configuration and Streamlit components.
- **value_dashboard/datalake/**: Persistent cache based on DuckDB.

## Metrics

Metrics section holds configuration settings for aggregating data to calculate supported KPIs.

- **global_filters**: List of dataset columns to be used for filtering data for all reports.

The application currently supports various metrics, configured through a TOML file. Those metrics include:

- **Conversion**
    - ***Conversion Rate***: The percentage of users who take a desired action, such as making a purchase or signing up
      for a service.
    - ***Revenue***: Revenue aggregations.

```toml
[metrics.conversion]
group_by = ["Day", "Month", "Year", "Quarter", "Channel", "PlacementType", "PropensitySource", "Issue", "Group"]
filter = ""
scores = ["ConversionRate", "Revenue"]
positive_model_response = ["Conversion"]
negative_model_response = ["Impression"]
```

- **Engagement**: Measures of user interaction with the product or service.
    - ***Click-Through Rate (CTR)***: The ratio of users who click on an ad to the number of total users who view the
      ad.
    - ***Lift***: Measures the increase in a desired descriptive in the target group that received the action compared
      to a
      control group that received random action.

```toml
[metrics.engagement]
group_by = ["Day", "Month", "Year", "Quarter", "Channel", "PlacementType", "PropensitySource", "Issue", "Group"]
filter = """"""
scores = ["CTR", "Lift", "Lift_Z_Score", "Lift_P_Val", "Positives", "Negatives", "Count"]
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]
```

- **Machine learning and recommender systems related scores**
    - ***Area Under the ROC Curve (AUC)***: A performance measurement for classification problems at various threshold
      settings. It represents the degree or measure of separability achieved by the model.
    - ***Average Precision Score***: A summary of the precision-recall curve, combining precision and recall into a
      single metric.
    - ***Personalization***: Measures how tailored the recommendations are to the individual user.
    - ***Novelty***: Measures how new or unexpected the recommended items are to the user.

There are two options for ML metrics calculation:

- Calculate ROC AUC and average precision for smallest groups possible and aggregate as weighted average.
- Use [T-digests](https://github.com/tdunning/t-digest) to evaluate percentiles, compute TPR/FPR/FN from percentiles and
  derive ROC AUC and average precision. ***use_t_digest*** setting allows to select which approach to use.

```toml
[metrics.model_ml_scores]
group_by = ["Day", "Month", "Year", "Quarter", "Channel", "PlacementType", "PropensitySource", "Issue", "Group"]
filter = """"""
use_t_digest = "true"
scores = ["roc_auc", "average_precision", "personalization", "novelty"]
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]
```

- **Descriptive**: Describes the dataset by aggregating counts, sums, distributions, and related statistics across
  configured dimensions.
    - ***Count***: Return the number of non-null elements in the column.
    - ***Sum***: Get sum value for column.
    - ***Mean***: Get mean value.
    - ***Median***: Get median (50-percentile) value using t-digest data structure and corresponding algorithm.
    - ***p25***: Get 25-percentile using t-digest data structure and corresponding algorithm.
    - ***p75***: Get 75-percentile using t-digest data structure and corresponding algorithm.
    - ***p90***: Get 90-percentile using t-digest data structure and corresponding algorithm.
    - ***p95***: Get 95-percentile using t-digest data structure and corresponding algorithm.
    - ***Min / Max***: Get minimum and maximum values.
    - ***Std***: Get standard deviation (Delta Degrees of Freedom = 1).
    - ***Var***: Get variance (Delta Degrees of Freedom = 1).
    - ***Skew***: Compute Bowley's Skewness (Quartile Coefficient of Skewness) of a data
      set. $Skewness = \frac{(Q_3 + Q_1 - 2Q_2)}{Q_3 - Q_1} = \frac{(p75 + p25 - 2*p50)}{p75 - p25}$. For symmetrical
      data, the skewness should be about
      zero. For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in
      the right tail of the distribution.

```toml
[metrics.descriptive]
group_by = ["Day", "Month", "Year", "Quarter", "Channel", "PlacementType", "PropensitySource", "Issue", "Group", "Outcome"]
filter = """"""
use_t_digest = "true"
columns = ["Outcome", "Propensity", "FinalPropensity", "Priority", "ResponseTime", "Weight", "OutcomeWeight"]
scores = ["Count", "Sum", "Mean", "Median", "p25", "p75", "p90", "p95", "Std", "Var", "Skew", "Min", "Max"]
```

- **Experiment**: Various metrics used during A/B testing.
    - ***z_score***: z-test (normal approximation) statistics.
    - ***z_p_val***: z-test p-value.
    - ***g_stat***: g-test statistics.
    - ***g_p_val***: g-test p-value.
    - ***chi2_stat***: chi-square test of homogeneity statistics.
    - ***chi2_p_val***: chi-square test p-value.
    - ***odds_ratio_stat***: sample (or unconditional) estimate of contingency table, given
      by $\frac{table[0, 0]*table[1, 1]}{table[0, 1]*table[1, 0]}$
    - ***odds_ratio_ci_low/high***: the confidence interval of the odds ratio for the .95 confidence level.

```toml
[metrics.experiment]
group_by = ["Year", "Channel", "PlacementType", "PropensitySource"]
filter = """"""
experiment_name = "ExperimentName"
experiment_group = "ExperimentGroup"
scores = ["z_score", "z_p_val", "g_stat", "g_p_val", "chi2_stat", "chi2_p_val", "g_odds_ratio_stat", "g_odds_ratio_ci_low", "g_odds_ratio_ci_high", "chi2_odds_ratio_stat", "chi2_odds_ratio_ci_low", "chi2_odds_ratio_ci_high"]
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]
```

- **CLV**: Customer Lifetime Value analysis metrics.
    - ***frequency***: represents the number of repeat purchases that a customer has made, i.e. one less than the total
      number of purchases.
    - ***T (tenure)***: represents a customer’s “age”, i.e. the duration between a customer’s first purchase and the end
      of the period of study.
    - ***recency***: represents the time period when a customer made their most recent purchase. This is equal to the
      duration between a customer’s first and last purchase. If a customer has made only 1 purchase, their recency is 0.
    - ***monetary_value***: represents the average value of a given customer’s repeat purchases. Customers who have only
      made a single purchase have monetary values of zero.

```toml
[metrics.clv]
filter = """pl.col('PurchasedDateTime') > pl.datetime(2016, 12, 31)"""
group_by = ['ControlGroup']
scores = ['recency', 'frequency', 'monetary_value', 'tenure', 'lifetime_value']
order_id_col = "HoldingID"
customer_id_col = 'CustomerID'
monetary_value_col = 'OneTimeCost'
purchase_date_col = 'PurchasedDateTime'
model = 'non-contractual' # contractual, non-contractual
recurring_period = 'RecurringPeriod' # only for model = 'contractual'
recurring_cost = 'RecurringCost' # only for model = 'contractual'
lifespan = 9
rfm_segment_config = 'retail_banking' # telco, e-commerce
```

## Configuration

The application configuration is managed through a TOML file, which includes the following main sections:

1. **Application info and UX behaviour**: Branding, versioning, refresh, caching, and optional chat controls.
2. **Interaction History (`[ih]`)**: Raw data extraction and preprocessing settings.
3. **Product Holdings (`[holdings]`)**: Product-holdings extraction and preprocessing settings for CLV analysis.
4. **Metrics (`[metrics]`)**: Definitions of the metric families the dashboard calculates.
5. **Reports (`[reports]`)**: Configurable visual reports built from those metrics.
6. **Variants and Chat (`[variants]`, `[chat_with_data]`)**: Demo-mode metadata and AI prompt context.

---

### Config Authoring and Validation

The app includes two authoring experiences for TOML configuration:

- **Configuration Editor** works without optional AI dependencies. It edits the current config, shows Config Health,
  Review Progress, and report-level validation summaries, and blocks Apply/Download when validation errors remain.
- **AI Configuration Studio** is optional and requires the `ai` extra. It builds preprocessing from an uploaded sample,
  lets users choose approved fields and sample-value sharing, and shows an AI Privacy Summary before generation.

AI-generated config is never applied directly. The Studio holds generated sections as a pending draft, shows a reviewable
diff for metrics, reports, and variants, and lets users keep or reject generated metrics and reports. Report refresh and
AI Repair follow the same review-first pattern: the AI proposal is parsed, validated, shown as a diff, and only applied
after explicit approval. Final export stays disabled while blocking validation errors remain.

Validation checks include required Interaction History fields, runtime field availability, metric field references,
metric and report `group_by` compatibility, visual report-builder rules, and report references to metrics and scores.

---

#### Copyright Information

These settings provide versioning and release information about the application.

- **name**: The name of the application for copyright purposes.
- **version**: The current version of the application. For example, "0.1" indicates the initial version.
- **version_date**: The date of the current version release, formatted as YYYY-MM-DD.

---

#### User Experience (UX) Settings

These settings control the behavior of the application's user interface.

- **refresh_dashboard**: A boolean-like string that indicates whether the dashboard should automatically refresh.
  Possible values are "true" or "false".
- **refresh_interval**: The time interval (in milliseconds) for refreshing the dashboard automatically. The default
  template value is 180000, which equals 3 minutes.
- **data_cache_hours**: Cache processed data for N hours.
- **chat_with_data**: true/false - enable "Chat with your data" option.

---

#### Data load and pre-processing Settings

These settings define how input data files (usually Interaction History exports) are processed and recognized by the
application.

##### Interaction History

- **file_type**: The expected type of input data files. The default template setting is "pega_ds_export"; "parquet" is
  also supported for raw Interaction History.
- **file_pattern**: A glob pattern used to locate data files within the directory structure. The default template uses
  "**/*.zip" for Pega dataset exports.
- **ih_group_pattern**: A regular expression pattern used to extract date or identifier information from file names.
  The default template captures the date from Pega export file names. Data from files will be grouped before processing.
- **streaming**: Process the polars query in batches to handle larger-than-memory data. If set to False (default), the
  entire query is processed in a single batch. Should be changed to `true` if dataset files are larger than few GBs.
- **background**: Run the polars query in the background and return execution. Currently, all initial load frames are
  lazy frames, collected asynchronously.
- **hive_partitioning**: Expect data to be partitioned.
- **extensions**: Preprocessing rules applied before metric calculation.

The extensions section in the configuration file defines custom operations to manipulate and filter input data. These
operations are essential for optimizing performance and tailoring the data analysis to meet specific business needs. By
using these extensions, the application can efficiently preprocess data, enhancing its analytical capabilities and
providing insights that align with business objectives.

- **filter** option applies a global filter across the entire data load process. It is designed to improve performance
  by limiting data to only relevant records before further processing. The filter is constructed using specific
  conditions that must be met for each data record:
- **columns** option allows for the creation of new derived columns based on existing data. These transformations help
  in categorizing and labeling data for more accessible analysis and reporting.
- **default_values**: Default values for columns with empty/null cells.

##### Product Holdings

- **file_type**: The expected type of input data files. The default setting is "pega_ds_export". Product Holdings also
  supports "parquet", "csv", and "xlsx" when the matching config is selected.
  XLSX loading uses Polars Excel support, so install a compatible Excel engine if your environment does not include one.
- **file_pattern**: A glob pattern used to locate data files within the directory structure. For example: "**/*.json",
  works for "pega_ds_export".
- **file_group_pattern**: A regular expression pattern used to extract date or identifier information from file names.
- **streaming**: Process the polars query in batches to handle larger-than-memory data. If set to False (default), the
  entire query is processed in a single batch. Should be changed to `true` if dataset files are larger than few GBs.
- **background**: Run the polars query in the background and return execution. Currently, all initial load frames are
  lazy frames, collected asynchronously.
- **hive_partitioning**: Expect data to be partitioned.
- **extensions**: Preprocessing rules applied before CLV metric calculation.

The extensions section in the configuration file defines custom operations to manipulate and filter input data. These
operations are essential for optimizing performance and tailoring the data analysis to meet specific business needs. By
using these extensions, the application can efficiently preprocess data, enhancing its analytical capabilities and
providing insights that align with business objectives.

- **filter** option applies a global filter across the entire data load process. It is designed to improve performance
  by limiting data to only relevant records before further processing. The filter is constructed using specific
  conditions that must be met for each data record:
- **columns** option allows for the creation of new derived columns based on existing data. These transformations help
  in categorizing and labeling data for more accessible analysis and reporting.
- **default_values**: Default values for columns with empty/null cells.

---

### Reports' configuration parameters

The `[reports]` section in the configuration file allows for the definition of various analytical reports. Each report
is configured to display specific metrics and visualizations based on the application's requirements. These
configurations can be added or modified without changing the underlying code, providing flexibility in reporting.

Each report in the configuration file shares a set of common properties that establish its metric, type, description,
grouping, and visual attributes. These properties provide a consistent structure for defining various reports and ensure
that data is presented effectively.

#### 1. Metric

Definition: The metric property specifies the key performance indicator or measurement that the report focuses on. This
could be a business-related metric or a machine learning score.
Examples:
engagement (used to track user interactions like click-through rates)
model_ml_scores (used to monitor machine learning model performance metrics such as ROC AUC or average precision score)

#### 2. Type

Definition: The type property defines the visual representation or chart type for the report. It determines how the data
is displayed to the user.
The supported report types are defined under the `type` property in the configuration file. The types include:

- **line**: Line (or bar) plots for time-series or trend analysis.
- **gauge**: KPI gauge charts for engagement and conversion rates.
- **bar_polar**: Polar bar charts for categorical data visualization.
- **treemap**: Treemaps for hierarchical data representation.
- **heatmap**: Heatmaps for showing data density or correlation.
- **scatter**: Scatter plot.
- **boxplot**, **histogram**, **funnel**: Descriptive analysis plots.
- **exposure**, **corr**, **model**, **rfm_density**: CLV-specific views.
- **generic**: Fallback plot constructor for available dimensions and scores.

#### 3. Description

Definition: The description provides a brief summary of the report's purpose and focus. It often includes business or
technical context.
Format: The description often follows a standardized prefix to indicate the report's context, such as [BIZ] for business
metrics or [ML] for machine learning scores.

#### 4. Group By

Definition: The group_by property lists the data dimensions or categories by which the report is aggregated. It defines
how data is segmented for analysis.
Examples:

- `['Issue', 'Group']` for grouping data by specific issues and groups
- `['Day', 'Channel', 'PlacementType']` for analyzing daily trends across multiple dimensions

#### 5. Visual Attributes

- **Color**: Determines how data segments are colored in the visualization, often linked to a particular metric or
  dimension.
- **Axis Labels (x, y)**: Defines which data dimensions are plotted on the horizontal (x) and vertical (y) axes, crucial
  for interpreting the data correctly.
- **Faceting (facet_row, facet_column)**: Splits the visualization into multiple panels based on categorical variables,
  allowing for detailed comparisons across segments.
- **Legend**: Some reports may specify whether to show or hide the legend, impacting how data categories are annotated
  within the visualization.

---

### Variants

The `[variants]` section provides metadata and contextual information about the dashboard configuration. It is designed
to offer insights into the specific setup or version of the dashboard, helping to identify its intended use or
customization for particular clients or scenarios. This section is informational and does not directly impact the
functionality of the dashboard.

#### Properties in Variants section

- **name**

    - **Definition**: The `name` property assigns a short identifier or code to the current configuration variant.

- **description**

    - **Definition**: The `description` property provides a narrative or context regarding the purpose or target
      audience of this dashboard configuration variant.

- **demo_mode**
    - **Definition**: The `demo_mode` property allows the app to load sample data from the `data` directory
      automatically. Demo files must be available locally.

---

### Chat With Data

The `[chat_with_data]` section configures LLM context for questions about the loaded data and visualizations beyond the
reports configured for the dashboard. The page requires the optional AI dependencies.

Provider, model, and LiteLLM endpoint settings are read from `value_dashboard/config/llm_config.toml`. Override the path with
`--llm_config` or `VALUE_DASHBOARD_LLM_CONFIG` if you want to keep a separate local file.

Use the `OPENAI_API_KEY` environment variable to provide credentials, or paste the key directly in the UI form:

```bash
export OPENAI_API_KEY="<<your_openai_api_key>>"
```

For local models, edit `value_dashboard/config/llm_config.toml`, for example set `model = "ollama_chat/llama3.1"`,
`api_base = "http://localhost:11434"`, `api_key_required = false`, and set `reasoning_effort = ""` /
`verbosity = ""`.

#### Properties in section

- **agent_prompt**: description will be used to describe the agent in the chat and to provide more context for the LLM
  about how to respond to queries.
- **metric_descriptions**: detailed dataset description for the LLM.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
