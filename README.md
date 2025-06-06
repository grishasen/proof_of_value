# CDH Value Dashboard Application

## Overview

The CDH Value Dashboard Application is an open-source prototype designed to provide customizable metrics and insights
for linking marketing actions to business outcomes, such as customer lifetime value (CLV) and conversion rates. The
dashboard utilizes technologies like Streamlit, Polars, Plotly, and DuckDB for efficient data processing and interactive
visualizations, supporting decision-making through clear data visualization.

Go to [Wiki page](https://github.com/grishasen/proof_of_value/wiki) for additional info.

## Features

- **Data Import**: Upload and load data files (ZIP or Parquet) for analysis.
- **Interactive Dashboard**: Visualize data and apply filters to explore and analyze the data interactively.
- **Configuration**: Customize the application settings through a TOML file.

## How To Use

1. **Data Import**:
    - Navigate to the "Data Import" page.
    - Upload your data by selecting and loading ZIP or Parquet files.
    - **For demo**:
        - IH reporting: switch off `Import raw data` toggle. Upload JSON file available in `data` folder (unzip
          archive).
        - CLV analytics: import product holdings zip file directly.
    - Once the data is imported, it will be processed and prepared for visualization.

2. **Dashboard**:
    - Navigate to the "Dashboard" page.
    - View the visualized data and apply various filters to interactively explore and analyze the data.
    - Use "Chat with data" with own OpenAI key.

## Installation (from source)

To run the CDH Value Dashboard Application locally from source, follow these steps:

### Prerequisites

- Python 3.11
- Required Python libraries (can be installed via `requirements.txt`)

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/grishasen/proof_of_value.git
    cd proof_of_value
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Edit config file**:
   ```bash
   vi value_dashboard/config/config.toml
   ```

4. **Run the Application**:
   ```bash
   streamlit run vd_app.py
   ```
   or
   ```bash
   streamlit run vd_app.py -- --config=<config_file_path.toml>
   ```
## Installation (python package)

Install the package with PIP:
```bash
  pip install cdhdashboard
```

To verify that the package has been installed successfully, you can test your package by running. For example:

```bash
  cdhdashboard run  -- --config <config_file_path.toml>
```

## Using Poetry to Build and Install Wheel

Poetry is a robust dependency management and packaging tool for Python that simplifies the process of managing packages
and building distributions. It handles all aspects of a project, including dependency resolution, virtual environment
management, and packaging.

### Prerequisites

Ensure the following:

Poetry is installed on your system. You can install Poetry using the following command:

```bash
  curl -sSL https://install.python-poetry.org | python3 -
```

Your Python project is set up with a valid pyproject.toml file.

### Build a Wheel File

The wheel format is a built package that can be installed directly using pip, which makes it easier to distribute and
install Python projects.

To build the wheel file:

Ensure you are in your project directory.
Run the following Poetry command:

```bash
  cd proof_of_value
  poetry build
```

This will create two files in the dist/ directory:

A .tar.gz file, which is the source distribution.
A .whl file, which is the built wheel distribution.

### Install the Wheel File via Shell

After successfully building the wheel file, you can install it in any Python environment using pip. The installation can
be done either within a Poetry-managed virtual environment or a separate environment.
To install the wheel file locally, use pip with the following command:

```bash
  pip install dist/cdhdashboard-0.0.1-py3-none-any.whl
```

Installing the Wheel Globally or in Another Project
If you want to install the package globally or in another Python environment, simply copy the wheel file to that
environment and run the same pip installation command:

```bash
  pip install /path/to/cdhdashboard-0.0.1-py3-none-any.whl
```

### Verify the Installation

To verify that the package has been installed successfully, you can test your package by running. For example:

```bash
  cdhdashboard run  --server.maxUploadSize 2048 -- --config <config_file_path.toml>
```

If application starts as expected, your wheel installation was successful.

## File Structure

- **vd_app.py**: The main entry point of the application.
- **value_dashboard/pages/home.py**: Application description.
- **value_dashboard/pages/data_import.py**: Handles data import functionality.
- **value_dashboard/pages/dashboard.py**: Contains the dashboard for IH data visualization and interaction.
- **value_dashboard/pages/clv_analysis.py**: Contains the dashboard for Product Holdings data and CLV-related metrics.
- **value_dashboard/pages/chat_with_data.py**: Chat with your data (enagagement, conversion, experiment).
- **value_dashboard/pages/toml_editor.py**: Configuration page for customizing application settings.
- **value_dashboard/metrics/**: Calculation of various metrics supported by the application.
- **value_dashboard/pipeline/**: Data loading and processing steps.
- **value_dashboard/reports/**: Plots and data visualization functions.
- **value_dashboard/utils/**: Utility functions for configuration and Streamlit components.

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
group_by = ['Day', 'Month', 'Year', 'Quarter', 'Channel', 'ModelType', 'Issue', 'CustomerType']
filter = """"""
scores = ['ConversionRate', 'Revenue']
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
group_by = ['Day', 'Month', 'Year', 'Quarter', 'Channel', 'CustomerType', 'Placement', 'Issue', 'Group']
filter = """"""
scores = ['CTR', 'Lift']
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
group_by = ['Day', 'Month', 'Year', 'Quarter', 'Channel', 'CustomerType', 'Placement']
filter = """"""
use_t_digest = "true" #Use t-digest of probs to calculate roc_auc and average_precision or weighted average
scores = ['roc_auc', 'average_precision', 'personalization', 'novelty']
positive_model_response = ["Clicked"]
negative_model_response = ["Impression", "Pending"]
```

- **Descriptive**: Describes dataset, aggregating counts, sums, etc. across givven dimensions
    - ***Count***: Return the number of non-null elements in the column.
    - ***Sum***: Get sum value for column.
    - ***Mean***: Get mean value.
    - ***Median***: Get median (50-percentile) value using t-digest data structure and corresponding algorithm.
    - ***p75***: Get 75-percentile using t-digest data structure and corresponding algorithm.
    - ***p90***: Get 90-percentile using t-digest data structure and corresponding algorithm.
    - ***p95***: Get 95-percentile using t-digest data structure and corresponding algorithm.
    - ***Std***: Get standard deviation (Delta Degrees of Freedom = 1).
    - ***Var***: Get variance (Delta Degrees of Freedom = 1).
    - ***Skew***: Compute Bowley's Skewness (Quartile Coefficient of Skewness) of a data
      set. $Skewness = \frac{(Q_3 + Q_1 - 2Q_2)}{Q_3 - Q_1} = \frac{(p75 + p25 - 2*p50)}{p75 - p25}$. For symmetrical
      data, the skewness should be about
      zero. For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in
      the right tail of the distribution.

```toml
[metrics.descriptive]
group_by = ['Day', 'Month', 'Year', 'Quarter', 'Channel', 'CustomerType', 'Placement', 'Issue', 'Group', 'Outcome']
filter = """"""
columns = ['Outcome', 'Propensity', 'FinalPropensity', 'Priority']
scores = ['Count', 'Sum', 'Mean', 'Median', 'p75', 'p90', 'p95', 'Std', 'Var', 'Skew']
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
group_by = ['Year', 'Channel', 'CustomerType']
filter = """(pl.col("ModelControlGroup").is_in(["Test", "Control"]))"""
experiment_name = 'ExperimentName'
experiment_group = 'ExperimentGroup'
scores = ['z_score', 'z_p_val', 'g_stat', 'g_p_val', 'chi2_stat', 'chi2_p_val', 'g_odds_ratio_stat', 'g_odds_ratio_ci_low', 'g_odds_ratio_ci_high', 'chi2_odds_ratio_stat', 'chi2_odds_ratio_ci_low', 'chi2_odds_ratio_ci_high']
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
lifespan = 3
rfm_segment_config = { "Premium Customer" = ["334", "443", "444", "344", "434", "433", "343", "333"], "Repeat Customer" = ["244", "234", "232", "332", "143", "233", "243", "242"], "Top Spender" = ["424", "414", "144", "314", "324", "124", "224", "423", "413", "133", "323", "313", "134"], "At Risk Customer" = ["422", "223", "212", "122", "222", "132", "322", "312", "412", "123", "214"], "Inactive Customer" = ["411", "111", "113", "114", "112", "211", "311"]}
```

## Configuration

The application configuration is managed through a TOML file, which includes the following main sections:

1. **Application info and UX behaviour**: Branding, styling, layout
2. **IH (Data Extraction)**: Configuration for data extraction processes.
3. **Metrics (Provided Functionality)**: Definitions of various metrics that the dashboard supports.
4. **Reports (Configurable Reports)**: Definitions of reports that can be generated based on the metrics.

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
  value is 120000, which equals 2 minutes.
- **data_profiling**: true/false - show additional EDA page
- **data_cache_hours**: Cache processed data for N hours
- **chat_with_data**: true/false - enable "Chat with your data" option

---

#### Data load and pre-processing Settings

These settings define how input data files (usually Interaction History exports) are processed and recognized by the
application.

##### Interaction History

- **file_type**: The expected type of input data files. The default setting is "parquet", indicating that files should
  be in Apache Parquet format.
- **file_pattern**: A glob pattern used to locate data files within the directory structure. For example: "**/*
  .parquet", which recursively searches for all files with a .parquet extension.
- **ih_group_pattern**: A regular expression pattern used to extract date or identifier information from file names.
  E.g. "ih_(\\d{8})", which captures date in YYYYMMDD format. Data from files will be grouped before processing.
- **streaming**: Process the polars query in batches to handle larger-than-memory data. If set to False (default), the
  entire query is processed in a single batch. Should be changed to `true` if dataset files are larger than few GBs.
- **background**: Run the polars query in the background and return execution. Currently, all initial load frames are
  lazy frames, collected asynchronously.
- **hive_partitioning**: Expect data to be partitioned.
- **extensions**: A placeholder for any additional file handling extensions or configurations that might be added later.

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

- **file_type**: The expected type of input data files. The default setting is "pega_ds_export".
- **file_pattern**: A glob pattern used to locate data files within the directory structure. For example: "**/*.json",
  works for "pega_ds_export".
- **file_group_pattern**: A regular expression pattern used to extract date or identifier information from file names.
- **streaming**: Process the polars query in batches to handle larger-than-memory data. If set to False (default), the
  entire query is processed in a single batch. Should be changed to `true` if dataset files are larger than few GBs.
- **background**: Run the polars query in the background and return execution. Currently, all initial load frames are
  lazy frames, collected asynchronously.
- **hive_partitioning**: Expect data to be partitioned.
- **extensions**: A placeholder for any additional file handling extensions or configurations that might be added later.

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

#### 3. Type

Definition: The type property defines the visual representation or chart type for the report. It determines how the data
is displayed to the user.
The supported report types are defined under the `type` property in the configuration file. The types include:

- **line**: Line (or bar) plots for time-series or trend analysis.
- **bar_polar**: Polar bar charts for categorical data visualization.
- **treemap**: Treemaps for hierarchical data representation.
- **heatmap**: Heatmaps for showing data density or correlation.

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
- `['Day', 'Channel', 'CustomerType']` for analyzing daily trends across multiple dimensions

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

---

### Chat With Data

The `[chat_with_data]` section used to configure integration with a chatbot for questions on the data and visualizations
beyond ad-hoc reports and queries configured for dashboard.

Use `OPENAI_API_KEY` environment variable to set integration with ChatGPT, run
`export OPENAI_API_KEY="<<your_open_api_key>>"` before starting the application or paste key directly in the UI form.

As soon as this functionality is based on Pandas AI framework, refer
to [Pandas AI documentation](https://docs.getpanda.ai/v3/introduction) for deep-dive.

#### Properties in section

- **agent_prompt**: description will be used to describe the agent in the chat and to provide more context for the LLM
  about how to respond to queries.
- **metric_descriptions**: detailed dataset description for the LLM.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

