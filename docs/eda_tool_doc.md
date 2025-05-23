# Exploratory Data Analysis Tool Documentation

## Overview

The Exploratory Data Analysis (EDA) Tool is a comprehensive utility designed to analyze tabulated data (Excel or CSV files) and generate detailed insights, statistics, and visualizations. This tool automates the EDA process, providing a thorough examination of data structure, distributions, correlations, and relationships, helping users quickly understand their datasets without manual coding.

## Key Features

- **Comprehensive Data Profiling**: Analyzes data structure, types, and basic statistics
- **Automated Visualization Generation**: Creates distribution plots, correlation matrices, pairplots, and more
- **Statistical Analysis**: Calculates descriptive statistics and identifies outliers
- **Relationship Analysis**: Examines relationships between features and target variables
- **Time Series Analysis**: Detects and analyzes temporal patterns in datetime columns
- **Outlier Detection**: Identifies and visualizes outliers in numerical columns
- **Summary Reporting**: Generates detailed reports with key insights

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_path` | Path to input data file (CSV or Excel) |
| `output_dir` | Output directory where analysis results will be saved |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `sheet_name` | Sheet name for Excel files (ignored for CSV files) | `None` |
| `target_column` | Target column for analyzing relationships | `None` |
| `categorical_threshold` | Maximum unique values to consider a column categorical | `10` |
| `correlation_method` | Method for correlation calculation ('pearson', 'spearman', 'kendall') | `pearson` |
| `visualize_distributions` | Generate distribution plots for numerical columns | `True` |
| `visualize_correlations` | Generate correlation heatmap | `True` |
| `visualize_pairplot` | Generate pairplot for numerical columns | `True` |
| `visualize_target_relationships` | Generate plots showing relationships with target column | `True` |
| `max_categories_pie` | Maximum categories to display in pie charts | `10` |
| `sampling_for_large_data` | Sample data if it's too large for visualization | `True` |
| `sample_size` | Number of rows to sample for large datasets | `10000` |
| `time_series_analysis` | Perform time series analysis if datetime columns are detected | `True` |
| `columns_to_exclude` | Comma-separated list of columns to exclude from analysis | `None` |
| `detect_outliers` | Detect and analyze outliers in numerical columns | `True` |
| `create_summary_report` | Create a comprehensive summary report in text format | `True` |
| `max_columns_for_correlation` | Maximum columns to include in correlation matrix | `100` |
| `max_columns_for_pairplot` | Maximum columns to include in pairplot | `10` |
| `create_figures` | Master switch to enable/disable all visualizations | `True` |

## How It Works

The EDA tool follows a systematic approach to data analysis:

1. **Data Loading and Validation**: Loads data from CSV or Excel files, with intelligent handling of encodings and data types
2. **Data Profiling**: Analyzes data structure, including column types, missing values, and basic statistics
3. **Column Type Detection**: Identifies numeric, categorical, and datetime columns
4. **Statistical Analysis**: Calculates descriptive statistics for each column
5. **Visualization Generation**: Creates various plots based on data types and relationships
6. **Outlier Detection**: Identifies outliers in numerical columns using IQR method
7. **Report Generation**: Compiles findings into a comprehensive report

## Generated Outputs

The tool generates a structured output directory containing:

### Data Profiling

- `data_profile.json`: Detailed profile of all columns, including types, statistics, and distributions
- `summary_statistics.txt`: Human-readable summary of key statistics for each column

### Visualizations

The tool creates a variety of visualizations in the `figures` directory:

#### Distribution Analysis
- Histograms with KDE for numerical columns
- Box plots for numerical columns
- Pie charts for categorical columns with few categories
- Bar charts for categorical columns with many categories
- Missing value visualization

#### Correlation Analysis
- Correlation heatmaps showing relationships between numeric variables
- Pairplots for multivariate analysis

#### Target Relationship Analysis
- Box plots showing distribution of numeric features by categorical target
- Scatter plots showing relationships between numeric features and numeric target
- Bar charts showing categorical feature distributions by categorical target
- Heatmaps showing associations between categorical features and target

#### Time Series Analysis
- Time series plots for numeric columns vs. datetime columns
- Count plots by time periods

#### Outlier Analysis
- Box plots highlighting outliers
- Histograms showing distribution with outliers

### Reporting

- `eda_summary_report.txt`: Comprehensive report summarizing all findings
- `eda_log_[timestamp].txt`: Detailed log of the analysis process

## Advanced Features

### Column Type Detection

The tool intelligently detects column types:
- **Numeric columns**: Columns with numeric data types
- **Categorical columns**: Non-numeric columns or numeric columns with few unique values
- **Datetime columns**: Columns containing date/time information

### Handling Large Datasets

For large datasets, the tool implements:
- **Data sampling**: Reduces dataset size for visualization while maintaining representativeness
- **Chunked processing**: Breaks large correlation matrices into manageable chunks
- **Memory optimization**: Carefully manages memory usage for large files

### Outlier Detection

The tool uses the Interquartile Range (IQR) method to detect outliers:
1. Calculates Q1 (25th percentile) and Q3 (75th percentile)
2. Computes IQR = Q3 - Q1
3. Defines outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
4. Provides detailed statistics on outliers for each column

### Time Series Analysis

When datetime columns are detected, the tool performs:
- **Trend analysis**: Visualizes how numeric variables change over time
- **Temporal distribution**: Shows count distribution across time periods
- **Periodicity**: Identifies daily, monthly, or yearly patterns when present

## Example Use Cases

### Basic Exploratory Data Analysis

```python
result = exploratory_data_analysis(
    input_path="data.csv",
    output_dir="eda_results"
)
```

This performs a complete analysis of data.csv with all default settings.

### Targeted Analysis with a Specific Focus

```python
result = exploratory_data_analysis(
    input_path="customer_data.xlsx",
    output_dir="customer_analysis",
    sheet_name="2023_Q2",
    target_column="ChurnStatus",
    categorical_threshold=15,
    correlation_method="spearman",
    columns_to_exclude="ID,CustomerCode,Notes",
    max_categories_pie=8
)
```

This analyzes a specific Excel sheet with focus on relationships with the "ChurnStatus" column.

### Large Dataset Analysis

```python
result = exploratory_data_analysis(
    input_path="large_dataset.csv",
    output_dir="large_dataset_analysis",
    sampling_for_large_data=True,
    sample_size=5000,
    max_columns_for_correlation=50,
    max_columns_for_pairplot=6,
    visualize_pairplot=False
)
```

This optimizes analysis for a large dataset by sampling and limiting the more computationally expensive visualizations.

### Minimal Analysis Without Visualizations

```python
result = exploratory_data_analysis(
    input_path="data.csv",
    output_dir="quick_profile",
    create_figures=False,
    create_summary_report=True
)
```

This performs a quick analysis without generating visualizations, only producing the data profile and summary report.

## Best Practices

1. **Start with Default Settings**: For initial exploration, use default settings to get a comprehensive overview
2. **Target-Focused Analysis**: Specify a target column to understand potential predictors for machine learning
3. **Handle Large Datasets**: Enable sampling for datasets with millions of rows
4. **Exclude Irrelevant Columns**: Use `columns_to_exclude` to remove ID columns or irrelevant features
5. **Review the Summary Report**: The summary report provides key insights and recommendations
6. **Follow Up on Outliers**: Investigate significant outliers identified by the tool
7. **Leverage Time Series Analysis**: For temporal data, review the time series visualizations for patterns

## Limitations and Considerations

1. **Memory Usage**: Very large datasets may require sampling to avoid memory issues
2. **Visualization Density**: Datasets with hundreds of columns will generate many visualizations
3. **Categorical Data**: Columns with very high cardinality may not display well in categorical visualizations
4. **Datetime Parsing**: Some complex datetime formats may require preprocessing before analysis
5. **Column Names**: Column names with special characters may be sanitized in output filenames

## Conclusion

The Exploratory Data Analysis Tool provides a comprehensive, automated approach to understanding your data. By generating detailed profiles, statistics, visualizations, and reports, it significantly accelerates the initial data exploration phase of any data science project. The tool is flexible enough to handle various data types and sizes, providing valuable insights that inform feature engineering, data cleaning, and modeling strategies.
