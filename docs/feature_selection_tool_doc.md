# Feature Importance and Feature Selection Tool

## Overview

The Feature Importance Analysis Tool is a comprehensive utility for identifying and selecting the most relevant features in tabular datasets for machine learning tasks. This tool evaluates feature relevance using various statistical and machine learning methods, helping data scientists and analysts optimize model performance by focusing on the most informative variables.

## Key Features

- **Multiple Selection Methods**: Implements four feature selection algorithms (Random Forest, F-test, Mutual Information, Recursive Feature Elimination)
- **Automatic Task Detection**: Identifies whether the problem is classification or regression based on target characteristics
- **Categorical Handling**: Provides multiple strategies for encoding categorical variables
- **Visual Analysis**: Generates informative visualizations of feature importance and relationships
- **Dimensionality Reduction**: Creates PCA and t-SNE visualizations to explore feature space
- **Multi-threshold Selection**: Generates datasets with different numbers of top features
- **Interpretability Support**: Maintains mappings between original and encoded features

## Use Cases

- **Model Optimization**: Identify the most predictive features to improve model performance
- **Dimensionality Reduction**: Reduce computational requirements by eliminating irrelevant features
- **Data Understanding**: Gain insights into which variables drive your target variable
- **Feature Engineering Guidance**: Discover which features warrant further engineering effort
- **Model Explainability**: Make models more interpretable by focusing on key features

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_path` | Path to input data file (CSV format) |
| `output_dir` | Output directory where results will be saved |
| `target_column` | Name of the target column for prediction |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `task_type` | Type of machine learning task ('classification' or 'regression') | Auto-detected |
| `method` | Feature selection method ('rf', 'f_test', 'mutual_info', 'rfe') | 'rf' |
| `top_features` | Comma-separated list of top features counts to select | '10,50,100' |
| `encode_categorical` | Method to encode categorical features ('auto', 'onehot', 'label', 'target', 'none') | 'auto' |
| `max_onehot_cardinality` | Maximum unique values for one-hot encoding if encode_categorical='auto' | 10 |
| `create_plots` | Whether to generate visualization plots | True |
| `top_n_plot` | Number of top features to show in importance plots | 30 |

## Automatic Task Type Detection

When `task_type` is not specified, the tool automatically determines whether the problem is classification or regression using the following logic:

1. **Categorical Data Type**: If the target variable is a string, object, or categorical data type, it's classified as a classification task
2. **Unique Value Ratio**: If the number of unique values divided by the total number of samples is less than 5%, it's considered classification
3. **Absolute Unique Count**: If there are fewer than 10 distinct values in the target, it's considered classification
4. **Default**: Otherwise, it's considered a regression task

This heuristic approach works well for most datasets, but you can always explicitly specify `task_type` when needed, especially for:
- Regression problems with very few unique target values
- Classification problems with many classes (>10)
- Binary classification with numeric labels that have high unique ratios

## Feature Selection Methods

### Random Forest Importance (`rf`)
- **Description**: Uses the built-in feature importance from Random Forest models
- **Strengths**: Captures non-linear relationships, handles interactions between features
- **Best for**: General-purpose feature selection, complex relationships

### F-test (`f_test`)
- **Description**: Uses ANOVA F-value for classification or F-test for regression
- **Strengths**: Fast computation, statistical foundation
- **Best for**: Linear relationships, normally distributed data

### Mutual Information (`mutual_info`)
- **Description**: Measures how much information a feature provides about the target
- **Strengths**: Captures non-linear relationships, doesn't assume a particular distribution
- **Best for**: Capturing complex non-linear dependencies

### Recursive Feature Elimination (`rfe`)
- **Description**: Recursively removes features and evaluates model performance
- **Strengths**: Considers feature interactions during selection
- **Best for**: Finding the optimal feature subset for a specific model

## Categorical Encoding Options

### Auto (`auto`)
- Automatically selects between one-hot and label encoding based on cardinality
- Uses one-hot encoding for low-cardinality features (≤ `max_onehot_cardinality`)
- Uses label encoding for high-cardinality features

### One-hot Encoding (`onehot`)
- Creates binary columns for each category
- Best for nominal variables with no ordinal relationship
- May create too many features if cardinality is high

### Label Encoding (`label`)
- Converts categories to numerical values (0, 1, 2, etc.)
- Best for ordinal variables or when dimensionality must be preserved
- May introduce unintended ordinal relationships

### Target Encoding (`target`)
- Replaces categories with their mean target value
- Best for high-cardinality features
- Also creates frequency encoding as an additional feature

### None (`none`)
- Drops categorical features entirely
- Use when categorical features are known to be irrelevant or problematic

## Output Files

The tool generates several output files in the specified directory:

### Data Files
- **Top N features CSV files**: `top_X_features.csv` (where X is the number of features)
- **Feature importance rankings**: `feature_importance.csv`
- **Categorical encodings**: `categorical_encodings.json`
- **Feature metadata**: `top_X_features_metadata.json`

### Visualization Files
- **Feature importance bar plot**: `plots/feature_importance_plot.png`
- **Cumulative importance plot**: `plots/cumulative_importance_plot.png`
- **Feature correlation heatmap**: `plots/feature_correlation_heatmap.png`
- **PCA visualization**: `plots/pca_visualization_top_X.png`
- **t-SNE visualization**: `plots/tsne_visualization_top_X.png`

### Log Files
- **Analysis log**: `feature_importance_analysis.log`

## How It Works

### Data Preparation
1. The tool loads the dataset and separates features from the target variable
2. If task type is not specified, it automatically detects whether it's classification or regression
3. Categorical features are encoded according to the specified method
4. Missing values are handled appropriately (median for numeric, mode for categorical)

### Feature Selection Process
1. The specified feature selection method is applied to evaluate feature importance
2. Features are ranked by importance score
3. Multiple datasets are created, each containing the top N features as specified
4. For each feature subset, visualizations are generated to show relationships

### Visualization Generation
1. Bar plots show the importance score of top features
2. Cumulative importance plots identify thresholds where adding more features yields diminishing returns
3. Correlation heatmaps reveal relationships between top features
4. PCA and t-SNE plots visualize how well the selected features separate the target classes

## Example Usage Scenarios

### Basic Feature Selection

```python
result = feature_importance_analysis(
    input_path="customer_data.csv",
    output_dir="feature_analysis",
    target_column="churn",
    method="rf",
    top_features="10,20,50"
)
```

This performs Random Forest-based feature selection for predicting customer churn, generating datasets with the top 10, 20, and 50 features.

### Advanced Analysis with Different Methods

```python
result = feature_importance_analysis(
    input_path="medical_data.csv",
    output_dir="medical_features",
    target_column="disease_progression",
    task_type="regression",
    method="mutual_info",
    top_features="5,10,15,20",
    encode_categorical="target",
    create_plots=True,
    top_n_plot=15
)
```

This uses Mutual Information to select features for a medical regression problem, with target encoding for categorical variables.

### Quick Analysis Without Visualization

```python
result = feature_importance_analysis(
    input_path="sensor_data.csv",
    output_dir="sensor_analysis",
    target_column="equipment_failure",
    method="f_test",
    create_plots=False
)
```

This performs a fast F-test analysis without generating visualizations, useful for large datasets or quick exploration.

## Interpreting the Results

### Feature Importance Plot
- Shows the relative importance of each feature
- Longer bars indicate more important features
- Use this to identify your "heavy hitters" - features with outsized impact

### Cumulative Importance Plot
- Shows how quickly importance accumulates as you add features
- Helps identify the "sweet spot" for feature count
- Common thresholds (80%, 90%, 95%, 99%) are marked

### Feature Correlation Heatmap
- Reveals correlations between top features
- Highly correlated features may be redundant
- Look for clusters of correlated features that might represent the same underlying factor

### PCA and t-SNE Visualizations
- Show how well the selected features separate the target classes
- Clear clustering indicates good feature selection
- Overlapping classes may indicate that more or different features are needed

## Best Practices

1. **Start with Random Forest**: The 'rf' method provides a good baseline for most problems
2. **Compare Methods**: Try different methods and compare results for your specific dataset
3. **Evaluate Multiple Thresholds**: Use the different feature counts to find the optimal balance between model complexity and performance
4. **Check for Correlations**: Review the correlation heatmap to identify redundant features
5. **Experiment with Encoding**: Try different encoding strategies for categorical features
6. **Validate with Models**: Build models using the selected feature subsets to verify performance improvements

## Limitations

- **Computational Resources**: RFE and t-SNE can be computationally intensive for large datasets
- **Linear Assumptions**: F-test assumes linear relationships between features and target
- **Encoding Impact**: Different encoding methods can significantly affect feature importance results
- **Correlation vs. Causation**: Important features are statistically significant but don't necessarily imply causation

## Technical Details

### Random Forest Implementation
- Uses `RandomForestClassifier` or `RandomForestRegressor` with 100 trees
- Feature importance is derived from the mean decrease in impurity

### F-test Implementation
- Uses `f_classif` for classification problems
- Uses `f_regression` for regression problems
- Implemented via scikit-learn's `SelectKBest`

### Mutual Information Implementation
- Uses `mutual_info_classif` for classification problems
- Uses `mutual_info_regression` for regression problems
- Non-parametric and can capture non-linear relationships

### RFE Implementation
- Uses Random Forest as the base estimator
- Eliminates features recursively
- Step size of 0.1 (removes 10% of features in each iteration)

## Conclusion

The Feature Importance Analysis Tool provides a comprehensive approach to feature selection, combining multiple statistical and machine learning methods with informative visualizations. By identifying the most relevant features in your dataset, you can build more efficient, accurate, and interpretable models while gaining deeper insights into the factors driving your target variable.
