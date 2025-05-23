# PyCaret Regression Inference Tool Documentation

## Overview

The PyCaret Regression Inference Tool provides a streamlined interface for using trained PyCaret regression models to make predictions on new data. This tool handles the entire inference workflow, from loading models and data to generating predictions and calculating regression-specific performance metrics when ground truth is available. It's designed to work seamlessly with regression models created by the PyCaret Regression Tool.

## Key Features

- **Model Deployment**: Easily apply trained PyCaret regression models to new data
- **Performance Evaluation**: Calculate comprehensive regression metrics when ground truth is available
- **Residual Analysis**: Perform detailed analysis of prediction errors
- **Robust Error Handling**: Gracefully manages common issues with model loading and prediction
- **Multiple Output Formats**: Saves predictions and metrics in CSV and JSON formats

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_path` | Path to input data file (CSV format) for inference |
| `model_path` | Path to the saved PyCaret regression model |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Directory where prediction results will be saved | Same directory as input file |
| `ground_truth_column` | Name of the column containing ground truth values | None |
| `prediction_filename` | Name for the output prediction file | "predictions.csv" |
| `verbose` | Whether to print detailed logs | True |

## Workflow

The tool follows this workflow when making predictions:

1. **Initialization**: Sets up logging based on verbosity level
2. **Data Loading**: Reads the input CSV file 
3. **Model Loading**: Loads the specified PyCaret regression model
4. **Prediction**: Applies the model to generate predictions
5. **Output Generation**: Saves predictions to CSV
6. **Metrics Calculation**: Calculates regression performance metrics if ground truth is available
7. **Residual Analysis**: Analyzes prediction errors when ground truth is available
8. **Results Saving**: Stores metrics, comparison files, and residual analysis

## Output Files

When run successfully, the tool generates several files in the output directory:

| File | Description |
|------|-------------|
| `predictions.csv` | Model predictions for all input data |
| `pycaret_regression_inference.log` | Detailed log of the inference process |
| `prediction_vs_actual.csv` | Comparison of predicted and actual values with residuals (if ground truth is available) |
| `residual_analysis.csv` | Binned analysis of residuals showing distribution (if ground truth is available) |
| `metrics_report.csv` | Performance metrics in CSV format (if ground truth is available) |
| `metrics.json` | Performance metrics in JSON format (if ground truth is available) |

## Performance Metrics for Regression

When ground truth values are available, the tool calculates these regression-specific metrics:

### Core Regression Metrics
- **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same units as the target variable
- **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model (1 is perfect)
- **Explained Variance**: Similar to R² but focuses on explained variance

### Additional Metrics
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference (only calculated for non-zero actual values)
- **Mean Residual**: Average of prediction errors (indicates bias)
- **Median Residual**: Middle value of prediction errors
- **Min/Max Residual**: Extremes of prediction errors
- **Standard Deviation of Residuals**: Spread of prediction errors

## 🩺 Example Usage Scenarios for Regression Inference in Medical Applications

---

### 🧠 Basic Inference for Brain Volume Estimation

```python
result = pycaret_regression_inference(
    input_path="new_brain_features.csv",
    model_path="/models/brain_volume_model"
)
```

This applies a trained regression model to predict brain volume from new radiomics or deep learning features extracted from MRI scans.

---

### 🧪 Inference with Performance Evaluation for Dosimetry

```python
result = pycaret_regression_inference(
    input_path="test_patient_dvh_features.csv",
    model_path="/models/dose_prediction_model",
    output_dir="/results/dose_prediction_eval",
    ground_truth_column="MeanDose",
    prediction_filename="dose_predictions.csv"
)
```

This evaluates dose prediction performance by comparing model predictions against the actual mean dose values to critical structures.

---

### 🤫 Quiet Mode for Batch Tumor Size Forecasting

```python
result = pycaret_regression_inference(
    input_path="tumor_growth_input.csv",
    model_path="/models/tumor_growth_model",
    output_dir="/results/tumor_forecasts",
    verbose=False
)
```

This runs silent inference for forecasting tumor size progression in longitudinal imaging studies, suitable for automated pipelines.

---

## 📤 Return Value

The tool returns a comprehensive dictionary containing:

```python
{
    "status": "success",  # or "error" if something failed
    "predictions_path": "/path/to/predictions.csv",
    "log_file": "/path/to/pycaret_regression_inference.log",
    "metrics": {
        "mse": 123.45,
        "rmse": 11.11,
        "mae": 8.76,
        "r2": 0.89,
        "explained_variance": 0.90,
        "mape": 14.3,
        "mean_residual": -0.42
    },
    "metrics_df": {
        "mse": 123.45,
        "rmse": 11.11
    },
    "metrics_csv_path": "/path/to/metrics_report.csv",
    "metrics_json_path": "/path/to/metrics.json",
    "comparison_path": "/path/to/prediction_vs_actual.csv",
    "residuals_path": "/path/to/residual_analysis.csv",
    "input_path": "/path/to/input.csv",
    "model_path": "/path/to/model",
    "fixed_model_path": "/path/to/model",
    "output_dir": "/path/to/output",
    "num_predictions": 1000,
    "has_ground_truth": True
}
```

In case of errors, the dictionary will contain:

```python
{
    "status": "error",
    "error_message": "Detailed error description",
    "input_path": "/path/to/input.csv",
    "model_path": "/path/to/model"
}
```

## Understanding Residual Analysis

The residual analysis file provides valuable insights into model performance:

- **Residuals**: The difference between actual and predicted values (actual - predicted)
- **Absolute Residuals**: The absolute value of residuals (useful for understanding error magnitude)
- **Squared Residuals**: The squared value of residuals (what MSE is based on)
- **Percent Error**: The percentage difference between actual and predicted values

The `residual_analysis.csv` file bins residuals into groups to help visualize their distribution, showing:
- Count of residuals in each bin
- Mean value of residuals in each bin
- Minimum and maximum residual in each bin

A good regression model should have residuals that:
- Are centered around zero (mean residual near zero)
- Have a normal distribution
- Show no patterns when plotted against predicted values or features

## Best Practices

### Model Preparation
- Ensure the model was properly saved using PyCaret's `save_model()` function
- Models saved by the PyCaret Regression Tool are compatible with this inference tool
- When specifying the model path, you can include or exclude the `.pkl` extension

### Data Preparation
- Ensure the input data has the same features as the training data
- Feature names should match exactly (case-sensitive)
- Features should be in the same format (numeric, categorical, etc.)
- No need to include the target column unless you want to evaluate performance
- Make sure your features are scaled/transformed the same way as during training

### Performance Evaluation
- Include the ground truth column in your data to enable performance evaluation
- Analyze both the overall metrics and the distribution of residuals
- R² values close to 1 indicate a good fit, but also check RMSE/MAE for practical error magnitude
- Look for patterns in residuals to identify areas where the model might be biased

### File Paths
- Use absolute paths when possible to avoid path resolution issues
- If using relative paths, be aware they're relative to the current working directory
- The tool will attempt to handle common path issues like redundant extensions

## Conclusion

The PyCaret Regression Inference Tool provides a streamlined way to apply trained regression models to new data and evaluate their performance. The detailed residual analysis helps users understand not just how well their model performs overall, but also identify specific areas for improvement.
