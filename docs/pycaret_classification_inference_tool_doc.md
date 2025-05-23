# PyCaret Classification Inference Tool Documentation

## Overview

The PyCaret Inference Tool provides a streamlined interface for using trained PyCaret classification models to make predictions on new data. This tool handles the entire inference workflow, from loading models and data to generating predictions and calculating performance metrics when ground truth is available. It's designed to work seamlessly with models created by the PyCaret Classification Tool.

## Key Features

- **Model Deployment**: Easily apply trained PyCaret models to new data
- **Performance Evaluation**: Calculate comprehensive metrics when ground truth is available
- **Robust Error Handling**: Gracefully manages common issues with model loading and prediction
- **Multiple Output Formats**: Saves predictions and metrics in easy-to-use CSV and JSON formats
- **Comparative Analysis**: Generates actual vs. predicted comparison when ground truth is available

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_path` | Path to input data file (CSV format) for inference |
| `model_path` | Path to the saved PyCaret model |

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
3. **Model Loading**: Loads the specified PyCaret model
4. **Prediction**: Applies the model to generate predictions
5. **Output Generation**: Saves predictions to CSV
6. **Metrics Calculation**: Calculates performance metrics if ground truth is available
7. **Results Saving**: Stores metrics and comparison files

## Output Files

When run successfully, the tool generates several files in the output directory:

| File | Description |
|------|-------------|
| `predictions.csv` | Model predictions for all input data |
| `pycaret_inference.log` | Detailed log of the inference process |
| `prediction_vs_actual.csv` | Comparison of predicted and actual values (if ground truth is available) |
| `confusion_matrix.csv` | Confusion matrix for classification results (if ground truth is available) |
| `metrics_report.csv` | Performance metrics in CSV format (if ground truth is available) |
| `metrics.json` | Performance metrics in JSON format (if ground truth is available) |

## Performance Metrics

When ground truth values are available, the tool calculates various metrics:

### For All Classification Problems
- **Accuracy**: Proportion of correct predictions

### For Binary Classification
- **Precision**: Positive predictive value (TP / (TP + FP))
- **Recall/Sensitivity**: True positive rate (TP / (TP + FN))
- **Specificity**: True negative rate (TN / (TN + FP))
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve (if probability scores are available)
- **Confusion Matrix Components**: TP, TN, FP, FN counts

### For Multiclass Classification
- **Precision (macro)**: Macro-averaged precision across all classes
- **Recall (macro)**: Macro-averaged recall across all classes
- **F1 Score (macro)**: Macro-averaged F1 score across all classes
- **AUC (macro)**: Macro-averaged AUC for multiclass problems (when available)

## Example Usage Scenarios

### Basic Inference

```python
result = pycaret_inference(
    input_path="new_data.csv",
    model_path="/models/customer_churn_model"
)
```

This applies a trained model to new data and saves predictions to the same directory as the input file.

### Inference with Performance Evaluation

```python
result = pycaret_inference(
    input_path="test_data.csv",
    model_path="/models/credit_risk_model",
    output_dir="/results/credit_risk",
    ground_truth_column="DefaultStatus",
    prediction_filename="credit_risk_predictions.csv"
)
```

This runs inference and calculates performance metrics by comparing predictions with actual values in the "DefaultStatus" column.

### Quiet Mode for Production

```python
result = pycaret_inference(
    input_path="production_data.csv",
    model_path="/models/production_model",
    output_dir="/results/batch_predictions",
    verbose=False
)
```

This runs inference with minimal logging, suitable for production batch processing.

## Return Value

The tool returns a comprehensive dictionary containing:

```python
{
    "status": "success",  # or "error" if something failed
    "predictions_path": "/path/to/predictions.csv",
    "log_file": "/path/to/pycaret_inference.log",
    "metrics": {  # Only if ground truth was available
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.89,
        "f1": 0.90,
        # other metrics depending on classification type
    },
    "metrics_df": {  # Same metrics in DataFrame record format
        "accuracy": 0.95,
        "precision": 0.92,
        # ...
    },
    "metrics_csv_path": "/path/to/metrics_report.csv",
    "metrics_json_path": "/path/to/metrics.json",
    "comparison_path": "/path/to/prediction_vs_actual.csv",
    "confusion_matrix_path": "/path/to/confusion_matrix.csv",
    "input_path": "/path/to/input.csv",
    "model_path": "/path/to/model",
    "fixed_model_path": "/path/to/model",  # Path used for actual loading
    "output_dir": "/path/to/output",
    "num_predictions": 1000,  # Number of predictions made
    "has_ground_truth": True  # Whether ground truth was available
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

## Best Practices

### Model Preparation
- Ensure the model was properly saved using PyCaret's `save_model()` function
- Models saved by the PyCaret Classification Tool are compatible with this inference tool
- When specifying the model path, you can include or exclude the `.pkl` extension

### Data Preparation
- Ensure the input data has the same features as the training data
- Feature names should match exactly (case-sensitive)
- Features should be in the same format (numeric, categorical, etc.)
- No need to include the target column unless you want to evaluate performance

### Performance Evaluation
- Include the ground truth column in your data to enable performance evaluation
- Ensure the ground truth column has the same format and classes as during training
- Review both the metrics and confusion matrix for a complete understanding of model performance
- Check the prediction_vs_actual.csv file to analyze specific cases where the model may be making errors

### File Paths
- Use absolute paths when possible to avoid path resolution issues
- If using relative paths, be aware they're relative to the current working directory
- The tool will attempt to handle common path issues like redundant extensions

## Troubleshooting

### Common Issues

#### Model Loading Errors
- **Issue**: "File not found error" when loading model
- **Solution**: Check if the model path is correct. The tool will list available files in the directory to help identify the correct path.

#### Missing Probability Scores
- **Issue**: AUC calculation fails or probability scores are not available
- **Solution**: Check if the model supports probability prediction. Some models in PyCaret don't support probability outputs.

#### Metric Calculation Failures
- **Issue**: Some metrics couldn't be calculated
- **Solution**: Ensure ground truth data is in the correct format. For binary classification, verify there are two distinct classes.

## Conclusion

The PyCaret Inference Tool provides a streamlined way to apply trained classification models to new data and evaluate their performance. By providing insights into model performance alongside predictions, it helps users understand and improve their classification models.
