# PyCaret Classification Tool Documentation

## Overview

The PyCaret Classification Tool provides an automated machine learning solution for classification problems. It leverages the powerful PyCaret library to prepare data, train multiple classification models, tune hyperparameters, blend top-performing models, and generate comprehensive visualizations and interpretations. The tool is designed to accelerate the machine learning workflow from data to deployment, enabling both beginners and experienced data scientists to build high-quality classification models efficiently.

## Key Features

- **Automated Model Comparison**: Tests and compares multiple classification algorithms
- **Hyperparameter Tuning**: Automatically optimizes the top-performing models
- **Model Ensembling**: Creates a blended model from the best individual models
- **Visualization Suite**: Generates various plots for model performance and interpretation
- **Feature Importance**: Provides insights into which features drive predictions
- **Class Imbalance Handling**: Automatically addresses imbalanced datasets
- **Cross-Validation**: Ensures robust model evaluation with k-fold cross-validation
- **Preprocessing Options**: Offers various data preparation techniques
- **Model Persistence**: Saves trained models for future use
- **Interpretability**: Creates model explanations using techniques like SHAP

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_path` | Path to input data file (CSV format) |
| `output_dir` | Output directory where model and results will be saved |
| `target_column` | Name of the target column for classification |

### Optional Parameters

#### Experiment Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `experiment_name` | Name of the experiment | Auto-generated |
| `fold` | Number of cross-validation folds | 10 |
| `session_id` | Random seed for reproducibility | Random |
| `use_gpu` | Whether to use GPU for training (if available) | False |

#### Data Handling
| Parameter | Description | Default |
|-----------|-------------|---------|
| `fix_imbalance` | Whether to fix class imbalance | True |
| `data_split_stratify` | Whether to use stratified sampling for data splitting | True |
| `data_split_shuffle` | Whether to shuffle data before splitting | True |
| `preprocess` | Whether to apply preprocessing steps | True |
| `ignore_features` | Comma-separated list of features to ignore during training | None |
| `numeric_features` | Comma-separated list of numeric features | Auto-detected |
| `categorical_features` | Comma-separated list of categorical features | Auto-detected |
| `date_features` | Comma-separated list of date features | Auto-detected |

#### Feature Engineering
| Parameter | Description | Default |
|-----------|-------------|---------|
| `normalize` | Whether to normalize numeric features | False |
| `transformation` | Whether to apply transformation to numeric features | False |
| `pca` | Whether to apply PCA for dimensionality reduction | False |
| `pca_components` | Number of PCA components (float 0-1 or int > 1) | None |

#### Model Selection
| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_select` | Number of top models to select for blending | 3 |
| `include_models` | Comma-separated list of models to include in comparison | All models |
| `exclude_models` | Comma-separated list of models to exclude from comparison | None |

#### Testing
| Parameter | Description | Default |
|-----------|-------------|---------|
| `test_data_path` | Path to test/holdout data for independent evaluation | None |

#### Technical
| Parameter | Description | Default |
|-----------|-------------|---------|
| `ignore_gpu_errors` | Whether to ignore GPU-related errors and fall back to CPU | True |

## Available Classification Models

PyCaret can train and compare many classification algorithms, including:

- **Linear Models**: Logistic Regression, Linear Discriminant Analysis
- **Tree-based Models**: Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost
- **Support Vector Machines**: Linear SVC, SVC with various kernels
- **Neighbors-based**: K-Nearest Neighbors
- **Neural Networks**: Multi-layer Perceptron
- **Probabilistic Models**: Naive Bayes (Gaussian, Bernoulli, etc.)
- **Other Models**: Ridge Classifier, Quadratic Discriminant Analysis, AdaBoost, Gradient Boosting
- **Ensemble Techniques**: Voting Classifier, Stacking Classifier, Blended models

## Workflow and Outputs

The tool follows this workflow:

1. **Setup**: Prepares the data with preprocessing, feature engineering, and train-test splitting
2. **Model Comparison**: Trains and evaluates multiple models with cross-validation
3. **Model Tuning**: Performs hyperparameter tuning on the top N models
4. **Model Blending**: Creates an ensemble from the tuned models (if multiple are available)
5. **Visualization**: Generates performance plots and interpretability visualizations
6. **Evaluation**: Tests models on holdout data and compares results
7. **Persistence**: Saves models, predictions, and performance metrics

### Generated Output Files

The tool creates a structured directory with the following outputs:

```
output_dir/
├── model_comparison_results.csv      # Comparison of all model performances
├── model_summary.csv                 # Summary of all trained models with paths
├── [experiment_name]_final_model/    # Final model (best or blended)
└── models/                           # Directory containing all models
    ├── tuned_model_1/                # First tuned model
    │   ├── tuned_model_1_results.csv # Performance metrics
    │   ├── tuned_model_1/            # Saved model files
    │   ├── independent_eval_results_tuned_model_1.csv  # Test predictions
    │   └── plots/                    # Visualizations directory
    │       ├── auc.png               # ROC curve
    │       ├── confusion_matrix.png  # Confusion matrix
    │       ├── feature.png           # Feature importance
    │       ├── SHAP Summary.png      # SHAP feature importance
    │       └── [other plots]         # Various other visualizations
    ├── tuned_model_2/                # Second tuned model (similar structure)
    ├── tuned_model_3/                # Third tuned model (similar structure)
    └── blended_model/                # Blended ensemble model
        ├── blended_model_results.csv # Performance metrics
        ├── blended_model/            # Saved model files
        ├── independent_eval_results_blended_model.csv  # Test predictions
        └── plots/                    # Visualizations directory
```

### Generated Visualizations

The tool creates various visualizations for each model:

#### Performance Plots
- **AUC Plot**: ROC curve showing true positive vs. false positive rate
- **Confusion Matrix**: Visual representation of prediction errors
- **Precision-Recall Curve**: Precision vs. recall at different thresholds
- **Calibration Plot**: Reliability of predicted probabilities
- **Class Report**: Precision, recall, and F1-score for each class
- **Error Plot**: Analysis of error distribution
- **Learning Curve**: Model performance vs. training set size
- **Lift and Gain Charts**: Model effectiveness compared to random selection

#### Interpretability Plots
- **Feature Importance**: Relative importance of each feature
- **SHAP Summary**: Feature impact on model output using SHAP values
- **SHAP Reason Plots**: Explanation of individual predictions
- **Decision Boundary**: Visualization of model decision boundaries (for 2D projections)
- **Manifold Learning**: 2D projection of high-dimensional feature space


# 🧪 Example Usage Scenarios in Medical Imaging

---

### 🧠 Basic Classification for Tumor Grade Prediction

```python
result = pycaret_classification(
    input_path="radiomics_features.csv",
    output_dir="tumor_grade_classification",
    target_column="TumorGrade"
)
```

This trains models to classify tumor grades (e.g., low vs. high) from radiomics features extracted from MRI or CT scans using default settings including automatic preprocessing and cross-validation.

---

### 🚀 Custom Model Selection with GPU for Brain Tumor Diagnosis

```python
result = pycaret_classification(
    input_path="brain_mri_features.csv",
    output_dir="brain_tumor_diagnosis",
    target_column="Diagnosis",
    experiment_name="brain_tumor_gpu_models",
    include_models="lightgbm,xgboost,catboost,rf",
    use_gpu=True,
    session_id=42,
    n_select=3
)
```

This trains GPU-accelerated tree-based models to distinguish between glioma, meningioma, and healthy controls using features extracted from brain MRI.

---

### 🧬 Feature Engineering with PCA for Genomic Imaging Biomarkers

```python
result = pycaret_classification(
    input_path="multiomics_radiomics.csv",
    output_dir="genomic_biomarker_models",
    target_column="MutationStatus",
    normalize=True,
    transformation=True,
    pca=True,
    pca_components=0.90,
    ignore_features="PatientID,ScanDate"
)
```

This reduces the dimensionality of complex radiogenomic data to predict gene mutation status (e.g., EGFR mutation) using PCA.

---

### ⚖️ Handling Class Imbalance in Rare Cancer Detection

```python
result = pycaret_classification(
    input_path="rare_cancer_features.csv",
    output_dir="rare_cancer_classification",
    target_column="RareCancer",
    fix_imbalance=True,
    fold=5,
    data_split_stratify=True
)
```

This handles severe class imbalance (e.g., 1% positive class) in rare cancer detection by applying stratified 5-fold CV and class balancing techniques like SMOTE or class weights.

## Best Practices and Tips

### Data Preparation
- Ensure your data is clean and properly formatted before using the tool
- Include a unique identifier column if needed (mark it with `ignore_features`)
- Handle missing values in your data beforehand for better control
- Consider appropriate encoding for categorical variables with high cardinality

### Model Selection
- Start with `include_models=None` to try all available models
- For large datasets, consider using faster models first (e.g., `lightgbm`, `rf`)
- For complex relationships, include non-linear models (e.g., `xgboost`, `catboost`)
- If time permits, set a larger `n_select` value to try more models in the blending phase

### Performance Optimization
- Use `use_gpu=True` if you have compatible hardware and appropriate libraries installed
- For large datasets, ensure sufficient memory is available
- Set an appropriate `session_id` for reproducibility

### Interpretation
- Review the feature importance plots to understand key predictors
- Examine confusion matrices to identify which classes are harder to predict
- Use SHAP plots to understand model decisions for specific instances
- Check calibration plots to ensure probability estimates are reliable

### Deployment
- The tool saves models that can be loaded for future predictions
- Consider model size, inference speed, and interpretability for production deployment
- The blended model often offers the best performance but may be larger and slower

## Limitations and Considerations

- The tool requires the PyCaret library and its dependencies to be installed
- Large datasets may require significant computational resources
- GPU acceleration requires additional libraries (RAPIDS, cuML)
- Some models may not work well with certain data types or distributions
- The tool focuses on tabular data and may not be suitable for images, text, or time series without preprocessing
- Some visualizations may fail for certain model types but the tool will continue execution

## Conclusion

The PyCaret Classification Tool provides a comprehensive solution for building, evaluating, and interpreting classification models. By automating the machine learning workflow while offering extensive customization options, it enables both rapid prototyping and production-ready model development. The tool's output includes not only trained models but also detailed visualizations and performance metrics to support data-driven decision making.
