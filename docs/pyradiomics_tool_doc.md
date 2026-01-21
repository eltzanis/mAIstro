# PyRadiomics Feature Extraction Tool

## Overview

The PyRadiomics Feature Extraction Tool is a utility designed to extract quantitative features from medical images (CT, MRI, etc.) using the PyRadiomics library. This tool processes medical images and their corresponding segmentation masks to extract radiomics features, which are mathematical descriptions of image texture, shape, and intensity patterns that can be used for research, diagnostic, and predictive modeling purposes.

## Key Features

- **Multi-label Support**: Processes different anatomical regions or pathologies defined by different label values in segmentation masks
- **Flexible Image Processing**: Configurable preprocessing steps including normalization, resampling, and filtering
- **Comprehensive Feature Extraction**: Extracts various types of quantitative features (first-order statistics, shape metrics, texture features)
- **Parallel Processing**: Option to use multiple CPU cores for faster extraction
- **Target Value Integration**: Ability to incorporate clinical outcome data for machine learning
- **Detailed Logging**: Comprehensive logs of the extraction process

## Technical Background

Radiomics is the field of extracting numbers of quantitative features from medical images. These features can reveal patterns and characteristics that may not be visible to the human eye and can be used for:

- Tumor characterization
- Disease diagnosis and staging
- Treatment response prediction
- Survival analysis
- Radiogenomics (correlation with genomic data)

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `image_dir` | Directory containing the medical images (CT, MRI, etc.) |
| `mask_dir` | Directory containing the segmentation masks corresponding to the images |
| `output_dir` | Directory where extracted features will be saved |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `image_types` | Types of image preprocessing to apply (e.g., Original, Wavelet, LoG) | `["Original", "Wavelet"]` |
| `feature_classes` | Classes of features to extract (e.g., firstorder, shape, glcm) | All available classes |
| `specific_features` | Specific features to extract (if provided, only these features will be extracted) | None |
| `mask_labels` | Specific label values to extract features for | All non-zero labels found in masks |
| `image_pattern` | Filename pattern to match image files | `*.nii.gz` |
| `mask_pattern` | Filename pattern to match mask files | `*.nii.gz` |
| `normalize` | Whether to normalize the image intensity values | `True` |
| `bin_width` | Bin width for discretizing image intensities | `25.0` |
| `resample` | Whether to resample the image to isotropic voxels | `True` |
| `pixel_spacing` | Target pixel spacing for resampling [x, y, z] | `[1.0, 1.0, 1.0]` |
| `force_2d` | Whether to extract features slice by slice (2D) instead of 3D | `False` |
| `n_workers` | Number of parallel workers for processing images | `1` (serial processing) |
| `targets_csv` | Path to a CSV file containing target values for each subject | None |
| `id_column` | Name of the ID column in the targets CSV | First column in CSV |
| `target_column` | Name of the target column in the targets CSV | Second column in CSV |
| `id_pattern` | Regex pattern to extract subject ID from filenames | Uses filename without extension |

## External Clinical Data Integration

### Target CSV File

The tool provides support for integrating external clinical, genomic, or outcome data with the extracted radiomics features through the `targets_csv` parameter. This is particularly valuable for research aiming to correlate image-based features with clinical endpoints.

#### Structure of the Target CSV File

The target CSV file should contain at minimum:

1. **Subject Identifier Column**: Contains values that can be matched to the image/mask filenames
2. **Target Value Column**: Contains the clinical or outcome data to be associated with each subject

Example of a target CSV file:
```
PatientID,Survival_Days,Recurrence,Grade,Age,Gender
PT001,455,1,3,67,M
PT002,982,0,2,54,F
PT003,321,1,3,71,M
...
```

#### Matching Process

The tool matches subjects in the target CSV with image/mask pairs using the following process:

1. By default, it uses the first column of the CSV as the ID column and the second column as the target value
2. You can specify custom column names using `id_column` and `target_column` parameters
3. The tool will look for matches between the subject IDs in the CSV and those extracted from image/mask filenames
4. Subject IDs are extracted from filenames using:
   - The `id_pattern` parameter if specified (regex pattern with a capture group)
   - Otherwise, the filename without extension is used as the ID

#### How Target Values Are Incorporated

For each successfully matched subject:

1. The target value is added as an additional column in the output CSV files
2. This allows for direct correlation analysis or machine learning without needing separate data merging steps
3. If multiple target columns are needed, you can add them to your feature matrix in a post-processing step

### Use Cases for Target Integration

The target CSV can contain various types of clinical data:

- **Categorical data**: Disease subtypes, genetic mutations, response categories
- **Continuous data**: Survival time, lab values, quantitative measures
- **Binary outcomes**: Disease recurrence, treatment response (success/failure)
- **Temporal data**: Time-to-event information for survival analysis
- **Multi-dimensional data**: Multiple clinical variables for multivariate analysis

### Best Practices for Target CSV Files

1. **Data Cleaning**: Ensure the CSV file is clean, with no missing values in the ID column
2. **ID Consistency**: Make sure subject IDs in the CSV exactly match how they appear in filenames
3. **Data Types**: Include appropriate data types for statistical analysis
4. **Handling Missing Data**: Consider how to handle subjects with missing clinical data
5. **Multiple Targets**: If you need multiple target variables, include them all in the CSV file

### Example Target Integration

```python
result = pyradiomics_feature_extraction(
    image_dir="/path/to/images",
    mask_dir="/path/to/masks",
    output_dir="/path/to/output",
    targets_csv="/path/to/clinical_data.csv",
    id_column="PatientID",
    target_column="Survival_Days",
    id_pattern="PT(\\d+)"  # Extracts numeric ID from filenames like "PT001_scan.nii.gz"
)
```

This will extract radiomics features and incorporate survival data from the clinical CSV, matching patients by their ID.

## Available Image Types

- **Original**: No filter applied, uses original image
- **Wavelet**: Wavelet decomposition of the image
- **LoG** (Laplacian of Gaussian): Emphasizes edges at different scales
- **Exponential**: Exponential filter
- **Gradient**: Gradient filter
- **SquareRoot**: Square root filter
- **LBP2D**: Local Binary Pattern (2D)

## Available Feature Classes

- **firstorder**: Statistical features based on intensity histogram (mean, median, entropy, etc.)
- **shape**: 3D morphological features (volume, surface area, sphericity, etc.)
- **glcm**: Gray Level Co-occurrence Matrix features (texture features like contrast, correlation)
- **glrlm**: Gray Level Run Length Matrix features (run length, run percentage, etc.)
- **glszm**: Gray Level Size Zone Matrix features (size zone variability, etc.)
- **gldm**: Gray Level Dependence Matrix features (dependence entropy, etc.)
- **ngtdm**: Neighboring Gray Tone Difference Matrix features (coarseness, complexity, etc.)

## Output

The tool generates the following outputs:

1. **CSV Files**: One CSV file per segmentation label containing extracted features for each subject
2. **Parameter File**: YAML file documenting the parameters used for extraction
3. **Log File**: Detailed log of the extraction process

The CSV files include:
- Subject ID column
- Target value column (if provided)
- All extracted features, organized by feature type

## Example Use Cases

### Basic Feature Extraction

```python
result = pyradiomics_feature_extraction(
    image_dir="/path/to/images",
    mask_dir="/path/to/masks",
    output_dir="/path/to/output"
)
```

This extracts features from all images using default parameters.

### Targeted Feature Extraction with Clinical Data

```python
result = pyradiomics_feature_extraction(
    image_dir="/path/to/images",
    mask_dir="/path/to/masks",
    output_dir="/path/to/output",
    image_types=["Original", "Wavelet"],
    feature_classes=["firstorder", "shape", "glcm"],
    mask_labels=[1, 2],
    normalize=True,
    resample=True,
    targets_csv="/path/to/clinical_data.csv",
    id_column="PatientID",
    target_column="Survival"
)
```

This extracts a specific subset of features and incorporates clinical outcome data.

### High-Performance Parallel Processing

```python
result = pyradiomics_feature_extraction(
    image_dir="/path/to/images",
    mask_dir="/path/to/masks",
    output_dir="/path/to/output",
    n_workers=8,
    force_2d=True,
    bin_width=10.0
)
```

This uses 8 CPU cores for faster processing and extracts 2D features with finer intensity binning.

## Workflow Details

1. **Initialization**: The tool validates input directories and sets up logging
2. **Parameter Configuration**: It creates a parameter set for PyRadiomics based on user inputs
3. **File Matching**: Images and masks are matched by subject ID
4. **Label Identification**: Available segmentation labels are identified
5. **Feature Extraction**: For each subject and label:
   - Extracts a binary mask for the specific label
   - Applies configured preprocessing
   - Extracts all enabled features
6. **Result Compilation**: Features are organized by label and saved to CSV files
7. **Cleanup**: Temporary files are removed

## Best Practices

1. **Image Organization**: Ensure image and mask files use consistent naming
2. **Feature Selection**: Start with a smaller feature set before extracting all possible features
3. **Preprocessing**: Normalize and resample your images for more consistent results
4. **Computational Resources**: For large datasets, use parallel processing but ensure sufficient memory
5. **Quality Control**: Review logs for any warnings about specific images or labels

## Troubleshooting

- **No features extracted**: Verify that masks contain the expected label values
- **Missing subjects**: Check that image and mask filenames match appropriately
- **Memory errors**: Reduce the number of parallel workers or process images in batches
- **Slow processing**: Enable parallel processing with `n_workers` > 1 or reduce the number of image types

## Advanced Topics

### Custom Feature Selection

You can select specific features using the `specific_features` parameter, which allows precise control over which features are calculated, reducing computation time and focusing on relevant metrics.

### Image Type Customization

Different filters (specified via `image_types`) highlight different aspects of the images:
- **Wavelet**: Multi-resolution analysis useful for texture features
- **LoG**: Edge detection at different scales, helpful for boundary characterization
- **Gradient**: Emphasizes intensity transitions

### Integration with Machine Learning

The extracted features can be directly used for:
- Feature selection algorithms
- Predictive modeling
- Clustering and subtyping
- Correlation with clinical outcomes

## Conclusion

The PyRadiomics Feature Extraction Tool provides a comprehensive solution for extracting quantitative features from medical images. By understanding the various parameters and options, you can customize the extraction process to suit your specific research or clinical needs, enabling advanced quantitative analysis of medical imaging data.
