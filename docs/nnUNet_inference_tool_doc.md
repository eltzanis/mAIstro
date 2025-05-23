# nnUNet Inference Tool Documentation

## Overview

The nnUNet Inference Tool enables users to apply trained nnUNet segmentation models to new medical images. It uses the nnUNet framework to generate high-quality segmentation masks for various medical imaging tasks. This tool serves as the deployment component of the nnUNet workflow, allowing trained models to be used for practical clinical or research applications.

## What is nnUNet Inference?

Inference is the process of applying a trained neural network to new, unseen data to make predictions. In the context of nnUNet, inference involves:

1. Loading a trained segmentation model
2. Preprocessing new medical images to match the training data characteristics
3. Generating segmentation masks that identify anatomical structures or pathologies
4. Post-processing results to optimize the final segmentations

The nnUNet framework automatically handles all these steps using the same configurations that were determined during training.

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_folder` | Directory containing images to segment (with correct channel numbering, e.g., `case_0000.nii.gz`) |
| `output_folder` | Directory where segmentation results will be saved |
| `dataset_id` | Dataset ID used for training the model (will be zero-padded to 3 digits) |
| `configuration` | nnUNet configuration to use ('2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres') |

### Optional Parameters

#### Model Selection
| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_folder` | Path to the directory containing the trained model | None (auto-detected) |
| `results_dir` | Base directory containing nnUNet results (overrides environment variable) | None |
| `folds` | Comma-separated list of folds to use (e.g., '0,1,2,3,4' or 'all') | None (uses all folds) |
| `plans_identifier` | Plans identifier | 'nnUNetPlans' |
| `trainer` | Trainer class used for training | 'nnUNetTrainer' |
| `checkpoint` | Checkpoint name to use | 'checkpoint_final.pth' |

#### Inference Control
| Parameter | Description | Default |
|-----------|-------------|---------|
| `step_size` | Step size for sliding window prediction (0-1) | 0.5 |
| `disable_tta` | Disable test time augmentation (mirroring) | False |
| `save_probabilities` | Export predicted class probabilities | False |
| `continue_prediction` | Continue an aborted previous prediction | False |
| `device` | Device for inference ('cuda', 'cpu', 'mps') | None (auto-detect) |
| `verbose` | Enable verbose output | False |

#### Performance Optimization
| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_processes_preprocessing` | Number of processes for preprocessing | None (auto) |
| `num_processes_segmentation` | Number of processes for segmentation export | None (auto) |
| `num_parts` | Number of separate inference calls for parallelization | None |
| `part_id` | Which part of the parallel inference is this (0 to num_parts-1) | None |

#### Cascade Models
| Parameter | Description | Default |
|-----------|-------------|---------|
| `prev_stage_predictions` | Folder with predictions from previous stage | None |

## Understanding Input Data Requirements

### Image Format

Input images must follow these requirements:
- Format: NIfTI (.nii.gz or .nii)
- Naming: Each case should include channel indicators
  - Example: `patient001_0000.nii.gz` for channel 0, `patient001_0001.nii.gz` for channel 1
- Organization: All channels for one case must share the same identifier before the underscore

### Channel Numbering

- For multi-channel data (e.g., multi-modal MRI), each modality must be in a separate file
- Channel numbering must match the order used during training
- Example for a 4-channel MRI dataset:
  - `case1_0000.nii.gz` (T1)
  - `case1_0001.nii.gz` (T1ce)
  - `case1_0002.nii.gz` (T2)
  - `case1_0003.nii.gz` (FLAIR)

### Image Properties

- Images should have the same properties (dimensions, spacing, orientation) as the training data
- nnUNet will handle resampling if needed, but major differences might affect performance

## Output and Results

The tool returns a dictionary with:

```json
{
  "status": "success",
  "output_folder": "/path/to/output",
  "dataset_id": dataset_id,
  "configuration": "selected_configuration",
  "num_segmentations": X,
  "segmentation_files": ["/path/to/output/case1.nii.gz", ...]
}
```

Segmentations are saved in the output folder with the following characteristics:
- Format: NIfTI (.nii.gz)
- Content: Integer labels matching the training dataset
- Names: Matching the case identifiers from the input files (without channel suffix)

## Workflow Details

The inference process follows these steps:

1. **Model Loading**: The specified trained model is loaded from disk
2. **Preprocessing**: 
   - Input images are read and normalized 
   - Resampling to the required resolution
   - Cropping to remove unnecessary background
3. **Sliding Window Prediction**:
   - The model processes the image in overlapping patches
   - The step size parameter controls the overlap amount
4. **Test Time Augmentation** (unless disabled):
   - The model makes predictions on mirrored versions of the input
   - Results are averaged for improved accuracy
5. **Postprocessing**:
   - Predicted probabilities are converted to discrete labels
   - Results are resampled back to original resolution
   - Connected component analysis may be applied to remove small isolated predictions

## Key Features and Options

### Ensemble Prediction

The tool can use multiple models (from different cross-validation folds) for ensemble prediction:
- Specify multiple folds with the `folds` parameter
- Results from each model are averaged for more robust predictions
- Generally improves performance compared to single models

### Test Time Augmentation (TTA)

TTA involves:
- Applying mirroring augmentations during inference
- Averaging predictions across these augmentations
- Typically improves accuracy by 1-2% but increases inference time
- Can be disabled with `disable_tta=True` for faster prediction

### Probability Maps

Setting `save_probabilities=True` saves:
- Softmax probability maps for each class
- Useful for uncertainty estimation or further post-processing
- Required when planning to ensemble predictions from different configurations

### Cascade Models

For cascade models (3d_cascade_fullres):
1. First run inference with the 3d_lowres model
2. Then run the cascade model with `prev_stage_predictions` pointing to the lowres output

### Parallel Processing

For large datasets:
- Use `num_parts` and `part_id` to split inference across multiple machines
- Increase `num_processes_preprocessing` and `num_processes_segmentation` for multi-core processing

## Example Usage Scenarios

### Basic Inference on a GPU

```python
result = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/results",
    dataset_id=50,
    configuration="3d_fullres"
)
```

This runs inference using the trained 3D full-resolution model on Dataset050, using all available folds for ensemble prediction.

### Fast Inference (Speed Optimized)

```python
result = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/results",
    dataset_id=50,
    configuration="3d_fullres",
    folds="0",  # Use only one fold
    disable_tta=True,  # Disable test time augmentation
    step_size=0.75  # Larger step size (less overlap)
)
```

This configuration prioritizes speed over maximum accuracy.

### High-Quality Inference (Accuracy Optimized)

```python
result = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/results",
    dataset_id=50,
    configuration="3d_cascade_fullres",
    folds="all",  # Use all folds
    step_size=0.25  # Smaller step size (more overlap)
)
```

This configuration prioritizes segmentation quality over speed.

### Inference on CPU

```python
result = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/results",
    dataset_id=50,
    configuration="2d",  # 2D is typically faster on CPU
    device="cpu",
    disable_tta=True  # Disable test time augmentation for speed
)
```

This runs inference on CPU, suitable for environments without GPU acceleration.

### Running a Cascade Model

```python
# First run the lowres model
result_lowres = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/lowres_results",
    dataset_id=50,
    configuration="3d_lowres",
    save_probabilities=True  # Save probabilities for cascade
)

# Then run the cascade model
result_cascade = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/final_results",
    dataset_id=50,
    configuration="3d_cascade_fullres",
    prev_stage_predictions="/path/to/lowres_results"
)
```

This runs the two-stage cascade approach for maximum accuracy.

### Parallel Inference for Large Datasets

```python
# On machine 1
result_part0 = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/results",
    dataset_id=50,
    configuration="3d_fullres",
    num_parts=2,
    part_id=0
)

# On machine 2
result_part1 = nnunet_inference(
    input_folder="/path/to/test_images",
    output_folder="/path/to/results",
    dataset_id=50,
    configuration="3d_fullres",
    num_parts=2,
    part_id=1
)
```

This distributes inference across two machines for faster processing of large datasets.

## Best Practices

1. **Model Selection**:
   - Use ensemble prediction (multiple folds) for maximum accuracy
   - For datasets with small structures, consider cascade models
   - Balance speed vs. accuracy based on your requirements

2. **Hardware Utilization**:
   - Use GPU acceleration when available
   - For multi-GPU systems, distribute different cases across GPUs
   - Adjust preprocessing and segmentation processes based on available CPU cores

3. **Quality Control**:
   - Save probability maps for uncertainty assessment
   - Consider additional post-processing for specific applications
   - Validate results on a sample before processing large datasets

4. **Performance Optimization**:
   - For real-time applications, use 2D configuration with disabled TTA
   - For batch processing, use 3D configuration with ensemble prediction
   - Experiment with step_size to find optimal speed/quality balance

5. **Memory Management**:
   - For large images, use 3d_lowres or 2d configurations if memory is limited
   - Reduce step_size if out-of-memory errors occur
   - Consider parallel processing with num_parts for very large datasets

## Troubleshooting

- **Missing files error**: Ensure input images follow correct naming convention (_0000, _0001, etc.)
- **Resolution mismatch**: nnUNet handles resampling, but extreme differences might affect quality
- **Out of memory**: Try reducing step_size, using 2D configuration, or disabling TTA
- **Poor segmentation quality**: Ensure input images match the characteristics of the training data
- **Model not found**: Check the model_folder path or ensure correct dataset_id and configuration

### Support Resources

- nnUNet GitHub: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- Citation: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## Conclusion

The nnUNet Inference Tool provides a powerful and flexible way to apply state-of-the-art medical image segmentation to new data. By understanding the different parameters and configurations, users can optimize the inference process for their specific requirements, balancing speed, accuracy, and resource utilization.
