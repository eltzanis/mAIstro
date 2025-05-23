# nnUNet Training Tool Documentation

## Overview

The nnUNet Training Tool provides a streamlined interface for training state-of-the-art medical image segmentation models using the nnUNet framework. This tool handles both data preprocessing and model training, simplifying the process of developing accurate segmentation models for medical imaging applications.

## What is nnUNet?

nnUNet is a self-configuring method for deep learning-based medical image segmentation. It automatically adapts preprocessing, network architecture, training, and post-processing to the properties of any given dataset, achieving state-of-the-art performance across diverse medical segmentation tasks without manual intervention.

Key features of nnUNet include:
- Automatic configuration based on dataset properties
- Support for 2D and 3D segmentation
- Multi-resolution approaches for large images
- Ensemble learning capabilities through cross-validation

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `dataset_id` | Dataset ID number (will be zero-padded to 3 digits, e.g., 50 → Dataset050) |

### Optional Parameters

#### Configuration Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `configuration` | nnUNet configuration type ('2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres') | '3d_fullres' |
| `fold` | Cross-validation fold (0-4) or 'all' to train all folds | 'all' |
| `trainer` | Custom trainer class (e.g., 'nnUNetTrainer') | None (uses default) |
| `plans_identifier` | Custom plans identifier | None (uses 'nnUNetPlans') |

#### Training Resources
| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_gpus` | Number of GPUs to use for training | None (auto-detect) |
| `device` | Device to run on ('cuda', 'cpu', 'mps') | None (auto-detect) |
| `pretrained_weights` | Path to nnU-Net checkpoint for transfer learning | None |

#### Training Control
| Parameter | Description | Default |
|-----------|-------------|---------|
| `continue_training` | Continue training from latest checkpoint | False |
| `validation_only` | Only run validation (training must have finished) | False |
| `val_best` | Use checkpoint_best instead of checkpoint_final for validation | False |
| `disable_checkpointing` | Disable saving checkpoints during training | False |
| `npz` | Save softmax predictions from validation as npz files | False |
| `use_compressed` | Use compressed data without decompression | False |

#### Preprocessing Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `verify_dataset_integrity` | Verify dataset integrity during preprocessing | True |
| `no_preprocessing` | Skip preprocessing step (use if data is already preprocessed) | False |

## Understanding nnUNet Configurations

The tool supports four different configurations:

1. **2d**: Uses 2D U-Net architecture. Best for:
   - Slice-by-slice segmentation
   - Datasets with highly anisotropic voxel spacing
   - Limited GPU memory scenarios

2. **3d_fullres**: Standard 3D U-Net at full resolution. Best for:
   - Most 3D medical segmentation tasks
   - Datasets with isotropic or near-isotropic voxel spacing
   - Capturing 3D spatial context

3. **3d_lowres**: 3D U-Net at lower resolution. Best for:
   - Very large images that don't fit in GPU memory at full resolution
   - Initial coarse segmentation

4. **3d_cascade_fullres**: Two-stage approach combining 3d_lowres and 3d_fullres. Best for:
   - Very large images with small structures
   - Cases requiring both global context and fine details
   - Highest accuracy requirements (at the cost of training time)

## Dataset Preparation

Before using this tool, you must prepare your dataset according to the nnUNet requirements:

1. **Dataset Folder Structure**:
   ```
   nnUNet_raw/Dataset[DATASET_ID]/
   ├── imagesTr/        # Training images
   ├── labelsTr/        # Training labels
   ├── imagesTs/        # Test images (optional)
   └── dataset.json     # Dataset metadata
   ```

2. **Naming Convention**:
   - Images: `CASE_IDENTIFIER_XXXX.nii.gz` where XXXX is a 4-digit modality identifier
   - Labels: `CASE_IDENTIFIER.nii.gz`

3. **dataset.json Format**:
   ```json
   {
     "name": "Dataset Name",
     "description": "Dataset Description",
     "reference": "Paper reference",
     "licence": "Dataset license",
     "modality": {
       "0": "Modality 1",
       "1": "Modality 2"
     },
     "labels": {
       "0": "background",
       "1": "Label 1",
       "2": "Label 2"
     },
     "numTraining": number_of_training_cases,
     "numTest": number_of_test_cases
   }
   ```

## Workflow Details

### Preprocessing Step

The preprocessing step includes:
1. Analyzing dataset properties (spacing, intensity, etc.)
2. Creating optimal plans for network configuration
3. Resampling images to appropriate resolutions
4. Intensity normalization
5. Creating cropped datasets optimized for training

The tool stores preprocessed data in the nnUNet_preprocessed directory.

### Training Step

Training involves:
1. Creating the neural network based on the determined configuration
2. Running the training loop with data augmentation
3. Monitoring validation metrics
4. Saving model checkpoints
5. Performing final validation

Training progress is tracked with metrics like Dice score, IoU, and validation loss.

## Output and Results

The tool returns:

```json
{
  "status": "success",
  "model_path": "/path/to/model",
  "dataset_id": dataset_id,
  "configuration": "selected_configuration",
  "fold": "selected_fold",
  "metrics": {
    "dice": 0.XX,
    "iou": 0.XX,
    "validation_loss": X.XX
  }
}
```

Models are saved in the `nnUNet_results` environment variable directory with the following structure:
```
nnUNet_results/
└── Dataset[DATASET_ID]/
    └── nnUNetv2_[CONFIGURATION]/
        └── fold_[FOLD]/
            ├── checkpoint_final.pth
            ├── checkpoint_best.pth
            └── ...
```

## Metrics Interpretation

- **Dice Score**: Ranges from 0 to 1, with 1 being perfect overlap between prediction and ground truth. 
  
- **IoU (Intersection over Union)**: Also known as Jaccard index, ranges from 0 to 1. Typically lower than Dice score for the same prediction.
  
- **Validation Loss**: Lower values indicate better model fit.

## Example Usage Scenarios

### Basic Training on a Single GPU

```python
result = nnunet_training(
    dataset_id=50,
    configuration="3d_fullres",
    fold="0"
)
```

This trains a 3D full-resolution model on Dataset050 using fold 0 of cross-validation.

### Training All Folds with Data Verification

```python
result = nnunet_training(
    dataset_id=50,
    configuration="3d_fullres",
    fold="all",
    verify_dataset_integrity=True
)
```

This trains models for all 5 folds of cross-validation, verifying dataset integrity during preprocessing.

### Training a 2D Model on CPU

```python
result = nnunet_training(
    dataset_id=50,
    configuration="2d",
    device="cpu",
    fold="0"
)
```

This trains a 2D model on CPU, useful for environments without GPU access.

### Transfer Learning with Pretrained Weights

```python
result = nnunet_training(
    dataset_id=50,
    configuration="3d_fullres",
    pretrained_weights="/path/to/pretrained_model.pth",
    fold="0"
)
```

This initializes the network with weights from a pretrained model before training.

### Continue Training from Previous Checkpoint

```python
result = nnunet_training(
    dataset_id=50,
    configuration="3d_fullres",
    fold="0",
    continue_training=True
)
```

This continues training from the latest saved checkpoint.

### Training a Cascade Model

```python
# First train the lowres model
result_lowres = nnunet_training(
    dataset_id=50,
    configuration="3d_lowres",
    fold="0"
)

# Then train the cascade model
result_cascade = nnunet_training(
    dataset_id=50,
    configuration="3d_cascade_fullres",
    fold="0"
)
```

This trains a two-stage cascade model, which uses the lowres model's output as additional input for the fullres model.

## Best Practices

1. **Configuration Selection**:
   - Start with `3d_fullres` for most 3D medical imaging tasks
   - Use `2d` for highly anisotropic data or limited GPU memory
   - Try `3d_cascade_fullres` if you need maximum accuracy and have time/resources
   
2. **Cross-validation**:
   - Use `fold="all"` for final model ensembles
   - Use individual folds (`fold="0"`) for faster prototyping
   
3. **Runtime Expectations**:
   - Preprocessing: 1-8 hours depending on dataset size
   - Training per fold: 12-72 hours depending on configuration and hardware

4. **Memory Management**:
   - If encountering memory errors, try reducing batch size by modifying nnUNet plans
   - For very large datasets, use `use_compressed=True` to save disk space

## Troubleshooting

- **Preprocessing Fails**: Check dataset format, dataset.json structure, and file naming
- **Out of Memory Errors**: Try a 2D configuration or reduce patch size in plans
- **Training Diverges**: Check for data inconsistencies or try different learning rates
- **Poor Performance**: Ensure proper intensity normalization and consider ensemble models

### Support Resources

- nnUNet GitHub: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- Citation: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## Conclusion

The nnUNet Training Tool provides a streamlined way to leverage the powerful nnUNet framework for medical image segmentation. By automating preprocessing and training with sensible defaults while allowing customization, it enables researchers and practitioners to efficiently develop high-performance segmentation models for diverse medical imaging applications.
