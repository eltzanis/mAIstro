# TotalSegmentator Tool Documentation

## Overview

The TotalSegmentator Tool provides a powerful interface for automated segmentation of anatomical structures in CT and MR images. Built on the TotalSegmentator deep learning framework, this tool can identify and label dozens of anatomical structures in medical images with a single command. It supports various segmentation tasks, output formats, and performance optimization options.

## What is TotalSegmentator?

TotalSegmentator is a deep learning-based medical image segmentation tool that can automatically segment multiple anatomical structures in CT and MR images.

Key capabilities include:
- Full-body segmentation of major organs, bones, muscles, and vessels
- Specialized segmentation for specific anatomical regions (brain, liver, heart, etc.)
- Support for both CT and MR imaging modalities
- Multiple resolution options for balancing speed and accuracy
- Statistical analysis of segmented structures (volume, intensity, radiomics)

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `input_path` | Path to input CT/MR NIFTI image or folder of DICOM slices |
| `output_dir` | Output directory where segmentation masks will be saved |

### Optional Parameters

#### Output Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_type` | Format of segmentations: 'nifti' or 'dicom' | 'nifti' |
| `multilabel` | Save one multilabel image for all classes instead of separate binary masks | False |
| `preview` | Generate a PNG preview of segmentation | False |
| `skip_saving` | Skip saving segmentations for faster runtime if only interested in statistics | False |

#### Task Selection
| Parameter | Description | Default |
|-----------|-------------|---------|
| `task` | Segmentation task to perform (see Available Tasks section) | 'total' |
| `roi_subset` | Subset of classes to segment (comma-separated list) | None |
| `roi_subset_robust` | Like roi_subset but uses a more robust model | None |

#### Performance Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `fast` | Use faster lower resolution model (3mm) | False |
| `fastest` | Use even faster lower resolution model (6mm) | False |
| `nr_threads_resampling` | Number of threads for resampling | Auto |
| `nr_threads_saving` | Number of threads for saving segmentations | Auto |
| `device` | Device for inference: 'gpu', 'cpu', or 'mps' (Apple Silicon) | 'gpu' |
| `force_split` | Process image in 3 chunks for less memory consumption | False |
| `body_seg` | Do initial rough body segmentation and crop to body region | False |

#### Analysis Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `statistics` | Calculate volume (mm³) and mean intensity | False |
| `radiomics` | Calculate radiomics features (requires pyradiomics) | False |
| `stats_include_incomplete` | Calculate statistics for ROIs cut off at image boundaries | False |

#### Other Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `crop_path` | Custom path to masks used for cropping | None |
| `no_derived_masks` | Do not create derived masks (e.g., skin from body mask) | False |
| `nora_tag` | Tag in NORA as mask (pass NORA project ID) | None |
| `quiet` | Suppress console output | False |
| `verbose` | Enable verbose output | False |
| `license_number` | License number for TotalSegmentator | None |

## Available Tasks and Anatomical Structures

TotalSegmentator offers several task options optimized for different use cases:

### Main Tasks

- **total**: Full-body segmentation of 104 anatomical structures in CT scans
- **total_mr**: Full-body segmentation optimized for MR images

### Specialized Tasks

#### CT-specific Tasks
- **lung_vessels**: Detailed segmentation of lung vessels
- **body**: Body and body parts segmentation
- **cerebral_bleed**: Intracranial hemorrhage segmentation
- **hip_implant**: Hip implant detection and segmentation
- **pleural_pericard_effusion**: Pleural and pericardial effusion segmentation
- **head_glands_cavities**: Head glands and cavities segmentation
- **head_muscles**: Head muscles segmentation
- **headneck_bones_vessels**: Head and neck bones and vessels segmentation
- **headneck_muscles**: Head and neck muscles segmentation
- **liver_vessels**: Liver vessel segmentation
- **oculomotor_muscles**: Eye muscles segmentation
- **lung_nodules**: Lung nodule detection and segmentation
- **kidney_cysts**: Kidney cyst segmentation
- **breasts**: Breast tissue segmentation
- **liver_segments**: Couinaud liver segments segmentation
- **heartchambers_highres**: High-resolution heart chamber segmentation
- **appendicular_bones**: Detailed bone segmentation
- **tissue_types**: Tissue type classification
- **tissue_4_types**: Four basic tissue types classification
- **brain_structures**: Brain structure segmentation
- **vertebrae_body**: Detailed vertebrae segmentation
- **face**: Facial structure segmentation
- **thigh_shoulder_muscles**: Thigh and shoulder muscles segmentation
- **coronary_arteries**: Coronary artery segmentation

#### MR-specific Tasks
- **body_mr**: Body segmentation for MR images
- **vertebrae_mr**: Vertebrae segmentation for MR images
- **liver_segments_mr**: Liver segments segmentation for MR images
- **tissue_types_mr**: Tissue types segmentation for MR images
- **face_mr**: Facial structure segmentation for MR images
- **thigh_shoulder_muscles_mr**: Thigh and shoulder muscle segmentation for MR images

### Segmented Structures

The main "total" task includes segmentation of 104 anatomical structures, including:

- **Organs**: Brain, lungs, heart, liver, spleen, pancreas, kidneys, etc.
- **Bones**: Skull, spine, ribs, pelvis, femur, etc.
- **Vessels**: Aorta, pulmonary artery/vein, etc.
- **Muscles**: Various muscle groups
- **Other structures**: Trachea, esophagus, urinary bladder, etc.

Specific tasks focus on subsets of these structures or additional specialized structures.

## Output Results

The tool returns a comprehensive result dictionary:

```json
{
  "status": "success",
  "output_dir": "/path/to/output",
  "command_output": "Command line output text...",
  "segmentation_files": [
    "/path/to/output/lung.nii.gz",
    "/path/to/output/liver.nii.gz",
    "..."
  ],
  "statistics": {
    "lung": {
      "volume_mm3": 4500.0,
      "mean_intensity": -600.0
    },
    "..."
  },
  "radiomics": {
    "lung": {
      "original_firstorder_Mean": 125.4,
      "original_glcm_Correlation": 0.6,
      "..."
    },
    "..."
  },
  "preview_file": "/path/to/output/preview.png",
  "task_info": {
    "task": "total",
    "is_mr_task": false,
    "image_type": "CT"
  }
}
```

### Segmentation Files

By default, the tool saves individual binary masks for each segmented structure in NIFTI format (.nii.gz). Each file contains a binary mask where 1 indicates the presence of the structure and 0 indicates background.

If the `multilabel` option is enabled, a single multilabel image is saved instead, where each voxel value corresponds to a specific anatomical structure.

### Statistics

When the `statistics` option is enabled, the tool calculates and saves:
- Volume in cubic millimeters (mm³)
- Mean intensity (Hounsfield units for CT)

### Radiomics Features

When the `radiomics` option is enabled, the tool calculates various radiomics features using the PyRadiomics library, including:
- First-order statistics (mean, median, entropy, etc.)
- Shape features (volume, surface area, sphericity, etc.)
- Texture features (GLCM, GLRLM, GLSZM, etc.)

## Example Usage Scenarios

### Basic CT Segmentation

```python
result = totalsegmentator(
    input_path="/path/to/ct_scan.nii.gz",
    output_dir="/path/to/output"
)
```

This performs full-body segmentation of 104 anatomical structures in a CT scan using the default settings.

### MR Image Segmentation

```python
result = totalsegmentator(
    input_path="/path/to/mr_scan.nii.gz",
    output_dir="/path/to/output",
    task="total_mr"
)
```

This performs segmentation optimized for MR images.

### Fast Segmentation for Quick Results

```python
result = totalsegmentator(
    input_path="/path/to/ct_scan.nii.gz",
    output_dir="/path/to/output",
    fast=True,
    preview=True
)
```

This uses a faster, lower-resolution model (3mm) and generates a preview image.

### Segmenting Specific Structures

```python
result = totalsegmentator(
    input_path="/path/to/ct_scan.nii.gz",
    output_dir="/path/to/output",
    roi_subset="liver,spleen,stomach,pancreas",
    statistics=True
)
```

This segments only the specified organs and calculates volume and intensity statistics.

### Advanced Analysis with Radiomics

```python
result = totalsegmentator(
    input_path="/path/to/ct_scan.nii.gz",
    output_dir="/path/to/output",
    task="lung_vessels",
    statistics=True,
    radiomics=True
)
```

This focuses on lung vessel segmentation and performs both basic statistics and advanced radiomics analysis.

### Memory-Efficient Processing for Large Images

```python
result = totalsegmentator(
    input_path="/path/to/large_ct.nii.gz",
    output_dir="/path/to/output",
    force_split=True,
    body_seg=True
)
```

This uses techniques to reduce memory consumption when processing large images.


## Best Practices

1. **Task Selection**:
   - Use the correct task for your imaging modality (CT vs. MR)
   - For targeted analysis, use specialized tasks rather than the full "total" task
   - Use "_mr" tasks for MR images to ensure proper segmentation

2. **Resolution Trade-offs**:
   - Default resolution (1.5mm) provides highest accuracy
   - Fast mode (3mm) is suitable for most clinical applications
   - Fastest mode (6mm) is primarily for preview purposes

3. **Memory Management**:
   - For large images (e.g., whole-body scans), use `force_split=True`
   - `body_seg=True` can reduce memory usage by cropping to the body region
   - Reduce thread count if experiencing memory issues

4. **Output Handling**:
   - Use `multilabel=True` to save disk space with a single output file
   - For radiotherapy or surgical planning, use DICOM output format
   - Enable `preview` for quick visual verification of results

## Technical Details

### Model Architecture

TotalSegmentator uses a 3D U-Net architecture (based on nnUNet framework) adapted for multi-class segmentation. The models have been trained on a large dataset of manually annotated CT and MR scans.

### Pre-processing

The tool automatically handles:
- Intensity normalization
- Resampling to the appropriate resolution
- Cropping to relevant regions

### Inference Process

1. Image loading and preprocessing
2. Task-specific model selection
3. Sliding window inference
4. Post-processing and mask generation
5. Statistics calculation (if enabled)
6. Output formatting and saving

### Support Resources

- TotalSegmentator GitHub: [https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- Citation: Wasserthal J, Breit HC, Meyer MT, Pradella M, Hinck D, Sauter AW, Heye T, Boll DT, Cyriac J, Yang S, Bach M, Segeroth M. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiol Artif Intell. 2023 Jul 5;5(5):e230024. doi: 10.1148/ryai.230024.

## Conclusion

The TotalSegmentator Tool provides a powerful and flexible solution for automated anatomical segmentation in medical imaging. By understanding the available options and best practices, users can efficiently generate accurate segmentations for clinical and research applications, from basic organ identification to advanced radiomics analysis.
