# Medical Image Classification Tools Documentation


# PyTorch ResNet Training Tool Documentation

## Overview

Trains a ResNet model for medical image classification using PyTorch. Supports pretrained models, early stopping, and validation/testing evaluations.

## Key Features

- Train ResNet 18, 34, 50, 101, or 152
- Pretrained weights support
- Validation and test evaluation
- Early stopping and learning rate scheduling

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Dataset root containing `train` (and optionally `val`) |
| `output_dir` | Directory to save models and outputs |
| `num_classes` | Number of classification categories |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|----------------|
| `model_type` | ResNet variant | `resnet50` |
| `num_epochs` | Training epochs | `10` |
| `batch_size` | Size of training batches | `16` |
| `pretrained` | Use pretrained ImageNet weights | `True` |
| `early_stopping` | Enable early stopping | `True` |
| `patience` | Epochs to wait before stopping | `5` |

## How It Works

1. Load dataset and configure model
2. Preprocess images and create loaders
3. Train and validate model
4. Save best/final models
5. Optionally evaluate on test set

## Generated Outputs

- `best_model.pt`, `final_model.pt`
- `training_plots.png`, `training_history.json`
- `model_config.json`, optional metrics



## Augmentation Techniques

During training, the ResNet tool applies the following image augmentations using PyTorch's `torchvision.transforms`:

- **RandomHorizontalFlip()** – Randomly flips images horizontally
- **RandomRotation(10)** – Applies random rotation within ±10 degrees

These are applied after resizing the images to 224x224 and before normalization.


## Example Usage

```python
tool = pytorch_resnet_training(
    data_dir="data/",
    output_dir="results/",
    num_classes=2
)
```

# PyTorch ResNet Inference Tool Documentation

## Overview

Performs inference on new data using a trained ResNet model. Optionally computes evaluation metrics if ground truth is available.

## Key Features

- Batch inference on image sets
- Classification metrics and visualizations
- Configurable model setup
- ROC/Confusion matrix outputs

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `image_dir` | Directory with inference images |
| `model_path` | Trained `.pt` model path |
| `output_dir` | Save predictions and metrics here |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|----------------|
| `config_path` | JSON model config | `None` |
| `ground_truth_file` | CSV with labels | `None` |
| `batch_size` | Batch size | `32` |

## How It Works

1. Load model and config
2. Run inference on batches
3. Optionally compute metrics
4. Save results and visualizations

## Generated Outputs

- `predictions.csv`, `metrics.json`
- `confusion_matrix.png`, `roc_curve.png`


## Example Usage

```python
tool = pytorch_resnet_inference(
    image_dir="images/",
    model_path="results/best_model.pt",
    output_dir="inference/"
)
```

# PyTorch Inception V3 Training Tool Documentation

## Overview

Trains Inception V3 models on 299x299 images for classification. Supports auxiliary logits, early stopping, and pretrained weights.

## Key Features

- Trains Inception V3 with aux logits
- Pretrained weights support
- Early stopping
- Generates evaluation plots and metrics

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Directory with `train` and optional `val` |
| `output_dir` | Save models and results |
| `num_classes` | Number of target classes |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|----------------|
| `aux_logits` | Use auxiliary outputs | `True` |
| `pretrained` | Use pretrained weights | `True` |

## How It Works

1. Resize data to 299x299
2. Train Inception V3 with aux outputs
3. Save model and metrics
4. Evaluate if test set present

## Generated Outputs

- `best_model.pt`, `training_plots.png`
- `confusion_matrix.png`, `model_config.json`



## Augmentation Techniques

During training, the Inception V3 tool uses:

- **RandomHorizontalFlip()** – Flips images horizontally at random
- **RandomRotation(10)** – Randomly rotates images within ±10 degrees

These augmentations are applied to 299x299 images before feeding into the network.


## Example Usage

```python
tool = pytorch_inception_v3_training(
    data_dir="data/",
    output_dir="inception_out/",
    num_classes=3
)
```

# PyTorch Inception V3 Inference Tool Documentation

## Overview

Runs inference using a trained Inception V3 model and computes evaluation metrics with optional ground truth.

## Key Features

- Inference on 299x299 images
- Supports aux logits
- Optional metrics and visualization
- Loads config JSON if provided

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `image_dir` | Input image folder |
| `model_path` | Trained Inception V3 `.pt` file |
| `output_dir` | Results output folder |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|----------------|
| `aux_logits` | Whether model used aux logits | `True` |
| `batch_size` | Inference batch size | `32` |

## How It Works

1. Load Inception V3 and config
2. Resize images to 299x299
3. Infer and save results
4. Evaluate if labels exist

## Generated Outputs

- `predictions.csv`, `metrics.json`
- `confusion_matrix.png`, `roc_curve.png`


## Example Usage

```python
tool = pytorch_inception_v3_inference(
    image_dir="images/",
    model_path="inception_out/best_model.pt",
    output_dir="inference/"
)
```

# PyTorch VGG16 Training Tool Documentation

## Overview

Trains a VGG16 model on medical image classification datasets. Similar structure to ResNet with configurable training and validation.

## Key Features

- VGG16 architecture training
- Supports pretrained weights
- Configurable epochs and batch size
- Early stopping support

## Input Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Dataset with training/validation data |
| `output_dir` | Save model checkpoints and logs |
| `num_classes` | Total number of target classes |

### Optional Parameters

| Parameter | Description | Default Value |
|-----------|-------------|----------------|
| `pretrained` | Use pretrained weights | `True` |
| `num_epochs` | Training epochs | `10` |
| `batch_size` | Batch size | `16` |

## How It Works

1. Load dataset and VGG16 model
2. Configure final FC layer
3. Train with early stopping
4. Save models and metrics

## Generated Outputs

- `vgg16_best_model.pt`, `training_history.json`
- `confusion_matrix.png`, `metrics.json`



## Augmentation Techniques

The VGG16 training tool includes standard image augmentations:

- **RandomHorizontalFlip()** – Random horizontal flipping
- **RandomRotation(10)** – Rotation within ±10 degrees

Augmentations are performed after resizing to 224x224 and before normalization.


## Example Usage

```python
tool = pytorch_vgg16_training(
    data_dir="data/",
    output_dir="vgg_output/",
    num_classes=4
)
```

