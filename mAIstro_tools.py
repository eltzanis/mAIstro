import os
import re
import yaml
import json
import glob
import time
from datetime import datetime
import logging
import tempfile
import shutil
import contextlib
import traceback
from pathlib import Path
import subprocess
from typing import Dict, Optional, List, Tuple, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, mean_absolute_percentage_error
)


from smolagents import Tool

#Tools

#Radiomic Extraction Tool

class PyRadiomicsFeatureExtractionTool(Tool):
    name = "pyradiomics_feature_extraction"
    description = """
    This tool extracts radiomic features from medical images (CT, MRI, etc.) using PyRadiomics.
    It processes images and their corresponding segmentation masks, and can handle multiple labels within masks.
    Features are extracted for each label and saved separately to CSV files.
    """
    
    inputs = {
        "image_dir": {
            "type": "string",
            "description": "Directory containing the medical images (e.g., MRI, CT)"
        },
        "mask_dir": {
            "type": "string",
            "description": "Directory containing the segmentation masks corresponding to the images"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where extracted features will be saved"
        },
        "image_types": {
            "type": "array",
            "description": "Types of image preprocessing to apply (e.g., Original, Wavelet, LoG). Default is all available types.",
            "required": False,
            "nullable": True
        },
        "feature_classes": {
            "type": "array",
            "description": "Classes of features to extract (e.g., firstorder, shape, glcm). Default is all available classes.",
            "required": False,
            "nullable": True
        },
        "specific_features": {
            "type": "array",
            "description": "Specific features to extract. If provided, only these features will be extracted.",
            "required": False,
            "nullable": True
        },
        "mask_labels": {
            "type": "array",
            "description": "Specific label values to extract features for. Default is all non-zero labels found in masks.",
            "required": False,
            "nullable": True
        },
        "image_pattern": {
            "type": "string",
            "description": "Filename pattern to match image files. Default is '*.nii.gz'.",
            "required": False,
            "nullable": True
        },
        "mask_pattern": {
            "type": "string",
            "description": "Filename pattern to match mask files. Default is '*.nii.gz'.",
            "required": False,
            "nullable": True
        },
        "normalize": {
            "type": "boolean",
            "description": "Whether to normalize the image intensity values. Default is True.",
            "required": False,
            "nullable": True
        },
        "bin_width": {
            "type": "number",
            "description": "Bin width for discretizing image intensities. Default is 25.",
            "required": False,
            "nullable": True
        },
        "resample": {
            "type": "boolean",
            "description": "Whether to resample the image to isotropic voxels. Default is True.",
            "required": False,
            "nullable": True
        },
        "pixel_spacing": {
            "type": "array",
            "description": "Target pixel spacing for resampling [x, y, z]. Default is [1, 1, 1].",
            "required": False,
            "nullable": True
        },
        "force_2d": {
            "type": "boolean",
            "description": "Whether to extract features slice by slice (2D) instead of 3D. Default is False.",
            "required": False,
            "nullable": True
        },
        "n_workers": {
            "type": "integer",
            "description": "Number of parallel workers for processing images. Default is 1 (serial processing).",
            "required": False,
            "nullable": True
        },
        "targets_csv": {
            "type": "string",
            "description": "Path to a CSV file containing target values for each subject. Optional.",
            "required": False,
            "nullable": True
        },
        "id_column": {
            "type": "string",
            "description": "Name of the ID column in the targets CSV. Only used if targets_csv is provided.",
            "required": False,
            "nullable": True
        },
        "target_column": {
            "type": "string",
            "description": "Name of the target column in the targets CSV. Only used if targets_csv is provided.",
            "required": False,
            "nullable": True
        },
        "id_pattern": {
            "type": "string",
            "description": "Regex pattern to extract subject ID from filenames. Default is to use the filename without extension.",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        image_dir: str,
        mask_dir: str,
        output_dir: str,
        image_types: Optional[List[str]] = None,
        feature_classes: Optional[List[str]] = None,
        specific_features: Optional[List[str]] = None,
        mask_labels: Optional[List[int]] = None,
        image_pattern: Optional[str] = "*.nii.gz",
        mask_pattern: Optional[str] = "*.nii.gz",
        normalize: Optional[bool] = True,
        bin_width: Optional[float] = 25.0,
        resample: Optional[bool] = True,
        pixel_spacing: Optional[List[float]] = None,
        force_2d: Optional[bool] = False,
        n_workers: Optional[int] = 1,
        targets_csv: Optional[str] = None,
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
        id_pattern: Optional[str] = None
    ):
        """
        Extract radiomic features from medical images using PyRadiomics.
        
        Args:
            image_dir: Directory containing the medical images
            mask_dir: Directory containing the segmentation masks
            output_dir: Directory where extracted features will be saved
            image_types: Image preprocessing types (e.g., ["Original", "Wavelet", "LoG"])
            feature_classes: Feature classes to extract (e.g., ["firstorder", "shape", "glcm"])
            specific_features: Specific features to extract (e.g., ["original_firstorder_Mean"])
            mask_labels: Specific label values to extract features for
            image_pattern: Filename pattern to match image files
            mask_pattern: Filename pattern to match mask files
            normalize: Whether to normalize image intensities
            bin_width: Bin width for discretizing image intensities
            resample: Whether to resample the image
            pixel_spacing: Target pixel spacing for resampling [x, y, z]
            force_2d: Whether to extract features in 2D mode
            n_workers: Number of parallel workers
            targets_csv: Path to CSV with target values
            id_column: Name of ID column in targets CSV
            target_column: Name of target column in targets CSV
            id_pattern: Regex pattern to extract subject ID from filenames
            
        Returns:
            Dictionary with feature extraction results and file paths
        """
        # Create a temp directory variable that we can clean up in finally block
        temp_dir = None
        
        try:
            # Validate input directories exist
            self._validate_directories(image_dir, mask_dir, output_dir)
            
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "radiomics_extraction.log")
            
            # Reset the root logger to prevent duplicate logging
            root_logger = logging.getLogger()
            if root_logger.handlers:
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)
                    
            # Configure logging with basicConfig (affects the global state)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            # Use the root logger
            logger = logging.getLogger()
            
            # Set default values
            if pixel_spacing is None:
                pixel_spacing = [1.0, 1.0, 1.0]
            
            # Validate id_pattern if provided
            if id_pattern:
                try:
                    re.compile(id_pattern)
                except re.error:
                    logger.warning(f"Invalid regex pattern provided: {id_pattern}. Defaulting to filename matching.")
                    id_pattern = None
            
            # Log configuration
            logger.info(f"Starting radiomic feature extraction with configuration:")
            logger.info(f"Image directory: {image_dir}")
            logger.info(f"Mask directory: {mask_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Image types: {image_types if image_types else 'All available'}")
            logger.info(f"Feature classes: {feature_classes if feature_classes else 'All available'}")
            logger.info(f"Specific features: {specific_features if specific_features else 'None'}")
            logger.info(f"Mask labels: {mask_labels if mask_labels else 'All non-zero'}")
            logger.info(f"Normalize: {normalize}")
            logger.info(f"Bin width: {bin_width}")
            logger.info(f"Resample: {resample}")
            logger.info(f"Pixel spacing: {pixel_spacing}")
            logger.info(f"Force 2D: {force_2d}")
            logger.info(f"Number of workers: {n_workers}")
            
            # Prepare PyRadiomics parameters
            params = self._create_pyradiomics_params(
                image_types=image_types,
                feature_classes=feature_classes,
                specific_features=specific_features,
                normalize=normalize,
                bin_width=bin_width,
                resample=resample,
                pixel_spacing=pixel_spacing,
                force_2d=force_2d
            )
            
            # Save parameters to a YAML file
            params_file = os.path.join(output_dir, "radiomics_params.yaml")
            with open(params_file, 'w') as f:
                yaml.dump(params, f)
            logger.info(f"PyRadiomics parameters saved to {params_file}")
            
            # Initialize feature extractor
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
            logger.info(f"Feature extractor initialized with {len(extractor.enabledFeatures)} features")
            
            # Load target values if provided
            target_dict = None
            if targets_csv:
                if not os.path.exists(targets_csv):
                    logger.warning(f"Target CSV file not found: {targets_csv}")
                else:
                    logger.info(f"Loading target values from {targets_csv}")
                    target_dict = self._load_targets(
                        targets_csv, 
                        id_column=id_column, 
                        target_column=target_column
                    )
                    logger.info(f"Loaded {len(target_dict)} target values")
            
            # Find image and mask files
            image_files = sorted(glob.glob(os.path.join(image_dir, image_pattern)))
            mask_files = sorted(glob.glob(os.path.join(mask_dir, mask_pattern)))
            
            if not image_files:
                logger.error(f"No image files found in {image_dir} with pattern {image_pattern}")
                return {
                    "status": "error",
                    "error_message": f"No image files found in {image_dir} with pattern {image_pattern}",
                    "image_dir": image_dir,
                    "mask_dir": mask_dir
                }
                
            if not mask_files:
                logger.error(f"No mask files found in {mask_dir} with pattern {mask_pattern}")
                return {
                    "status": "error",
                    "error_message": f"No mask files found in {mask_dir} with pattern {mask_pattern}",
                    "image_dir": image_dir,
                    "mask_dir": mask_dir
                }
            
            logger.info(f"Found {len(image_files)} image files and {len(mask_files)} mask files")
            
            # Match images and masks
            image_mask_pairs, unmatched = self._match_images_and_masks(
                image_files, 
                mask_files, 
                id_pattern=id_pattern,
                logger=logger
            )
            
            logger.info(f"Matched {len(image_mask_pairs)} image-mask pairs")
            if unmatched['images'] or unmatched['masks']:
                logger.warning(f"Unmatched images: {len(unmatched['images'])}, unmatched masks: {len(unmatched['masks'])}")
                if unmatched['images']:
                    logger.debug(f"Unmatched image examples: {unmatched['images'][:5]}")
                if unmatched['masks']:
                    logger.debug(f"Unmatched mask examples: {unmatched['masks'][:5]}")
            
            # Check if we have any pairs to process
            if not image_mask_pairs:
                logger.error("No matching image-mask pairs found. Check your files and patterns.")
                return {
                    "status": "error",
                    "error_message": "No matching image-mask pairs found",
                    "image_dir": image_dir,
                    "mask_dir": mask_dir
                }
            
            # Determine available labels in masks if not specified
            if mask_labels is None:
                mask_labels = self._find_available_labels(mask_files, logger=logger)
                logger.info(f"Found {len(mask_labels)} unique labels in masks: {mask_labels}")
            
            # Create a temporary directory for extracted masks by label
            temp_dir = tempfile.mkdtemp(dir=output_dir)
            logger.info(f"Created temporary directory for masks: {temp_dir}")
            
            # Process each image-mask pair
            start_time = time.time()
            
            # Initialize feature dictionaries for each label
            all_features_by_label = {label: [] for label in mask_labels}
            
            # Process serially or in parallel based on n_workers
            if n_workers <= 1:
                # Serial processing
                for i, (subject_id, files) in enumerate(image_mask_pairs.items()):
                    try:
                        self._process_subject(
                            subject_id, 
                            files, 
                            mask_labels, 
                            extractor, 
                            temp_dir, 
                            all_features_by_label, 
                            target_dict,
                            i + 1,
                            len(image_mask_pairs),
                            logger=logger
                        )
                    except Exception as e:
                        logger.error(f"Error processing subject {subject_id}: {str(e)}")
                        logger.debug(traceback.format_exc())
            else:
                # Parallel processing
                logger.info(f"Using {n_workers} workers for parallel processing")
                
                # Create arguments for parallel processing
                process_args = []
                for i, (subject_id, files) in enumerate(image_mask_pairs.items()):
                    # Create a unique temp directory for each subject to avoid conflicts
                    subject_temp_dir = os.path.join(temp_dir, f"subject_{subject_id}")
                    os.makedirs(subject_temp_dir, exist_ok=True)
                    
                    target_value = None
                    if target_dict and subject_id in target_dict:
                        target_value = target_dict[subject_id]
                    
                    process_args.append((
                        subject_id, 
                        files['image'],
                        files['mask'], 
                        mask_labels, 
                        params_file, 
                        subject_temp_dir,
                        target_value,
                        i + 1,
                        len(image_mask_pairs),
                        log_file  # Pass log file path to enable logging in subprocess
                    ))
                
                # Execute in parallel
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(self._process_subject_parallel, *args) for args in process_args]
                    
                    for future in as_completed(futures):
                        try:
                            subject_id, subject_features_by_label = future.result()
                            # Merge results
                            for label, features in subject_features_by_label.items():
                                if features:  # Only add if features were extracted
                                    all_features_by_label[label].append(features)
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {str(e)}")
                            logger.debug(traceback.format_exc())
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Feature extraction completed in {processing_time:.2f} seconds")
            
            # Save features to CSV files
            csv_paths = {}
            
            # Check if any features were extracted
            total_features_count = sum(len(features) for features in all_features_by_label.values())
            if total_features_count == 0:
                logger.warning("No features were successfully extracted for any label")
                
            for label, features in all_features_by_label.items():
                if not features:
                    logger.warning(f"No features extracted for label {label}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(features)
                
                # Arrange columns (ID and target columns first, then features)
                id_cols = [col for col in df.columns if col.lower() in ['id', 'subject_id', 'patientid']]
                target_cols = [col for col in df.columns if col.lower() in ['target', 'label', 'outcome']]
                feature_cols = [col for col in df.columns if col not in id_cols + target_cols]
                
                # Sort feature columns for consistency
                feature_cols.sort()
                
                # Reorder columns - make sure all columns exist
                all_cols = [col for col in id_cols + target_cols + feature_cols if col in df.columns]
                df = df[all_cols]
                
                # Save to CSV
                csv_path = os.path.join(output_dir, f"radiomic_features_label_{label}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(df)} subjects with {len(df.columns) - len(id_cols) - len(target_cols)} features for label {label} to {csv_path}")
                
                csv_paths[f"label_{label}"] = csv_path
            
            # Clean up temporary directory (this will be done in the finally block)
            
            # Calculate feature counts properly
            feature_counts = {}
            for label, features_list in all_features_by_label.items():
                if features_list:
                    df = pd.DataFrame(features_list)
                    id_cols = [col for col in df.columns if col.lower() in ['id', 'subject_id', 'patientid']]
                    target_cols = [col for col in df.columns if col.lower() in ['target', 'label', 'outcome']]
                    feature_count = len(df.columns) - len(id_cols) - len(target_cols)
                    feature_counts[label] = feature_count
            
            # Return results
            return {
                "status": "success",
                "csv_paths": csv_paths,
                "num_subjects": len(image_mask_pairs),
                "num_features": feature_counts,
                "mask_labels": mask_labels,
                "log_file": log_file,
                "params_file": params_file,
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            # Handle any unexpected exceptions
            if 'logger' in locals():
                logger.error(f"Error during radiomic feature extraction: {str(e)}")
                logger.debug(traceback.format_exc())
            else:
                print(f"Error during radiomic feature extraction: {str(e)}")
                traceback.print_exc()
                
            return {
                "status": "error",
                "error_message": str(e),
                "image_dir": image_dir,
                "mask_dir": mask_dir,
                "output_dir": output_dir
            }
            
        finally:
            # Clean up temp directory if it was created
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    if 'logger' in locals():
                        logger.info(f"Removed temporary directory: {temp_dir}")
                    else:
                        print(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    if 'logger' in locals():
                        logger.warning(f"Failed to remove temporary directory {temp_dir}: {str(e)}")
                    else:
                        print(f"Failed to remove temporary directory {temp_dir}: {str(e)}")
    
    def _validate_directories(self, image_dir, mask_dir, output_dir):
        """Validate that input directories exist"""
        for directory, name in [(image_dir, "Image directory"), (mask_dir, "Mask directory")]:
            if not os.path.exists(directory):
                raise ValueError(f"{name} does not exist: {directory}")
            if not os.path.isdir(directory):
                raise ValueError(f"{name} is not a directory: {directory}")
    
    def _create_pyradiomics_params(
        self, 
        image_types=None, 
        feature_classes=None, 
        specific_features=None,
        normalize=True,
        bin_width=25.0,
        resample=True,
        pixel_spacing=None,
        force_2d=False
    ):
        """Create PyRadiomics parameters dictionary"""
        # Default image types based on modality - more conservative default
        if image_types is None:
            image_types = ["Original", "Wavelet"]
        
        # Default feature classes with all their features
        if feature_classes is None and specific_features is None:
            feature_classes = [
                "firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"
            ]
        
        # Initialize parameters dictionary
        params = {
            "imageType": {},
            "featureClass": {},
            "setting": {
                "force2D": force_2d,
                "normalize": normalize,
                "normalizeScale": 100,
                "binWidth": bin_width,
                "interpolator": "sitkBSpline",
                "correctMask": True,
                "padDistance": 10,
                "removeOutliers": 3
            }
        }
        
        # Add resampling if needed
        if resample:
            if pixel_spacing is None:
                pixel_spacing = [1.0, 1.0, 1.0]
            params["setting"]["resampledPixelSpacing"] = pixel_spacing
        
        # Configure image types
        for img_type in image_types:
            if img_type == "Original":
                params["imageType"]["Original"] = {}
            elif img_type == "Wavelet":
                params["imageType"]["Wavelet"] = {}
            elif img_type == "LoG":
                # Configure LoG filter with different sigma values
                params["imageType"]["LoG"] = {"sigma": [1.0, 2.0, 3.0, 4.0, 5.0]}
            elif img_type == "Exponential":
                params["imageType"]["Exponential"] = {}
            elif img_type == "Gradient":
                params["imageType"]["Gradient"] = {}
            elif img_type == "SquareRoot":
                params["imageType"]["SquareRoot"] = {}
            elif img_type == "LBP2D":
                params["imageType"]["LBP2D"] = {}
            elif img_type == "LBP3D":
                params["imageType"]["LBP3D"] = {}
        
        # Configure feature classes if using all features
        if specific_features is None:
            for feature_class in feature_classes:
                params["featureClass"][feature_class] = []
        else:
            # If specific features are provided, parse and enable them
            for feature in specific_features:
                # Parse feature string (e.g., "original_firstorder_Mean")
                parts = feature.split('_')
                if len(parts) >= 2:
                    # The last part is the feature name
                    feature_name = parts[-1]
                    # The second-to-last part is the feature class
                    feature_class = parts[-2]
                    
                    # Ensure the feature class exists in the parameters
                    if feature_class not in params["featureClass"]:
                        params["featureClass"][feature_class] = []
                    
                    # Add the feature to the list if it's not already there
                    if feature_name not in params["featureClass"][feature_class]:
                        params["featureClass"][feature_class].append(feature_name)
        
        return params
    
    def _load_targets(self, targets_csv, id_column=None, target_column=None, logger=None):
        """Load target values from CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(targets_csv)
            
            # Determine columns to use
            if id_column is None:
                id_column = df.columns[0]  # Use first column as ID
                if logger:
                    logger.info(f"No ID column specified, using first column: {id_column}")
            
            if target_column is None:
                target_column = df.columns[1]  # Use second column as target
                if logger:
                    logger.info(f"No target column specified, using second column: {target_column}")
            
            # Check if columns exist
            if id_column not in df.columns:
                if logger:
                    logger.error(f"ID column '{id_column}' not found in {targets_csv}")
                raise ValueError(f"ID column '{id_column}' not found in {targets_csv}")
            
            if target_column not in df.columns:
                if logger:
                    logger.error(f"Target column '{target_column}' not found in {targets_csv}")
                raise ValueError(f"Target column '{target_column}' not found in {targets_csv}")
            
            # Create dictionary mapping IDs to target values
            target_dict = {}
            for _, row in df.iterrows():
                # Convert ID to string for consistent matching
                subject_id = str(row[id_column])
                target_dict[subject_id] = row[target_column]
            
            return target_dict
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading targets: {str(e)}")
                logger.debug(traceback.format_exc())
            return {}
    
    def _match_images_and_masks(self, image_files, mask_files, id_pattern=None, logger=None):
        """Match image and mask files by subject ID"""
        image_mask_pairs = {}
        unmatched = {"images": [], "masks": []}
        
        # Extract subject IDs based on the pattern or use filenames
        def extract_id(filepath, pattern=None):
            filename = os.path.basename(filepath)
            if pattern:
                try:
                    match = re.search(pattern, filename)
                    if match and match.groups():
                        return match.group(1)
                except re.error:
                    if logger:
                        logger.warning(f"Invalid regex pattern: {pattern}. Falling back to filename matching.")
                    return os.path.splitext(filename)[0]
            # Default: use filename without extension
            return os.path.splitext(filename)[0]
        
        # Create dictionaries for images and masks with subject IDs as keys
        images_dict = {}
        for f in image_files:
            subject_id = extract_id(f, id_pattern)
            # Handle duplicate subject IDs
            if subject_id in images_dict:
                if logger:
                    logger.warning(f"Duplicate subject ID '{subject_id}' found in images: {images_dict[subject_id]} and {f}")
            images_dict[subject_id] = f
            
        masks_dict = {}
        for f in mask_files:
            subject_id = extract_id(f, id_pattern)
            # Handle duplicate subject IDs
            if subject_id in masks_dict:
                if logger:
                    logger.warning(f"Duplicate subject ID '{subject_id}' found in masks: {masks_dict[subject_id]} and {f}")
            masks_dict[subject_id] = f
        
        # Find matching pairs
        all_subjects = set(images_dict.keys()) | set(masks_dict.keys())
        for subject_id in all_subjects:
            if subject_id in images_dict and subject_id in masks_dict:
                image_mask_pairs[subject_id] = {
                    "image": images_dict[subject_id],
                    "mask": masks_dict[subject_id]
                }
            else:
                if subject_id in images_dict:
                    unmatched["images"].append(images_dict[subject_id])
                if subject_id in masks_dict:
                    unmatched["masks"].append(masks_dict[subject_id])
        
        return image_mask_pairs, unmatched
    
    def _find_available_labels(self, mask_files, max_files_to_check=10, logger=None):
        """Find unique label values in mask files"""
        unique_labels = set()
        
        # Limit the number of files to check to avoid lengthy processing
        files_to_check = mask_files[:min(max_files_to_check, len(mask_files))]
        
        for mask_file in files_to_check:
            try:
                # Read the mask file
                mask_image = sitk.ReadImage(mask_file)
                mask_array = sitk.GetArrayFromImage(mask_image)
                
                # Find unique non-zero values
                labels = np.unique(mask_array)
                labels = labels[labels > 0]  # Exclude background (0)
                
                for label in labels:
                    unique_labels.add(int(label))
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Error reading mask {mask_file}: {str(e)}")
                
        # Ensure we found at least one label
        if not unique_labels and mask_files:
            if logger:
                logger.warning("No non-zero labels found in the mask files.")
            # Fallback to label 1 if no labels were found
            unique_labels.add(1)
            if logger:
                logger.info("Added fallback label 1 to ensure processing continues.")
        
        return sorted(list(unique_labels))
    
    def _process_subject(
        self, 
        subject_id, 
        files, 
        mask_labels, 
        extractor, 
        temp_dir, 
        all_features_by_label, 
        target_dict,
        subject_index,
        total_subjects,
        logger=None
    ):
        """Process a single subject and extract features for each label"""
        try:
            if logger:
                logger.info(f"Processing subject {subject_id} ({subject_index}/{total_subjects})")
            
            # Read image and mask
            image_path = files['image']
            mask_path = files['mask']
            
            # Validate files exist
            if not os.path.exists(image_path):
                if logger:
                    logger.error(f"Image file does not exist: {image_path}")
                return
                
            if not os.path.exists(mask_path):
                if logger:
                    logger.error(f"Mask file does not exist: {mask_path}")
                return
            
            # Read the image and mask
            orig_image = sitk.ReadImage(image_path)
            orig_mask = sitk.ReadImage(mask_path)
            
            # Process each label
            for label in mask_labels:
                # Create a binary mask for this label
                mask_array = sitk.GetArrayFromImage(orig_mask)
                binary_mask = (mask_array == label).astype(np.uint8)
                
                # Skip if no voxels with this label
                if np.sum(binary_mask) == 0:
                    if logger:
                        logger.info(f"Subject {subject_id}: No voxels with label {label}, skipping")
                    continue
                
                # Create a SimpleITK image from the binary mask
                binary_mask_image = sitk.GetImageFromArray(binary_mask)
                binary_mask_image.CopyInformation(orig_mask)
                
                # Save the binary mask temporarily
                label_mask_path = os.path.join(temp_dir, f"{subject_id}_label_{label}.nii.gz")
                sitk.WriteImage(binary_mask_image, label_mask_path)
                
                # Extract features
                try:
                    result = extractor.execute(image_path, label_mask_path)
                    
                    # Convert to regular dictionary and remove diagnostic keys
                    features = {str(key): value for key, value in result.items() 
                              if not key.startswith('diagnostics_')}
                    
                    # Add subject ID and target value if available
                    features['subject_id'] = subject_id
                    if target_dict and subject_id in target_dict:
                        features['target'] = target_dict[subject_id]
                    
                    # Add to the corresponding label's feature list
                    all_features_by_label[label].append(features)
                    
                    if logger:
                        logger.info(f"Subject {subject_id}: Extracted {len(features)} features for label {label}")
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error extracting features for subject {subject_id}, label {label}: {str(e)}")
                        logger.debug(traceback.format_exc())
                
                # Clean up the temporary mask file
                try:
                    if os.path.exists(label_mask_path):
                        os.remove(label_mask_path)
                except Exception as e:
                    if logger:
                        logger.warning(f"Error removing temporary mask file {label_mask_path}: {str(e)}")
            
        except Exception as e:
            if logger:
                logger.error(f"Error processing subject {subject_id}: {str(e)}")
                logger.debug(traceback.format_exc())
    
    @staticmethod
    def _process_subject_parallel(
        subject_id,
        image_path,
        mask_path, 
        mask_labels, 
        params_file,
        temp_dir,
        target_value,
        subject_index,
        total_subjects,
        log_file
    ):
        """Process a single subject in parallel mode"""
        # Set up basic logging for the subprocess
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file)
            ]
        )
        
        # Use the root logger
        logger = logging.getLogger()
        
        # Initialize variables for cleanup
        created_files = []
        
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
            
            logger.info(f"Parallel process: processing subject {subject_id} ({subject_index}/{total_subjects})")
            
            # Validate files exist
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return subject_id, {}
                
            if not os.path.exists(mask_path):
                logger.error(f"Mask file does not exist: {mask_path}")
                return subject_id, {}
            
            # Initialize local extractor
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
            
            # Read image and mask
            orig_image = sitk.ReadImage(image_path)
            orig_mask = sitk.ReadImage(mask_path)
            
            # Initialize features by label
            features_by_label = {label: None for label in mask_labels}
            
            # Process each label
            for label in mask_labels:
                # Create a binary mask for this label
                mask_array = sitk.GetArrayFromImage(orig_mask)
                binary_mask = (mask_array == label).astype(np.uint8)
                
                # Skip if no voxels with this label
                if np.sum(binary_mask) == 0:
                    logger.info(f"Subject {subject_id}: No voxels with label {label}, skipping")
                    continue
                
                # Create a SimpleITK image from the binary mask
                binary_mask_image = sitk.GetImageFromArray(binary_mask)
                binary_mask_image.CopyInformation(orig_mask)
                
                # Save the binary mask temporarily
                label_mask_path = os.path.join(temp_dir, f"{subject_id}_label_{label}.nii.gz")
                sitk.WriteImage(binary_mask_image, label_mask_path)
                created_files.append(label_mask_path)
                
                # Extract features
                try:
                    result = extractor.execute(image_path, label_mask_path)
                    
                    # Convert to regular dictionary and remove diagnostic keys
                    features = {str(key): value for key, value in result.items() 
                              if not key.startswith('diagnostics_')}
                    
                    # Add subject ID and target value if available
                    features['subject_id'] = subject_id
                    if target_value is not None:
                        features['target'] = target_value
                    
                    # Store features for this label
                    features_by_label[label] = features
                    
                    logger.info(f"Subject {subject_id}: Extracted {len(features)} features for label {label}")
                    
                except Exception as e:
                    logger.error(f"Error extracting features for subject {subject_id}, label {label}: {str(e)}")
                    logger.debug(traceback.format_exc())
            
            return subject_id, features_by_label
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            return subject_id, {}
            
        finally:
            # Clean up temporary files
            for file_path in created_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
            
            # Try to clean up the temp directory, but only if it's empty
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {temp_dir}: {str(e)}")

#Exploratory Data Analysis Tool

class EDAToolException(Exception):
    """Custom exception for EDA Tool errors."""
    pass

class ExploratoryDataAnalysisTool(Tool):
    name = "exploratory_data_analysis"
    description = """
    This tool performs comprehensive exploratory data analysis on tabulated data (Excel or CSV).
    It analyzes the data structure, generates statistics, creates visualizations, and produces reports.
    Results are saved to a specified output directory for further inspection.
    """
    
    # Valid correlation methods
    VALID_CORRELATION_METHODS = ['pearson', 'spearman', 'kendall']
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input data file (CSV or Excel)"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory where analysis results will be saved"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (ignored for CSV files)",
            "required": False,
            "nullable": True
        },
        "target_column": {
            "type": "string",
            "description": "Target column for analyzing relationships (e.g., for classification/regression)",
            "required": False,
            "nullable": True
        },
        "categorical_threshold": {
            "type": "integer",
            "description": "Maximum number of unique values to consider a column categorical",
            "required": False,
            "nullable": True
        },
        "correlation_method": {
            "type": "string",
            "description": "Method for correlation calculation ('pearson', 'spearman', or 'kendall')",
            "required": False,
            "nullable": True
        },
        "visualize_distributions": {
            "type": "boolean",
            "description": "Generate distribution plots for numerical columns",
            "required": False,
            "nullable": True
        },
        "visualize_correlations": {
            "type": "boolean",
            "description": "Generate correlation heatmap",
            "required": False,
            "nullable": True
        },
        "visualize_pairplot": {
            "type": "boolean",
            "description": "Generate pairplot for numerical columns",
            "required": False,
            "nullable": True
        },
        "visualize_target_relationships": {
            "type": "boolean",
            "description": "Generate plots showing relationships with target column",
            "required": False,
            "nullable": True
        },
        "max_categories_pie": {
            "type": "integer",
            "description": "Maximum number of categories to display in pie charts",
            "required": False,
            "nullable": True
        },
        "sampling_for_large_data": {
            "type": "boolean", 
            "description": "Sample data if it's too large for visualization",
            "required": False,
            "nullable": True
        },
        "sample_size": {
            "type": "integer",
            "description": "Number of rows to sample for large datasets",
            "required": False,
            "nullable": True
        },
        "time_series_analysis": {
            "type": "boolean",
            "description": "Perform time series analysis if datetime columns are detected",
            "required": False,
            "nullable": True
        },
        "columns_to_exclude": {
            "type": "string",
            "description": "Comma-separated list of columns to exclude from analysis",
            "required": False,
            "nullable": True
        },
        "detect_outliers": {
            "type": "boolean",
            "description": "Detect and analyze outliers in numerical columns",
            "required": False,
            "nullable": True
        },
        "create_summary_report": {
            "type": "boolean",
            "description": "Create a comprehensive summary report in text format",
            "required": False,
            "nullable": True
        },
        "max_columns_for_correlation": {
            "type": "integer",
            "description": "Maximum number of columns to include in correlation matrix",
            "required": False,
            "nullable": True
        },
        "max_columns_for_pairplot": {
            "type": "integer",
            "description": "Maximum number of columns to include in pairplot",
            "required": False,
            "nullable": True
        },
        "create_figures": {
            "type": "boolean",
            "description": "Master switch to enable/disable all visualizations",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        output_dir: str,
        sheet_name: Optional[str] = None,
        target_column: Optional[str] = None,
        categorical_threshold: Optional[int] = 10,
        correlation_method: Optional[str] = "pearson",
        visualize_distributions: Optional[bool] = True,
        visualize_correlations: Optional[bool] = True,
        visualize_pairplot: Optional[bool] = True,
        visualize_target_relationships: Optional[bool] = True,
        max_categories_pie: Optional[int] = 10,
        sampling_for_large_data: Optional[bool] = True,
        sample_size: Optional[int] = 10000,
        time_series_analysis: Optional[bool] = True,
        columns_to_exclude: Optional[str] = None,
        detect_outliers: Optional[bool] = True,
        create_summary_report: Optional[bool] = True,
        max_columns_for_correlation: Optional[int] = 100,
        max_columns_for_pairplot: Optional[int] = 10,
        create_figures: Optional[bool] = True
    ):
        """
        Perform exploratory data analysis on tabulated data and save results.
        
        Args:
            input_path: Path to input data file (CSV or Excel)
            output_dir: Output directory where analysis results will be saved
            sheet_name: Sheet name for Excel files
            target_column: Target column for analyzing relationships
            categorical_threshold: Max unique values to consider a column categorical
            correlation_method: Method for correlation calculation
            visualize_distributions: Generate distribution plots
            visualize_correlations: Generate correlation heatmap
            visualize_pairplot: Generate pairplot for numerical columns
            visualize_target_relationships: Generate plots showing relationships with target
            max_categories_pie: Maximum categories to display in pie charts
            sampling_for_large_data: Sample data if it's too large
            sample_size: Number of rows to sample for large datasets
            time_series_analysis: Perform time series analysis if possible
            columns_to_exclude: Columns to exclude from analysis
            detect_outliers: Detect and analyze outliers
            create_summary_report: Create a comprehensive summary report
            max_columns_for_correlation: Maximum number of columns to include in correlation matrix
            max_columns_for_pairplot: Maximum number of columns to include in pairplot
            create_figures: Master switch to enable/disable all visualizations
            
        Returns:
            Dictionary with analysis results and file paths
        """
        # Set up logging
        logger = self._setup_logging()
        
        # Setup output directory and log file
        log_file = None
        
        try:
            # Validate inputs
            self._validate_inputs(
                input_path, 
                output_dir, 
                correlation_method, 
                categorical_threshold,
                max_categories_pie,
                sample_size,
                max_columns_for_correlation,
                max_columns_for_pairplot
            )
            
            # Override visualization settings if create_figures is False
            if not create_figures:
                visualize_distributions = False
                visualize_correlations = False
                visualize_pairplot = False
                visualize_target_relationships = False
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up logging file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(output_dir, f"eda_log_{timestamp}.txt")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
            logger.addHandler(file_handler)
            
            logger.info(f"EDA started for file: {input_path}")
            logger.info(f"Output directory: {output_dir}")
            
            # Load data
            logger.info("Loading data...")
            df = self._load_data(input_path, sheet_name)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Process columns to exclude
            excluded_columns = []
            if columns_to_exclude:
                excluded_columns = [col.strip() for col in columns_to_exclude.split(',')]
                existing_excluded = [col for col in excluded_columns if col in df.columns]
                
                if len(existing_excluded) > 0:
                    df = df.drop(columns=existing_excluded)
                    logger.info(f"Excluded {len(existing_excluded)} columns: {existing_excluded}")
                
                if len(existing_excluded) < len(excluded_columns):
                    not_found = set(excluded_columns) - set(existing_excluded)
                    logger.warning(f"Some columns to exclude were not found in the dataset: {not_found}")
            
            # Check for target column
            if target_column and target_column not in df.columns:
                logger.warning(f"Target column '{target_column}' not found in the dataset. Target-based visualizations will be skipped.")
                target_column = None
                visualize_target_relationships = False
            
            # Sample data if needed
            original_row_count = len(df)
            if sampling_for_large_data and len(df) > sample_size:
                logger.info(f"Sampling data from {len(df)} to {sample_size} rows for visualization")
                df_sampled = df.sample(sample_size, random_state=42)
            else:
                df_sampled = df
            
            # Basic data profiling
            logger.info("Performing basic data profiling...")
            profile_results = self._profile_data(df)
            profile_path = os.path.join(output_dir, "data_profile.json")
            with open(profile_path, 'w') as f:
                json.dump(profile_results, f, indent=2, default=str)
            logger.info(f"Data profile saved to {profile_path}")
            
            # Save data summary
            summary_stats_path = self._create_data_summary(df, output_dir, categorical_threshold)
            logger.info(f"Summary statistics saved to {summary_stats_path}")
            
            # Create a figures directory
            figures_dir = os.path.join(output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Detect column types
            column_types = self._detect_column_types(df_sampled, categorical_threshold)
            numeric_cols = column_types['numeric']
            categorical_cols = column_types['categorical']
            datetime_cols = column_types['datetime']
            
            logger.info(f"Detected {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns, and {len(datetime_cols)} datetime columns")
            
            # Initialize list to track generated files
            generated_files = []
            
            # Perform visualizations
            if create_figures:
                logger.info("Starting visualizations...")
                
                # Visualize distributions
                if visualize_distributions:
                    logger.info("Generating distribution visualizations...")
                    try:
                        dist_files = self._visualize_distributions(
                            df_sampled, 
                            figures_dir, 
                            categorical_threshold, 
                            max_categories_pie
                        )
                        generated_files.extend(dist_files)
                        logger.info(f"Generated {len(dist_files)} distribution visualizations")
                    except Exception as e:
                        logger.error(f"Error generating distribution visualizations: {str(e)}")
                        logger.debug(traceback.format_exc())
                
                # Visualize correlations
                if visualize_correlations and len(numeric_cols) > 1:
                    logger.info("Generating correlation visualizations...")
                    try:
                        # Limit columns for correlation if specified
                        if max_columns_for_correlation and len(numeric_cols) > max_columns_for_correlation:
                            logger.info(f"Too many numeric columns ({len(numeric_cols)}) for correlation matrix. Limiting to top {max_columns_for_correlation}.")
                            numeric_cols_corr = numeric_cols[:max_columns_for_correlation]
                        else:
                            numeric_cols_corr = numeric_cols
                            
                        corr_files = self._visualize_correlations(
                            df_sampled[numeric_cols_corr], 
                            figures_dir, 
                            correlation_method
                        )
                        if corr_files:
                            if isinstance(corr_files, list):
                                generated_files.extend(corr_files)
                                logger.info(f"Generated {len(corr_files)} correlation visualizations")
                            else:
                                generated_files.append(corr_files)
                                logger.info("Generated correlation visualization")
                    except Exception as e:
                        logger.error(f"Error generating correlation visualization: {str(e)}")
                        logger.debug(traceback.format_exc())
                
                # Visualize pairplot
                if visualize_pairplot and len(numeric_cols) > 1:
                    logger.info("Generating pairplot...")
                    try:
                        # Limit columns for pairplot
                        max_cols = max_columns_for_pairplot or 10
                        num_cols_for_pairplot = min(len(numeric_cols), max_cols)
                        if len(numeric_cols) > num_cols_for_pairplot:
                            logger.info(f"Too many numeric columns ({len(numeric_cols)}) for pairplot. Limiting to top {num_cols_for_pairplot}.")
                            numeric_cols_subset = numeric_cols[:num_cols_for_pairplot]
                        else:
                            numeric_cols_subset = numeric_cols
                            
                        pairplot_file = self._visualize_pairplot(
                            df_sampled, 
                            numeric_cols_subset, 
                            figures_dir, 
                            target_column
                        )
                        if pairplot_file:
                            generated_files.append(pairplot_file)
                            logger.info("Generated pairplot visualization")
                    except Exception as e:
                        logger.error(f"Error generating pairplot: {str(e)}")
                        logger.debug(traceback.format_exc())
                
                # Visualize relationships with target if requested
                if visualize_target_relationships and target_column and target_column in df.columns:
                    logger.info(f"Generating visualizations for relationships with target: {target_column}")
                    try:
                        target_files = self._visualize_target_relationships(
                            df_sampled, 
                            target_column, 
                            figures_dir, 
                            categorical_threshold
                        )
                        generated_files.extend(target_files)
                        logger.info(f"Generated {len(target_files)} target relationship visualizations")
                    except Exception as e:
                        logger.error(f"Error generating target relationship visualizations: {str(e)}")
                        logger.debug(traceback.format_exc())
                
                # Perform time series analysis if requested and datetime columns exist
                if time_series_analysis and datetime_cols:
                    logger.info(f"Performing time series analysis on columns: {datetime_cols}")
                    try:
                        time_series_files = self._time_series_analysis(df, datetime_cols, figures_dir)
                        generated_files.extend(time_series_files)
                        logger.info(f"Generated {len(time_series_files)} time series visualizations")
                    except Exception as e:
                        logger.error(f"Error generating time series visualizations: {str(e)}")
                        logger.debug(traceback.format_exc())
            
            # Detect outliers
            outlier_info = None
            if detect_outliers and numeric_cols:
                logger.info("Detecting outliers in numerical columns...")
                try:
                    outlier_info = self._detect_outliers(df, numeric_cols)
                    outlier_path = os.path.join(output_dir, "outliers.json")
                    with open(outlier_path, 'w') as f:
                        json.dump(outlier_info, f, indent=2, default=str)
                    logger.info(f"Outlier information saved to {outlier_path}")
                    
                    # Visualize outliers only if visualizations are enabled
                    if create_figures and visualize_distributions:
                        outlier_files = self._visualize_outliers(df_sampled, numeric_cols, figures_dir)
                        generated_files.extend(outlier_files)
                        logger.info(f"Generated {len(outlier_files)} outlier visualizations")
                except Exception as e:
                    logger.error(f"Error in outlier detection: {str(e)}")
                    logger.debug(traceback.format_exc())
            
            # Create summary report
            report_path = None
            if create_summary_report:
                logger.info("Creating summary report...")
                try:
                    report_path = self._create_summary_report(
                        df, 
                        profile_results, 
                        outlier_info, 
                        generated_files, 
                        output_dir,
                        column_types
                    )
                    logger.info(f"Summary report saved to {report_path}")
                except Exception as e:
                    logger.error(f"Error creating summary report: {str(e)}")
                    logger.debug(traceback.format_exc())
            
            logger.info("EDA completed successfully.")
            
            return {
                "status": "success",
                "input_path": input_path,
                "output_dir": output_dir,
                "profile_path": profile_path,
                "summary_stats_path": summary_stats_path,
                "generated_files": generated_files,
                "report_path": report_path,
                "row_count": original_row_count,
                "column_count": len(df.columns) + len(excluded_columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols,
                "has_missing_data": df.isna().any().any(),
                "log_file": log_file
            }
            
        except Exception as e:
            error_msg = str(e)
            if logger:
                logger.error(f"Error during EDA: {error_msg}")
                logger.debug(traceback.format_exc())
            
            return {
                "status": "error",
                "error_message": error_msg,
                "input_path": input_path,
                "output_dir": output_dir,
                "log_file": log_file
            }
    
    def _setup_logging(self):
        """Set up logging for the tool"""
        logger = logging.getLogger('eda_tool')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if logger.handlers:
            logger.handlers.clear()
            
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(console_handler)
        
        return logger
    
    def _validate_inputs(
        self, 
        input_path, 
        output_dir, 
        correlation_method,
        categorical_threshold,
        max_categories_pie,
        sample_size,
        max_columns_for_correlation,
        max_columns_for_pairplot
    ):
        """Validate input parameters"""
        # Check if input file exists
        if not os.path.exists(input_path):
            raise EDAToolException(f"Input file not found: {input_path}")
            
        # Check if input file is a file
        if not os.path.isfile(input_path):
            raise EDAToolException(f"Input path is not a file: {input_path}")
            
        # Validate correlation method
        if correlation_method not in self.VALID_CORRELATION_METHODS:
            raise EDAToolException(
                f"Invalid correlation method: {correlation_method}. "
                f"Valid methods are: {', '.join(self.VALID_CORRELATION_METHODS)}"
            )
            
        # Validate numeric parameters
        if categorical_threshold is not None and categorical_threshold < 1:
            raise EDAToolException(f"categorical_threshold must be at least 1, got {categorical_threshold}")
            
        if max_categories_pie is not None and max_categories_pie < 1:
            raise EDAToolException(f"max_categories_pie must be at least 1, got {max_categories_pie}")
            
        if sample_size is not None and sample_size < 1:
            raise EDAToolException(f"sample_size must be at least 1, got {sample_size}")
            
        if max_columns_for_correlation is not None and max_columns_for_correlation < 2:
            raise EDAToolException(f"max_columns_for_correlation must be at least 2, got {max_columns_for_correlation}")
            
        if max_columns_for_pairplot is not None and max_columns_for_pairplot < 2:
            raise EDAToolException(f"max_columns_for_pairplot must be at least 2, got {max_columns_for_pairplot}")
    
    def _sanitize_filename(self, name):
        """
        Sanitize a string to be used as a valid filename.
        Replaces invalid characters with underscores.
        """
        # Pattern of invalid characters in filenames
        invalid_chars = r'[<>:"/\\|?*\s]'
        return re.sub(invalid_chars, '_', str(name))
        
    def _load_data(self, input_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.
        
        Args:
            input_path: Path to the input file
            sheet_name: Sheet name for Excel files
            
        Returns:
            Pandas DataFrame containing the data
        """
        file_extension = os.path.splitext(input_path)[1].lower()
        
        try:
            # Handle CSV files
            if file_extension in ['.csv', '.txt']:
                # Try UTF-8 encoding first
                try:
                    df = pd.read_csv(input_path)
                except UnicodeDecodeError:
                    # Try Latin-1 encoding if UTF-8 fails
                    df = pd.read_csv(input_path, encoding='latin1')
                except pd.errors.ParserError:
                    # Try with different separator
                    try:
                        df = pd.read_csv(input_path, sep=';')
                    except UnicodeDecodeError:
                        df = pd.read_csv(input_path, sep=';', encoding='latin1')
            
            # Handle Excel files
            elif file_extension in ['.xlsx', '.xls']:
                try:
                    if sheet_name:
                        df = pd.read_excel(input_path, sheet_name=sheet_name)
                    else:
                        df = pd.read_excel(input_path)
                except Exception as e:
                    raise EDAToolException(f"Error reading Excel file: {str(e)}")
            else:
                # Unsupported file format
                raise EDAToolException(f"Unsupported file format: {file_extension}")
            
            # Convert date columns to datetime
            for col in df.columns:
                # Only try conversion for object dtype columns with reasonable number of values
                if df[col].dtype == 'object' and df[col].nunique() < len(df) / 2:
                    try:
                        # Use pandas datetime conversion with coercion to safely convert dates
                        date_series = pd.to_datetime(df[col], errors='coerce')
                        # Only convert if more than 90% of values could be converted successfully
                        if date_series.notna().mean() > 0.9:
                            df[col] = date_series
                    except (ValueError, TypeError):
                        # Keep original if conversion fails
                        pass
            
            return df
        
        except Exception as e:
            if isinstance(e, EDAToolException):
                raise
            else:
                raise EDAToolException(f"Error loading data: {str(e)}")
    
    def _create_data_summary(self, df: pd.DataFrame, output_dir: str, categorical_threshold: int) -> str:
        """
        Create and save a summary of the dataset.
        
        Args:
            df: DataFrame to summarize
            output_dir: Directory to save summary
            categorical_threshold: Max unique values to consider a column categorical
            
        Returns:
            Path to the summary file
        """
        summary_stats_path = os.path.join(output_dir, "summary_statistics.txt")
        
        with open(summary_stats_path, 'w') as f:
            f.write("=== DATA SUMMARY ===\n\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Total columns: {len(df.columns)}\n\n")
            
            f.write("=== COLUMN INFORMATION ===\n\n")
            for col in df.columns:
                f.write(f"Column: {col}\n")
                f.write(f"  Type: {df[col].dtype}\n")
                f.write(f"  Missing values: {df[col].isna().sum()} ({(df[col].isna().sum() / len(df)) * 100:.2f}%)\n")
                
                if is_numeric_dtype(df[col]):
                    try:
                        f.write(f"  Min: {df[col].min()}\n")
                        f.write(f"  Max: {df[col].max()}\n")
                        f.write(f"  Mean: {df[col].mean()}\n")
                        f.write(f"  Median: {df[col].median()}\n")
                        f.write(f"  Std: {df[col].std()}\n")
                    except (TypeError, ValueError) as e:
                        f.write(f"  Error calculating statistics: {str(e)}\n")
                
                unique_count = df[col].nunique()
                f.write(f"  Unique values: {unique_count}\n")
                
                if unique_count <= categorical_threshold or is_categorical_dtype(df[col]):
                    f.write("  Value counts:\n")
                    value_counts = df[col].value_counts().head(10)
                    for val, count in value_counts.items():
                        val_str = str(val)
                        if len(val_str) > 50:  # Truncate long string values
                            val_str = val_str[:47] + "..."
                        f.write(f"    {val_str}: {count} ({(count / len(df)) * 100:.2f}%)\n")
                    if unique_count > 10:
                        f.write(f"    ... and {unique_count - 10} more values\n")
                
                f.write("\n")
        
        return summary_stats_path
    
    def _detect_column_types(self, df: pd.DataFrame, categorical_threshold: int) -> Dict[str, List[str]]:
        """
        Detect the types of columns in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            categorical_threshold: Max unique values to consider a column categorical
            
        Returns:
            Dictionary containing lists of column names by type
        """
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        
        for col in df.columns:
            # Check if datetime
            if is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            # Check if numeric
            elif is_numeric_dtype(df[col]):
                # Numeric columns with few unique values are also considered categorical
                if df[col].nunique() <= categorical_threshold:
                    categorical_cols.append(col)
                numeric_cols.append(col)
            # Not numeric or datetime, must be categorical
            else:
                categorical_cols.append(col)
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    
    def _profile_data(self, df: pd.DataFrame) -> Dict:
        """
        Generate a profile of the data.
        
        Args:
            df: Pandas DataFrame to profile
            
        Returns:
            Dictionary containing profile information
        """
        profile = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "duplicated_rows": int(df.duplicated().sum()),
            "columns": {},
            "missing_data": {
                "total_missing_cells": int(df.isna().sum().sum()),
                "total_cells": df.size,
                "missing_percentage": float((df.isna().sum().sum() / df.size) * 100) if df.size > 0 else 0.0
            }
        }
        
        for col in df.columns:
            try:
                col_profile = {
                    "dtype": str(df[col].dtype),
                    "is_numeric": is_numeric_dtype(df[col]),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percentage": float((df[col].isna().sum() / len(df)) * 100) if len(df) > 0 else 0.0,
                    "unique_count": int(df[col].nunique())
                }
                
                if is_numeric_dtype(df[col]):
                    # Handle numeric statistics, safely handling NaN values
                    col_min = df[col].min() if not df[col].isna().all() else None
                    col_max = df[col].max() if not df[col].isna().all() else None
                    col_mean = df[col].mean() if not df[col].isna().all() else None
                    col_median = df[col].median() if not df[col].isna().all() else None
                    col_std = df[col].std() if not df[col].isna().all() else None
                    
                    # Only calculate skewness and kurtosis for columns with sufficient non-null values
                    has_valid_stats = not df[col].isna().all() and len(df[col].dropna()) > 3
                    col_skew = float(df[col].skew()) if has_valid_stats else None
                    col_kurt = float(df[col].kurtosis()) if has_valid_stats else None
                    
                    col_profile.update({
                        "min": float(col_min) if col_min is not None and not pd.isna(col_min) else None,
                        "max": float(col_max) if col_max is not None and not pd.isna(col_max) else None,
                        "mean": float(col_mean) if col_mean is not None and not pd.isna(col_mean) else None,
                        "median": float(col_median) if col_median is not None and not pd.isna(col_median) else None,
                        "std": float(col_std) if col_std is not None and not pd.isna(col_std) else None,
                        "skewness": col_skew,
                        "kurtosis": col_kurt
                    })
                
                # For categorical or low-cardinality columns, include value counts
                if col_profile["unique_count"] <= 20 or not col_profile["is_numeric"]:
                    value_counts = df[col].value_counts().head(20).to_dict()
                    col_profile["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
                
                profile["columns"][col] = col_profile
            
            except Exception as e:
                # If there's an error profiling a column, include error info but continue
                profile["columns"][col] = {
                    "dtype": str(df[col].dtype),
                    "error": f"Error profiling column: {str(e)}"
                }
        
        return profile
    
    def _visualize_distributions(
        self, 
        df: pd.DataFrame, 
        output_dir: str, 
        categorical_threshold: int, 
        max_categories_pie: int
    ) -> List[str]:
        """
        Create visualizations for the distributions of each column.
        
        Args:
            df: DataFrame to visualize
            output_dir: Directory to save visualizations
            categorical_threshold: Maximum number of unique values to consider categorical
            max_categories_pie: Maximum number of categories to display in pie charts
            
        Returns:
            List of paths to generated visualization files
        """
        generated_files = []
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Histograms for numeric columns
        if numeric_cols:
            for col in numeric_cols:
                # Skip columns with all NaN values
                if df[col].isna().all():
                    continue
                    
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Histogram with KDE
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    
                    # Add basic statistics as text
                    if len(df[col].dropna()) > 0:
                        try:
                            stats_text = (
                                f"Mean: {df[col].mean():.2f}\n"
                                f"Median: {df[col].median():.2f}\n"
                                f"Std Dev: {df[col].std():.2f}\n"
                                f"Min: {df[col].min():.2f}\n"
                                f"Max: {df[col].max():.2f}"
                            )
                            plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                                        va='top', ha='right')
                        except (ValueError, TypeError):
                            # Skip stats annotation if there's an error calculating statistics
                            pass
                    
                    safe_col_name = self._sanitize_filename(col)
                    file_path = os.path.join(output_dir, f"dist_{safe_col_name}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    generated_files.append(file_path)
                    
                    # Box plot
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=df[col].dropna())
                    plt.title(f'Box Plot of {col}')
                    plt.xlabel(col)
                    plt.grid(True, alpha=0.3)
                    
                    file_path = os.path.join(output_dir, f"boxplot_{safe_col_name}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    generated_files.append(file_path)
                except Exception as e:
                    logging.warning(f"Error creating distribution plots for column {col}: {str(e)}")
        
        # Bar charts or pie charts for categorical columns
        for col in df.columns:
            # Skip if it's a numeric column above categorical threshold
            if col in numeric_cols and df[col].nunique() > categorical_threshold:
                continue
                
            # Skip if all values are NaN
            if df[col].isna().all():
                continue
                
            try:
                value_counts = df[col].value_counts()
                
                # Skip columns with too many unique values
                if len(value_counts) > 100:
                    continue
                    
                safe_col_name = self._sanitize_filename(col)
                
                if len(value_counts) <= max_categories_pie:
                    # Pie chart for fewer categories
                    plt.figure(figsize=(12, 8))
                    
                    # Convert value_counts labels to strings for consistent display
                    plt.pie(
                        value_counts, 
                        labels=[str(x)[:20] + '...' if len(str(x)) > 20 else str(x) for x in value_counts.index], 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        shadow=True
                    )
                    plt.axis('equal')
                    plt.title(f'Distribution of {col}')
                    
                    file_path = os.path.join(output_dir, f"pie_{safe_col_name}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    generated_files.append(file_path)
                else:
                    # Bar chart for more categories (limit to top categories)
                    plt.figure(figsize=(14, 8))
                    top_categories = value_counts.head(max_categories_pie)
                    
                    # Convert index to string to handle mixed types
                    top_categories.index = [str(x) for x in top_categories.index]
                    
                    # Create bar plot
                    sns.barplot(x=top_categories.index, y=top_categories.values)
                    plt.title(f'Top {max_categories_pie} Categories of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    
                    file_path = os.path.join(output_dir, f"bar_{safe_col_name}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    generated_files.append(file_path)
            except Exception as e:
                logging.warning(f"Error creating categorical plots for column {col}: {str(e)}")
                
        # Missing values visualization
        missing_data = df.isna().sum()
        if missing_data.sum() > 0:
            try:
                plt.figure(figsize=(14, 8))
                
                # Only plot columns with missing values
                missing_cols = missing_data[missing_data > 0]
                
                # Create the barplot, with column names converted to strings
                sns.barplot(x=missing_cols.index.astype(str), y=missing_cols.values)
                plt.title('Missing Values by Column')
                plt.xlabel('Column')
                plt.ylabel('Missing Values Count')
                plt.xticks(rotation=45, ha='right')
                
                file_path = os.path.join(output_dir, "missing_values.png")
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                generated_files.append(file_path)
            except Exception as e:
                logging.warning(f"Error creating missing values plot: {str(e)}")
        
        return generated_files
    
    def _visualize_correlations(
        self, 
        df: pd.DataFrame, 
        output_dir: str, 
        correlation_method: str
    ) -> Union[str, List[str]]:
        """
        Create correlation heatmap.
        
        Args:
            df: DataFrame to visualize
            output_dir: Directory to save visualizations
            correlation_method: Method for correlation calculation
            
        Returns:
            Path to the generated correlation heatmap file or list of paths
        """
        # Get only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['number'])
        
        # Need at least 2 columns for correlation
        if numeric_df.shape[1] < 2:
            logging.warning("Not enough numeric columns for correlation analysis")
            return None
            
        # If we have too many columns, limit to the most relevant ones
        if numeric_df.shape[1] > 100:
            logging.info(f"Too many numeric columns ({numeric_df.shape[1]}) for correlation visualization. Limiting to top 100.")
            
            # Use columns with most non-null values
            non_null_counts = numeric_df.count()
            top_cols = non_null_counts.nlargest(100).index.tolist()
            numeric_df = numeric_df[top_cols]
        
        try:
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr(method=correlation_method)
            
            # For large correlation matrices, break into chunks
            if numeric_df.shape[1] > 50:
                # Split into chunks of 50 columns max
                chunk_size = 50
                chunks = []
                
                for i in range(0, len(corr_matrix.columns), chunk_size):
                    chunk_cols = corr_matrix.columns[i:i+chunk_size]
                    chunks.append(corr_matrix.loc[chunk_cols, chunk_cols])
                
                # Create correlation heatmaps for each chunk
                file_paths = []
                for i, chunk in enumerate(chunks):
                    plt.figure(figsize=(20, 16))
                    mask = np.triu(np.ones_like(chunk, dtype=bool))
                    cmap = sns.diverging_palette(230, 20, as_cmap=True)
                    
                    sns.heatmap(chunk, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                              annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .5})
                    
                    plt.title(f'Correlation Matrix ({correlation_method.capitalize()}) - Part {i+1}')
                    file_path = os.path.join(output_dir, f"correlation_{correlation_method}_part{i+1}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    file_paths.append(file_path)
                
                return file_paths
            else:
                # For smaller correlation matrices
                plt.figure(figsize=(max(12, len(numeric_df.columns)), max(10, len(numeric_df.columns))))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                
                sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                          annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .5})
                
                plt.title(f'Correlation Matrix ({correlation_method.capitalize()})')
                file_path = os.path.join(output_dir, f"correlation_{correlation_method}.png")
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()  # Fixed: Added parentheses to close the plot
                
                return file_path
        except Exception as e:
            logging.error(f"Error in correlation analysis: {str(e)}")
            return None
    
    def _visualize_pairplot(
        self, 
        df: pd.DataFrame, 
        numeric_cols: List[str], 
        output_dir: str, 
        target_column: Optional[str] = None
    ) -> str:
        """
        Create pairplot for numerical columns.
        
        Args:
            df: DataFrame to visualize
            numeric_cols: List of numeric columns to include
            output_dir: Directory to save visualization
            target_column: Target column for coloring (if applicable)
            
        Returns:
            Path to the generated pairplot file
        """
        # Limit the number of columns for pairplot to avoid performance issues
        if len(numeric_cols) > 10:
            logging.info(f"Too many numeric columns ({len(numeric_cols)}) for pairplot. Limiting to 10 columns.")
            numeric_cols = numeric_cols[:10]
            
        # Check if we have enough data
        if not numeric_cols or len(df) < 2:
            logging.warning("Not enough data for pairplot visualization")
            return None
            
        # Prepare data for plotting, handling potential issues
        try:
            plot_df = df[numeric_cols].copy()
            
            # Check for non-finite values and replace with NaN
            for col in plot_df.columns:
                if not np.isfinite(plot_df[col]).all():
                    plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
            
            # If target column exists and is categorical (for coloring)
            hue = None
            if target_column and target_column in df.columns:
                # Only use target for coloring if it has a reasonable number of categories
                if df[target_column].nunique() <= 10:
                    plot_df[target_column] = df[target_column]
                    hue = target_column
                
            # Create pairplot
            g = sns.pairplot(
                plot_df.dropna(), 
                hue=hue, 
                diag_kind='kde', 
                plot_kws={'alpha': 0.6}, 
                diag_kws={'alpha': 0.6}
            )
            
            plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=16)
            file_path = os.path.join(output_dir, "pairplot.png")
            plt.savefig(file_path)
            plt.close()
            
            return file_path
        except Exception as e:
            logging.error(f"Error creating pairplot: {str(e)}")
            # Try to create a correlation heatmap as a fallback
            try:
                return self._visualize_correlations(df[numeric_cols], output_dir, "pearson")
            except:
                return None
    
    def _visualize_target_relationships(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        output_dir: str, 
        categorical_threshold: int
    ) -> List[str]:
        """
        Create visualizations showing relationships between features and target.
        
        Args:
            df: DataFrame to visualize
            target_column: Target column for relationship analysis
            output_dir: Directory to save visualizations
            categorical_threshold: Maximum number of unique values to consider categorical
            
        Returns:
            List of paths to generated visualization files
        """
        if target_column not in df.columns:
            logging.warning(f"Target column '{target_column}' not found in dataframe")
            return []
            
        generated_files = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Determine if target is categorical or numerical
        target_is_categorical = df[target_column].nunique() <= categorical_threshold or not is_numeric_dtype(df[target_column])
        
        for col in df.columns:
            # Skip if column is the target or has all NaN values
            if col == target_column or df[col].isna().all():
                continue
                
            # Create a safe filename for the column
            safe_col_name = self._sanitize_filename(col)
                
            # Check if feature is categorical
            col_is_categorical = col not in numeric_cols or df[col].nunique() <= categorical_threshold
            
            try:
                # Case 1: Categorical target vs. Numerical feature
                if target_is_categorical and not col_is_categorical:
                    plt.figure(figsize=(14, 8))
                    try:
                        # Convert target to string for consistent display
                        sns.boxplot(x=df[target_column].astype(str), y=col, data=df)
                        plt.title(f'Distribution of {col} by {target_column}')
                        plt.xticks(rotation=45, ha='right')
                        
                        file_path = os.path.join(output_dir, f"target_box_{safe_col_name}.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                        
                        # Add violin plot for more detail
                        plt.figure(figsize=(14, 8))
                        sns.violinplot(x=df[target_column].astype(str), y=col, data=df)
                        plt.title(f'Violin Plot of {col} by {target_column}')
                        plt.xticks(rotation=45, ha='right')
                        
                        file_path = os.path.join(output_dir, f"target_violin_{safe_col_name}.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                    except Exception as e:
                        logging.warning(f"Error creating target relationship plots (case 1) for {col}: {str(e)}")
                    
                # Case 2: Numerical target vs. Numerical feature
                elif not target_is_categorical and not col_is_categorical:
                    try:
                        plt.figure(figsize=(12, 8))
                        sns.scatterplot(x=col, y=target_column, data=df, alpha=0.6)
                        
                        # Add regression line
                        try:
                            sns.regplot(x=col, y=target_column, data=df, scatter=False, line_kws={"color": "red"})
                        except:
                            pass
                            
                        plt.title(f'Relationship between {col} and {target_column}')
                        
                        file_path = os.path.join(output_dir, f"target_scatter_{safe_col_name}.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                    except Exception as e:
                        logging.warning(f"Error creating target relationship plots (case 2) for {col}: {str(e)}")
                    
                # Case 3: Categorical target vs. Categorical feature
                elif target_is_categorical and col_is_categorical:
                    try:
                        # Create grouped bar chart
                        plt.figure(figsize=(14, 10))
                        
                        # Create a crosstab
                        cross_tab = pd.crosstab(df[col], df[target_column], normalize='index')
                        cross_tab.plot(kind='bar', stacked=True)
                        
                        plt.title(f'Relationship between {col} and {target_column}')
                        plt.xlabel(col)
                        plt.ylabel('Proportion')
                        plt.xticks(rotation=45, ha='right')
                        plt.legend(title=target_column)
                        
                        file_path = os.path.join(output_dir, f"target_bar_{safe_col_name}.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                        
                        # Heatmap of association
                        plt.figure(figsize=(12, 8))
                        cross_tab_counts = pd.crosstab(df[col], df[target_column])
                        sns.heatmap(cross_tab_counts, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'Heatmap of {col} vs {target_column}')
                        plt.tight_layout()
                        
                        file_path = os.path.join(output_dir, f"target_heatmap_{safe_col_name}.png")
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                    except Exception as e:
                        logging.warning(f"Error creating target relationship plots (case 3) for {col}: {str(e)}")
                    
                # Case 4: Numerical target vs. Categorical feature
                elif not target_is_categorical and col_is_categorical:
                    try:
                        plt.figure(figsize=(14, 8))
                        sns.boxplot(x=col, y=target_column, data=df)
                        plt.title(f'Distribution of {target_column} by {col}')
                        plt.xticks(rotation=45, ha='right')
                        
                        file_path = os.path.join(output_dir, f"target_revbox_{safe_col_name}.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                    except Exception as e:
                        logging.warning(f"Error creating target relationship plots (case 4) for {col}: {str(e)}")
            except Exception as e:
                logging.warning(f"Error processing target relationships for column {col}: {str(e)}")
        
        return generated_files
    
    def _time_series_analysis(
        self, 
        df: pd.DataFrame, 
        datetime_cols: List[str], 
        output_dir: str
    ) -> List[str]:
        """
        Perform time series analysis for datetime columns.
        
        Args:
            df: DataFrame to analyze
            datetime_cols: List of datetime columns to analyze
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualization files
        """
        generated_files = []
        
        for date_col in datetime_cols:
            # Ensure column is datetime type
            if not is_datetime64_any_dtype(df[date_col]):
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    # Skip if conversion fails
                    continue
            
            try:
                # Sort by date
                df_sorted = df.sort_values(by=date_col)
                
                # Check if there are numeric columns to plot
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    continue
                    
                # Create a safe filename for the column
                safe_date_col = self._sanitize_filename(date_col)
                    
                # For each numeric column, create time series plot
                for num_col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    safe_num_col = self._sanitize_filename(num_col)
                    
                    # Skip if all values are NaN
                    if df[num_col].isna().all():
                        continue
                    
                    plt.figure(figsize=(16, 6))
                    
                    try:
                        # Plot time series
                        plt.plot(df_sorted[date_col], df_sorted[num_col])
                        plt.title(f'Time Series of {num_col} by {date_col}')
                        plt.xlabel(date_col)
                        plt.ylabel(num_col)
                        plt.grid(True, alpha=0.3)
                        
                        # Format x-axis dates
                        plt.gcf().autofmt_xdate()
                        
                        file_path = os.path.join(output_dir, f"timeseries_{safe_date_col}_{safe_num_col}.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        generated_files.append(file_path)
                    except Exception as e:
                        logging.warning(f"Error creating time series plot for {num_col} by {date_col}: {str(e)}")
                        plt.close()
                
                # Create count plot by time periods
                try:
                    plt.figure(figsize=(16, 6))
                    
                    # Extract date components and count by period
                    if len(df) > 1000:
                        # For large datasets, aggregate by month
                        date_counts = df[date_col].dt.to_period('M').value_counts().sort_index()
                        date_counts.index = date_counts.index.astype(str)
                    else:
                        # For smaller datasets, aggregate by day
                        date_counts = df[date_col].dt.date.value_counts().sort_index()
                        # Convert index to strings
                        date_counts.index = [str(d) for d in date_counts.index]
                    
                    plt.bar(date_counts.index, date_counts.values)
                    plt.title(f'Counts by {date_col}')
                    plt.xlabel(date_col)
                    plt.ylabel('Count')
                    plt.grid(True, alpha=0.3)
                    
                    # Format x-axis
                    plt.xticks(rotation=45, ha='right')
                    
                    file_path = os.path.join(output_dir, f"datecount_{safe_date_col}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    generated_files.append(file_path)
                except Exception as e:
                    logging.warning(f"Error creating date count plot for {date_col}: {str(e)}")
                    plt.close()
            except Exception as e:
                logging.warning(f"Error performing time series analysis for {date_col}: {str(e)}")
        
        return generated_files
    
    def _detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """
        Detect outliers in numerical columns.
        
        Args:
            df: DataFrame to analyze
            numeric_cols: List of numeric columns to check for outliers
            
        Returns:
            Dictionary containing outlier information
        """
        outlier_info = {}
        
        for col in numeric_cols:
            # Skip columns with all NaN values or less than 10 non-null values
            if df[col].isna().all() or df[col].count() < 10:
                continue
                
            try:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Find outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                # Calculate outlier statistics
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / df[col].count()) * 100 if df[col].count() > 0 else 0
                
                outlier_info[col] = {
                    "Q1": float(Q1),
                    "Q3": float(Q3),
                    "IQR": float(IQR),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": float(outlier_percentage),
                    "min_outlier": float(outliers.min()) if not outliers.empty else None,
                    "max_outlier": float(outliers.max()) if not outliers.empty else None
                }
            except Exception as e:
                logging.warning(f"Error detecting outliers for column {col}: {str(e)}")
            
        return outlier_info
    
    def _visualize_outliers(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: str) -> List[str]:
        """
        Create visualizations for outlier detection.
        
        Args:
            df: DataFrame to visualize
            numeric_cols: List of numeric columns to analyze
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualization files
        """
        generated_files = []
        
        for col in numeric_cols:
            # Skip columns with all NaN values or less than 10 non-null values
            if df[col].isna().all() or df[col].count() < 10:
                continue
                
            try:
                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
                
                # Box plot for outlier visualization
                sns.boxplot(x=df[col], ax=ax1)
                ax1.set_title(f'Box Plot with Outliers: {col}')
                ax1.grid(True, alpha=0.3)
                
                # Histogram with KDE for distribution with outliers
                sns.histplot(df[col].dropna(), kde=True, ax=ax2)
                ax2.set_title(f'Distribution with Outliers: {col}')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                safe_col_name = self._sanitize_filename(col)
                file_path = os.path.join(output_dir, f"outliers_{safe_col_name}.png")
                plt.savefig(file_path)
                plt.close(fig)  # Close the figure explicitly
                generated_files.append(file_path)
            except Exception as e:
                logging.warning(f"Error creating outlier visualization for column {col}: {str(e)}")
            
        return generated_files
    
    def _create_summary_report(
        self, 
        df: pd.DataFrame, 
        profile_results: Dict, 
        outlier_info: Dict, 
        generated_files: List[str],
        output_dir: str,
        column_types: Dict[str, List[str]]
    ) -> str:
        """
        Create a comprehensive summary report.
        
        Args:
            df: DataFrame analyzed
            profile_results: Data profile information
            outlier_info: Outlier detection results
            generated_files: List of generated visualization files
            output_dir: Directory to save the report
            column_types: Dictionary of column types
            
        Returns:
            Path to the generated report file
        """
        report_path = os.path.join(output_dir, "eda_summary_report.txt")
        
        # Extract column lists by type
        numeric_cols = column_types['numeric']
        cat_cols = column_types['categorical']
        datetime_cols = column_types['datetime']
        
        with open(report_path, 'w') as f:
            f.write("============================================\n")
            f.write("    EXPLORATORY DATA ANALYSIS REPORT        \n")
            f.write("============================================\n\n")
            
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. DATASET OVERVIEW\n")
            f.write("===================\n\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Total columns: {len(df.columns)}\n")
            f.write(f"Memory usage: {profile_results['memory_usage'] / (1024*1024):.2f} MB\n")
            f.write(f"Duplicate rows: {profile_results['duplicated_rows']}\n")
            f.write(f"Missing cells: {profile_results['missing_data']['total_missing_cells']} ({profile_results['missing_data']['missing_percentage']:.2f}%)\n\n")
            
            f.write("2. COLUMN SUMMARY\n")
            f.write("=================\n\n")
            
            # Numeric columns
            f.write(f"Numeric columns ({len(numeric_cols)}):\n")
            for col in numeric_cols:
                f.write(f"  - {col}\n")
            f.write("\n")
            
            # Categorical columns
            f.write(f"Categorical columns ({len(cat_cols)}):\n")
            for col in cat_cols:
                f.write(f"  - {col}\n")
            f.write("\n")
            
            # Datetime columns
            if datetime_cols:
                f.write(f"Datetime columns ({len(datetime_cols)}):\n")
                for col in datetime_cols:
                    f.write(f"  - {col}\n")
                f.write("\n")
            
            # Columns with missing values
            missing_cols = [col for col in df.columns if df[col].isna().any()]
            if missing_cols:
                f.write(f"Columns with missing values ({len(missing_cols)}):\n")
                for col in missing_cols:
                    missing_count = df[col].isna().sum()
                    missing_percent = (missing_count / len(df)) * 100
                    f.write(f"  - {col}: {missing_count} ({missing_percent:.2f}%)\n")
                f.write("\n")
            
            f.write("3. STATISTICAL SUMMARY\n")
            f.write("=====================\n\n")
            
            # Add numeric column statistics
            for col in numeric_cols:
                profile = profile_results['columns'].get(col, {})
                if 'error' in profile:
                    f.write(f"Column: {col}\n")
                    f.write(f"  - Error: {profile['error']}\n\n")
                    continue
                    
                f.write(f"Column: {col}\n")
                f.write(f"  - Type: {profile.get('dtype', 'Unknown')}\n")
                f.write(f"  - Missing: {profile.get('missing_count', 'N/A')} ({profile.get('missing_percentage', 'N/A'):.2f}%)\n")
                f.write(f"  - Unique values: {profile.get('unique_count', 'N/A')}\n")
                f.write(f"  - Min: {profile.get('min', 'N/A')}\n")
                f.write(f"  - Max: {profile.get('max', 'N/A')}\n")
                f.write(f"  - Mean: {profile.get('mean', 'N/A')}\n")
                f.write(f"  - Median: {profile.get('median', 'N/A')}\n")
                f.write(f"  - Std Dev: {profile.get('std', 'N/A')}\n")
                f.write(f"  - Skewness: {profile.get('skewness', 'N/A')}\n")
                f.write(f"  - Kurtosis: {profile.get('kurtosis', 'N/A')}\n")
                f.write("\n")
            
            # Add categorical column statistics
            for col in cat_cols:
                if col in numeric_cols:
                    continue  # Skip numeric columns already covered
                    
                profile = profile_results['columns'].get(col, {})
                if 'error' in profile:
                    f.write(f"Column: {col}\n")
                    f.write(f"  - Error: {profile['error']}\n\n")
                    continue
                    
                f.write(f"Column: {col}\n")
                f.write(f"  - Type: {profile.get('dtype', 'Unknown')}\n")
                f.write(f"  - Missing: {profile.get('missing_count', 'N/A')} ({profile.get('missing_percentage', 'N/A'):.2f}%)\n")
                f.write(f"  - Unique values: {profile.get('unique_count', 'N/A')}\n")
                
                value_counts = profile.get('value_counts', {})
                if value_counts:
                    f.write("  - Top categories:\n")
                    for val, count in list(value_counts.items())[:10]:
                        # Truncate long values for better readability
                        val_str = val[:50] + '...' if len(val) > 50 else val
                        percent = (count / len(df)) * 100
                        f.write(f"    * {val_str}: {count} ({percent:.2f}%)\n")
                f.write("\n")
            
            # Add outlier information
            if outlier_info:
                f.write("4. OUTLIER DETECTION\n")
                f.write("===================\n\n")
                
                for col, info in outlier_info.items():
                    f.write(f"Column: {col}\n")
                    f.write(f"  - IQR: {info['IQR']:.2f}\n")
                    f.write(f"  - Lower bound: {info['lower_bound']:.2f}\n")
                    f.write(f"  - Upper bound: {info['upper_bound']:.2f}\n")
                    f.write(f"  - Outlier count: {info['outlier_count']} ({info['outlier_percentage']:.2f}%)\n")
                    if info['outlier_count'] > 0:
                        f.write(f"  - Min outlier: {info['min_outlier']}\n")
                        f.write(f"  - Max outlier: {info['max_outlier']}\n")
                    f.write("\n")
            
            # Generated visualizations summary
            f.write("5. GENERATED VISUALIZATIONS\n")
            f.write("==========================\n\n")
            
            # Group visualizations by type
            viz_types = {
                "Distributions": [f for f in generated_files if "dist_" in os.path.basename(f)],
                "Box plots": [f for f in generated_files if "boxplot_" in os.path.basename(f)],
                "Pie charts": [f for f in generated_files if "pie_" in os.path.basename(f)],
                "Bar charts": [f for f in generated_files if "bar_" in os.path.basename(f) and not "target_" in os.path.basename(f)],
                "Correlation": [f for f in generated_files if "correlation_" in os.path.basename(f)],
                "Pairplot": [f for f in generated_files if "pairplot" in os.path.basename(f)],
                "Target relationships": [f for f in generated_files if "target_" in os.path.basename(f)],
                "Time series": [f for f in generated_files if "timeseries_" in os.path.basename(f) or "datecount_" in os.path.basename(f)],
                "Outliers": [f for f in generated_files if "outliers_" in os.path.basename(f)],
                "Missing values": [f for f in generated_files if "missing_values" in os.path.basename(f)]
            }
            
            for viz_type, files in viz_types.items():
                if files:
                    f.write(f"{viz_type} ({len(files)}):\n")
                    for file_path in files:
                        f.write(f"  - {os.path.basename(file_path)}\n")
                    f.write("\n")
            
            # Final insights
            f.write("6. KEY INSIGHTS\n")
            f.write("==============\n\n")
            
            # Missing data insights
            if profile_results['missing_data']['total_missing_cells'] > 0:
                missing_pct = profile_results['missing_data']['missing_percentage']
                if missing_pct > 20:
                    f.write("- High level of missing data detected (>20%). Consider imputation strategies.\n")
                elif missing_pct > 5:
                    f.write("- Moderate level of missing data detected (5-20%). Review columns with missing values.\n")
                else:
                    f.write("- Low level of missing data detected (<5%).\n")
            else:
                f.write("- No missing data detected.\n")
            
            # Outlier insights
            if outlier_info:
                outlier_cols = [col for col, info in outlier_info.items() if info['outlier_percentage'] > 5]
                if outlier_cols:
                    f.write(f"- Significant outliers detected in {len(outlier_cols)} columns: {', '.join(outlier_cols)}.\n")
                    f.write("  These may impact statistical models and should be investigated.\n")
            
            # Distribution insights
            skewed_cols = [col for col in numeric_cols 
                         if 'skewness' in profile_results['columns'].get(col, {}) 
                         and profile_results['columns'][col]['skewness'] is not None
                         and abs(profile_results['columns'][col]['skewness']) > 1]
            if skewed_cols:
                f.write(f"- Highly skewed distributions detected in {len(skewed_cols)} columns.\n")
                f.write("  Consider transformations (log, sqrt, etc.) for modeling.\n")
            
            # Duplicate data insight
            if profile_results['duplicated_rows'] > 0:
                dup_pct = (profile_results['duplicated_rows'] / len(df)) * 100
                f.write(f"- {profile_results['duplicated_rows']} duplicate rows detected ({dup_pct:.2f}%).\n")
            
            f.write("\n============================================\n")
            f.write("End of report\n")
        
        return report_path

#Feature Importance and Feature Selection Tool

class FeatureImportanceAnalysisTool(Tool):
    name = "feature_importance_analysis"
    description = """
    This tool performs feature importance analysis on tabular data for classification or regression tasks.
    It identifies the most relevant features using various methods and outputs CSV files with selected features.
    Optionally generates visualization plots for feature importance and data distribution.
    """
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input data file (CSV format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory where results will be saved"
        },
        "target_column": {
            "type": "string",
            "description": "Name of the target column for prediction"
        },
        "task_type": {
            "type": "string",
            "description": "Type of machine learning task ('classification' or 'regression')",
            "required": False,
            "nullable": True
        },
        "method": {
            "type": "string",
            "description": "Feature selection method ('rf', 'f_test', 'mutual_info', 'rfe')",
            "required": False,
            "nullable": True
        },
        "top_features": {
            "type": "string",
            "description": "Comma-separated list of top features counts to select (e.g., '10,50,100')",
            "required": False,
            "nullable": True
        },
        "encode_categorical": {
            "type": "string",
            "description": "Method to encode categorical features ('auto', 'onehot', 'label', 'target', 'none')",
            "required": False,
            "nullable": True
        },
        "max_onehot_cardinality": {
            "type": "integer",
            "description": "Maximum unique values for one-hot encoding if encode_categorical='auto'",
            "required": False,
            "nullable": True
        },
        "create_plots": {
            "type": "boolean",
            "description": "Whether to generate visualization plots",
            "required": False,
            "nullable": True
        },
        "top_n_plot": {
            "type": "integer",
            "description": "Number of top features to show in importance plots",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        output_dir: str,
        target_column: str,
        task_type: Optional[str] = None,
        method: Optional[str] = "rf",
        top_features: Optional[str] = "10,50,100",
        encode_categorical: Optional[str] = "auto",
        max_onehot_cardinality: Optional[int] = 10,
        create_plots: Optional[bool] = True,
        top_n_plot: Optional[int] = 30
    ):
        """
        Analyze feature importance for tabular data and output results.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save results
            target_column: Target column name for prediction
            task_type: 'classification' or 'regression' (auto-detected if None)
            method: Feature selection method ('rf', 'f_test', 'mutual_info', 'rfe')
            top_features: Comma-separated list of top features to select
            encode_categorical: Method for encoding categorical features
            max_onehot_cardinality: Max unique values for one-hot encoding if auto
            create_plots: Whether to generate visualization plots
            top_n_plot: Number of top features to show in importance plots
            
        Returns:
            Dictionary with analysis results and file paths
        """
        # Validate input parameters
        self._validate_parameters(input_path, output_dir, target_column, task_type, method, 
                               encode_categorical, top_features)
        
        # Create output directory first
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logger and log file
        log_file = os.path.join(output_dir, "feature_importance_analysis.log")
        logger = self._setup_logging(log_file)
        
        try:
            logger.info(f"Starting feature importance analysis for {input_path}")
            logger.info(f"Target column: {target_column}")
            
            # Load data and prepare it for analysis
            X, y, original_df, X_original, detected_task_type = self._load_and_prepare_data(
                input_path, 
                target_column, 
                task_type,
                encode_categorical, 
                max_onehot_cardinality,
                logger
            )
            
            if task_type is None:
                task_type = detected_task_type
                logger.info(f"Auto-detected task type: {task_type}")
            
            # Parse the list of top features counts
            top_features_list = [int(n) for n in top_features.split(',')]
            top_features_list.sort()  # Ensure ascending order
            
            # Apply feature selection based on the method
            result_files = self._perform_feature_selection(
                X, y, original_df, X_original,
                output_dir, target_column, task_type,
                method, top_features_list, create_plots, top_n_plot,
                logger
            )
            
            logger.info("Feature importance analysis completed successfully")
            
            return {
                "status": "success",
                "input_path": input_path,
                "output_dir": output_dir,
                "target_column": target_column,
                "task_type": task_type,
                "method": method,
                "result_files": result_files,
                "feature_count": X.shape[1],
                "log_file": log_file
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in feature importance analysis: {error_msg}", exc_info=True)
            return {
                "status": "error",
                "error_message": error_msg,
                "input_path": input_path,
                "output_dir": output_dir,
                "log_file": log_file
            }
    
    def _validate_parameters(self, input_path, output_dir, target_column, task_type, method, 
                         encode_categorical, top_features):
        """
        Validate input parameters to catch errors early.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save results
            target_column: Target column name
            task_type: Task type
            method: Feature selection method
            encode_categorical: Method to encode categorical features
            top_features: Comma-separated list of top features to select
        """
        # Check if input file exists and is a CSV
        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")
        
        if not input_path.lower().endswith('.csv'):
            raise ValueError(f"Input file must be a CSV file: {input_path}")
        
        # Validate task_type if provided
        if task_type is not None and task_type not in ['classification', 'regression']:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        
        # Validate method
        valid_methods = ['rf', 'f_test', 'mutual_info', 'rfe']
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
        
        # Validate encoding method
        valid_encodings = ['auto', 'onehot', 'label', 'target', 'none']
        if encode_categorical not in valid_encodings:
            raise ValueError(f"Invalid encode_categorical: {encode_categorical}. Must be one of {valid_encodings}")
        
        # Validate top_features format
        if not re.match(r'^(\d+,)*\d+$', top_features):
            raise ValueError(f"Invalid top_features format: {top_features}. Must be comma-separated integers (e.g., '10,50,100')")
    
    def _setup_logging(self, log_file):
        """
        Set up a logger specific to this instance to avoid modifying the root logger.
        
        Args:
            log_file: Path to log file
            
        Returns:
            Logger instance
        """
        # Create logger with unique name
        logger_name = f"feature_importance_{os.path.basename(log_file)}_{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Remove all existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode to overwrite existing log
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Make logger not propagate messages to parent loggers (root)
        # This prevents duplicate logs
        logger.propagate = False
        
        return logger
    
    def _load_and_prepare_data(
        self, 
        input_path: str, 
        target_column: str, 
        task_type: Optional[str],
        encode_categorical: str, 
        max_onehot_cardinality: int,
        logger: logging.Logger
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, str]:
        """
        Load data, separate features from target, and handle categorical features.
        
        Args:
            input_path: Path to input CSV file
            target_column: Target column name
            task_type: Task type ('classification' or 'regression')
            encode_categorical: Method to encode categorical features
            max_onehot_cardinality: Max unique values for one-hot encoding
            logger: Logger instance
            
        Returns:
            Tuple of (X processed, y, original dataframe, X original, detected task type)
        """
        logger.info(f"Loading data from {input_path}")
        
        try:
            # Load data
            df = pd.read_csv(input_path)
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Detect task type if not specified
            detected_task_type = self._detect_task_type(y) if task_type is None else task_type
            
            # Save original X for feature mapping
            X_original = X.copy()
            
            # Handle categorical features
            if encode_categorical != 'none':
                X = self._encode_categorical_features(X, y, target_column, encode_categorical, 
                                                 max_onehot_cardinality, input_path, logger)
            else:
                # Remove categorical columns if encoding is disabled
                cat_cols = X.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    logger.warning(f"Removing {len(cat_cols)} categorical columns as encoding is disabled")
                    X = X.select_dtypes(include=[np.number])
            
            # Handle missing values
            if X.isna().any().any():
                logger.warning(f"Dataset contains missing values. Filling with appropriate values.")
                # For numeric columns, fill with median
                num_cols = X.select_dtypes(include=[np.number]).columns
                X[num_cols] = X[num_cols].fillna(X[num_cols].median())
                
                # For any remaining columns, fill with mode
                for col in X.columns:
                    if col not in num_cols:
                        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
            
            logger.info(f"Data loaded successfully: {X.shape[0]} rows, {X.shape[1]} features")
            
            return X, y, df, X_original, detected_task_type
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """
        Automatically detect whether the task is classification or regression.
        
        Args:
            y: Target variable
            
        Returns:
            'classification' or 'regression'
        """
        # If target is object type, it's definitely classification
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        
        # Check number of unique values relative to the length of the series
        unique_ratio = len(y.unique()) / len(y)
        
        # Heuristic: if less than 5% unique values or fewer than 10 distinct values, likely classification
        if unique_ratio < 0.05 or len(y.unique()) < 10:
            return 'classification'
        else:
            return 'regression'
    
    def _encode_categorical_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        target_column: str,
        encode_method: str, 
        max_onehot_cardinality: int,
        input_path: str,
        logger: logging.Logger
    ) -> pd.DataFrame:
        """
        Encode categorical features using the specified method.
        
        Args:
            X: Feature dataframe
            y: Target variable
            target_column: Name of target column (for target encoding)
            encode_method: Encoding method
            max_onehot_cardinality: Maximum cardinality for one-hot encoding
            input_path: Path to input file (for saving encoding mappings)
            logger: Logger instance
            
        Returns:
            Processed dataframe with encoded features
        """
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            logger.info("No categorical features found")
            return X
        
        logger.info(f"Found {len(cat_cols)} categorical features. Encoding method: {encode_method}")
        
        # Create a copy to work with
        X_encoded = X.copy()
        
        # Store encoding mappings for interpretability
        encoding_mappings = {}
        
        for col in cat_cols:
            unique_count = X[col].nunique()
            logger.info(f"Column '{col}' has {unique_count} unique values")
            
            # Determine encoding method if auto
            method = encode_method
            if encode_method == 'auto':
                if unique_count <= max_onehot_cardinality:
                    method = 'onehot'
                else:
                    method = 'label'
            
            # Apply the selected encoding method
            if method == 'onehot':
                logger.info(f"Using one-hot encoding for '{col}'")
                # Get dummies and drop original column
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                # Store mapping info
                encoding_mappings[col] = {'method': 'onehot', 'categories': X[col].unique().tolist()}
                
            elif method == 'label':
                logger.info(f"Using label encoding for '{col}'")
                le = LabelEncoder()
                X_encoded[f"{col}_label"] = le.fit_transform(X[col].astype(str))
                # Store mapping
                encoding_mappings[col] = {
                    'method': 'label',
                    'mapping': dict(zip(le.classes_, range(len(le.classes_))))
                }
                
            elif method == 'target':
                logger.info(f"Using target encoding for '{col}'")
                # Create temporary df with target for encoding
                temp_df = pd.concat([X[[col]], y], axis=1)
                
                # Target encoding (mean target value per category)
                means = temp_df.groupby(col)[target_column].mean()
                X_encoded[f"{col}_target"] = X[col].map(means)
                
                # Add frequency encoding as well
                freq = X[col].value_counts(normalize=True)
                X_encoded[f"{col}_freq"] = X[col].map(freq)
                
                # Store mapping
                encoding_mappings[col] = {
                    'method': 'target',
                    'target_means': means.to_dict(),
                    'frequency': freq.to_dict()
                }
        
        # Drop original categorical columns after encoding
        X_encoded = X_encoded.drop(columns=cat_cols)
        
        # Handle any NaN values from the encoding process
        if X_encoded.isna().any().any():
            missing_count = X_encoded.isna().sum().sum()
            logger.warning(f"Encoding created {missing_count} missing values. Filling with zeros.")
            X_encoded = X_encoded.fillna(0)
        
        # Save encoding mappings for interpretability
        # Use the output directory from the input path instead of a hardcoded directory
        output_dir = os.path.dirname(os.path.abspath(input_path))
        encoding_file = os.path.join(output_dir, 'categorical_encodings.json')
        
        with open(encoding_file, 'w') as f:
            json.dump(encoding_mappings, f, indent=2, default=str)
        logger.info(f"Saved categorical encoding mappings to {encoding_file}")
        
        return X_encoded
    
    def _perform_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        original_df: pd.DataFrame,
        X_original: pd.DataFrame,
        output_dir: str,
        target_column: str,
        task_type: str,
        method: str,
        top_features_list: List[int],
        create_plots: bool,
        top_n_plot: int,
        logger: logging.Logger
    ) -> Dict[str, str]:
        """
        Perform feature selection using the specified method.
        
        Args:
            X: Processed feature dataframe
            y: Target variable
            original_df: Original dataframe with all columns
            X_original: Original features before encoding
            output_dir: Directory to save results
            target_column: Target column name
            task_type: 'classification' or 'regression'
            method: Feature selection method
            top_features_list: List of top feature counts to select
            create_plots: Whether to create visualization plots
            top_n_plot: Number of top features to show in plots
            logger: Logger instance
            
        Returns:
            Dictionary of output file paths
        """
        result_files = {}
        
        # Define available methods with their functions
        if task_type == 'classification':
            methods = {
                'rf': self._select_features_rf_classifier,
                'f_test': self._select_features_f_test_classifier,
                'mutual_info': self._select_features_mutual_info_classifier,
                'rfe': self._select_features_rfe_classifier
            }
        else:  # regression
            methods = {
                'rf': self._select_features_rf_regressor,
                'f_test': self._select_features_f_test_regressor,
                'mutual_info': self._select_features_mutual_info_regressor,
                'rfe': self._select_features_rfe_regressor
            }
        
        # Check if method is valid
        if method not in methods:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        selection_method = methods[method]
        logger.info(f"Using {method} method for feature selection")
        
        # Create feature importance file
        importance_file = os.path.join(output_dir, 'feature_importance.csv')
        result_files['importance'] = importance_file
        
        # Output files for selected features
        selected_files = {}
        
        # For RFE, we need to run the algorithm separately for each n_features
        if method == 'rfe':
            logger.info("Using Recursive Feature Elimination (RFE)")
            selected_features_all = {}
            
            for n_features in top_features_list:
                if n_features > X.shape[1]:
                    logger.warning(f"Requested {n_features} features, but only {X.shape[1]} are available")
                    n_features = X.shape[1]
                
                # Get selected features
                selected_features, scores = selection_method(X, y, n_features, logger)
                selected_features_all[n_features] = selected_features
                
                # Save selected features
                output_file = self._save_selected_features(
                    original_df, target_column, selected_features, 
                    output_dir, n_features, X_original, X, logger
                )
                selected_files[n_features] = output_file
                
                # Generate visualizations if requested
                if create_plots and n_features > 1 and n_features <= 500:
                    self._create_visualizations(X, y, selected_features, output_dir, n_features, task_type, logger)
            
            # For RFE, create an importance file using RF for visualization purposes
            if create_plots:
                all_features, scores = (self._select_features_rf_classifier(X, y, X.shape[1], logger) if task_type == 'classification' 
                                     else self._select_features_rf_regressor(X, y, X.shape[1], logger))
                
                importance_df = pd.DataFrame({
                    'feature': all_features,
                    'importance': scores
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df.to_csv(importance_file, index=False)
                
                # Create visualization plots
                self._plot_feature_importance(importance_df, output_dir, top_n_plot, logger)
                self._plot_cumulative_importance(importance_df, output_dir, logger)
                
                # Plot correlation of top features
                top_features = importance_df['feature'].iloc[:min(20, len(importance_df))].tolist()
                self._plot_feature_correlation(X, top_features, output_dir, logger)
        
        else:  # For other methods
            # Get all features with importance values
            all_features, scores = selection_method(X, y, X.shape[1], logger)
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': scores
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df.to_csv(importance_file, index=False)
            logger.info(f"Saved feature importance to {importance_file}")
            
            # Generate visualization plots if requested
            if create_plots:
                logger.info("Generating visualization plots...")
                
                # Plot feature importance
                self._plot_feature_importance(importance_df, output_dir, top_n_plot, logger)
                
                # Plot cumulative importance
                thresholds = self._plot_cumulative_importance(importance_df, output_dir, logger)
                logger.info(f"Feature threshold analysis: {thresholds}")
                
                # Plot correlation of top features
                top_features = importance_df['feature'].iloc[:min(20, len(importance_df))].tolist()
                self._plot_feature_correlation(X, top_features, output_dir, logger)
            
            # For each number of features, save a CSV and create visualizations
            for n_features in top_features_list:
                if n_features > X.shape[1]:
                    logger.warning(f"Requested {n_features} features, but only {X.shape[1]} are available")
                    n_features = X.shape[1]
                
                # Get top n_features from importance dataframe
                selected_features = importance_df['feature'].iloc[:n_features].tolist()
                
                # Save selected features to CSV
                output_file = self._save_selected_features(
                    original_df, target_column, selected_features, 
                    output_dir, n_features, X_original, X, logger
                )
                selected_files[n_features] = output_file
                
                # Generate PCA and t-SNE visualizations if requested
                if create_plots and n_features > 1 and n_features <= 500:
                    self._create_visualizations(X, y, selected_features, output_dir, n_features, task_type, logger)
                
        # Add selected feature files to result
        result_files['selected_features'] = selected_files
        
        return result_files
    
    def _select_features_rf_classifier(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using Random Forest classifier importance"""
        logger.info(f"Selecting top {n_features} features using Random Forest classifier importance")
        
        # Create and fit Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create indices of top features
        indices = np.argsort(importances)[::-1][:n_features]
        
        return X.columns[indices].tolist(), importances
    
    def _select_features_rf_regressor(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using Random Forest regressor importance"""
        logger.info(f"Selecting top {n_features} features using Random Forest regressor importance")
        
        # Create and fit Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create indices of top features
        indices = np.argsort(importances)[::-1][:n_features]
        
        return X.columns[indices].tolist(), importances
    
    def _select_features_f_test_classifier(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using ANOVA F-value for classification"""
        logger.info(f"Selecting top {n_features} features using ANOVA F-test for classification")
        
        # Apply SelectKBest with f_classif
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature indices
        mask = selector.get_support()
        
        return X.columns[mask].tolist(), selector.scores_
    
    def _select_features_f_test_regressor(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using F-test for regression"""
        logger.info(f"Selecting top {n_features} features using F-test for regression")
        
        # Apply SelectKBest with f_regression
        selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature indices
        mask = selector.get_support()
        
        return X.columns[mask].tolist(), selector.scores_
    
    def _select_features_mutual_info_classifier(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using mutual information for classification"""
        logger.info(f"Selecting top {n_features} features using mutual information for classification")
        
        # Apply SelectKBest with mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature indices
        mask = selector.get_support()
        
        return X.columns[mask].tolist(), selector.scores_
    
    def _select_features_mutual_info_regressor(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using mutual information for regression"""
        logger.info(f"Selecting top {n_features} features using mutual information for regression")
        
        # Apply SelectKBest with mutual_info_regression
        selector = SelectKBest(mutual_info_regression, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature indices
        mask = selector.get_support()
        
        return X.columns[mask].tolist(), selector.scores_
    
    def _select_features_rfe_classifier(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using RFE with Random Forest classifier"""
        logger.info(f"Selecting top {n_features} features using RFE with classifier")
        
        try:
            # Create RFE with RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=0.1)
            
            # Fit RFE
            selector.fit(X, y)
            
            # Get selected feature indices
            mask = selector.get_support()
            
            return X.columns[mask].tolist(), selector.ranking_
        except Exception as e:
            logger.error(f"Error in RFE classification: {str(e)}. Falling back to Random Forest importance.")
            return self._select_features_rf_classifier(X, y, n_features, logger)
    
    def _select_features_rfe_regressor(self, X: pd.DataFrame, y: pd.Series, n_features: int, logger: logging.Logger) -> Tuple[List[str], np.ndarray]:
        """Select features using RFE with Random Forest regressor"""
        logger.info(f"Selecting top {n_features} features using RFE with regressor")
        
        try:
            # Create RFE with RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=0.1)
            
            # Fit RFE
            selector.fit(X, y)
            
            # Get selected feature indices
            mask = selector.get_support()
            
            return X.columns[mask].tolist(), selector.ranking_
        except Exception as e:
            logger.error(f"Error in RFE regression: {str(e)}. Falling back to Random Forest importance.")
            return self._select_features_rf_regressor(X, y, n_features, logger)
    
    def _save_selected_features(
        self, 
        original_df: pd.DataFrame, 
        target_column: str, 
        selected_features: List[str],
        output_dir: str, 
        n_features: int,
        X_original: Optional[pd.DataFrame] = None,
        X_encoded: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None
    ) -> str:
        """
        Save selected features to CSV file with interpretation of encoded features.
        
        Args:
            original_df: Original dataframe with all columns
            target_column: Target column name
            selected_features: List of selected feature names
            output_dir: Directory to save output
            n_features: Number of features selected
            X_original: Original features before encoding
            X_encoded: Encoded features
            logger: Logger instance
            
        Returns:
            Path to the output CSV file
        """
        output_file = os.path.join(output_dir, f'top_{n_features}_features.csv')
        meta_file = os.path.join(output_dir, f'top_{n_features}_features_metadata.json')
        
        # Create feature metadata for interpretability
        if X_encoded is not None and X_original is not None:
            feature_metadata = {}
            
            for feature in selected_features:
                # Check if this is an encoded feature
                if '_' in feature and any(col in feature for col in X_original.columns):
                    # This is likely an encoded feature
                    # Extract original column name and encoding method
                    
                    # Try to find the original column name
                    possible_orig_cols = []
                    for col in X_original.columns:
                        if feature.startswith(col + '_'):
                            possible_orig_cols.append(col)
                    
                    if possible_orig_cols:
                        # Sort by length to get the longest match (most specific)
                        orig_col = sorted(possible_orig_cols, key=len, reverse=True)[0]
                        encoding_type = feature[len(orig_col)+1:]
                        
                        # Store metadata about the encoding
                        feature_metadata[feature] = {
                            'original_column': orig_col,
                            'encoding_type': encoding_type
                        }
            
            # Save feature metadata for interpretability
            if feature_metadata:
                with open(meta_file, 'w') as f:
                    json.dump(feature_metadata, f, indent=2)
                if logger:
                    logger.info(f"Saved feature metadata to {meta_file}")
        
        # Prepare data to save
        try:
            # Include target column in selected features
            columns_to_save = selected_features.copy()
            if target_column not in columns_to_save:
                columns_to_save.append(target_column)
            
            # Check if all columns exist in the original dataframe
            missing_cols = [col for col in columns_to_save if col not in original_df.columns]
            
            if missing_cols:
                # Some selected features are encoded and not in original_df
                if logger:
                    logger.info(f"Creating dataset with encoded features: {', '.join(missing_cols)}")
                
                # Start with features that exist in original_df
                existing_cols = [col for col in columns_to_save if col in original_df.columns]
                selected_df = original_df[existing_cols].copy()
                
                # Add encoded features from X_encoded
                for col in missing_cols:
                    if X_encoded is not None and col in X_encoded.columns:
                        selected_df[col] = X_encoded[col]
                    else:
                        if logger:
                            logger.warning(f"Feature {col} not found in encoded or original data")
            else:
                # All selected features are in the original dataframe
                selected_df = original_df[columns_to_save].copy()
        except Exception as e:
            error_msg = str(e)
            if logger:
                logger.error(f"Error creating selected features dataset: {error_msg}")
            # Fallback: just save the selected features from X_encoded with the target
            selected_df = pd.DataFrame()
            
            if X_encoded is not None:
                for col in selected_features:
                    if col in X_encoded.columns:
                        selected_df[col] = X_encoded[col]
                selected_df[target_column] = original_df[target_column].reset_index(drop=True)
            else:
                # Last resort fallback
                selected_df = pd.DataFrame({target_column: original_df[target_column]})
        
        # Save to CSV
        selected_df.to_csv(output_file, index=False)
        if logger:
            logger.info(f"Saved top {n_features} features to {output_file}")
        
        return output_file
    
    def _create_visualizations(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        selected_features: List[str],
        output_dir: str,
        n_features: int,
        task_type: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Create visualization plots for the selected features.
        
        Args:
            X: Feature dataframe
            y: Target variable
            selected_features: List of selected feature names
            output_dir: Directory to save output
            n_features: Number of features selected
            task_type: 'classification' or 'regression'
            logger: Logger instance
        """
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # Generate PCA visualization
            self._plot_pca_visualization(X, y, selected_features, plots_dir, n_features, task_type, logger)
            
            # Generate t-SNE visualization for smaller datasets
            if len(X) <= 5000 or n_features <= 100:
                self._plot_tsne_visualization(X, y, selected_features, plots_dir, n_features, task_type, logger)
            else:
                if logger:
                    logger.info(f"Skipping t-SNE for {n_features} features due to dataset size")
        except Exception as e:
            if logger:
                logger.error(f"Error creating visualizations: {str(e)}")
    
    def _plot_feature_importance(
        self, 
        importance_df: pd.DataFrame, 
        output_dir: str, 
        top_n: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Create a bar plot of feature importance for the top N features.
        
        Args:
            importance_df: DataFrame with feature names and importance scores
            output_dir: Directory to save the plot
            top_n: Number of top features to show
            logger: Logger instance
        """
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Use only top N features for readability
            plot_df = importance_df.head(min(top_n, len(importance_df)))
            
            # Create horizontal bar plot
            sns.barplot(x='importance', y='feature', data=plot_df)
            
            # Customize plot
            plt.title(f'Top {len(plot_df)} Feature Importance', fontsize=16)
            plt.xlabel('Importance Score', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.tight_layout()
            
            # Save plot
            importance_plot_file = os.path.join(plots_dir, 'feature_importance_plot.png')
            plt.savefig(importance_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if logger:
                logger.info(f"Saved feature importance plot to {importance_plot_file}")
        except Exception as e:
            if logger:
                logger.error(f"Error creating feature importance plot: {str(e)}")
            plt.close()
    
    def _plot_cumulative_importance(
        self, 
        importance_df: pd.DataFrame, 
        output_dir: str,
        logger: Optional[logging.Logger] = None
    ) -> Dict[float, int]:
        """
        Create a plot of cumulative feature importance.
        
        Args:
            importance_df: DataFrame with feature names and importance scores
            output_dir: Directory to save the plot
            logger: Logger instance
            
        Returns:
            Dictionary mapping importance thresholds to number of features
        """
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate cumulative importance
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            cumulative_importance = importance_df['importance'].cumsum() / importance_df['importance'].sum()
            
            # Plot
            plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-')
            plt.xlabel('Number of Features', fontsize=14)
            plt.ylabel('Cumulative Importance', fontsize=14)
            plt.title('Cumulative Feature Importance', fontsize=16)
            
            # Add horizontal lines at common thresholds
            thresholds = [0.8, 0.9, 0.95, 0.99]
            colors = ['r', 'g', 'orange', 'purple']
            
            threshold_results = {}
            for threshold, color in zip(thresholds, colors):
                n_features = (cumulative_importance >= threshold).argmax() + 1
                threshold_results[threshold] = n_features
                
                plt.axhline(y=threshold, color=color, linestyle='--', 
                          label=f'{threshold*100:.0f}% importance: {n_features} features')
                plt.axvline(x=n_features, color=color, linestyle='--')
            
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            cumulative_plot_file = os.path.join(plots_dir, 'cumulative_importance_plot.png')
            plt.savefig(cumulative_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if logger:
                logger.info(f"Saved cumulative importance plot to {cumulative_plot_file}")
            
            return threshold_results
        except Exception as e:
            if logger:
                logger.error(f"Error creating cumulative importance plot: {str(e)}")
            plt.close()
            return {}
    
    def _plot_feature_correlation(
        self, 
        X: pd.DataFrame, 
        selected_features: List[str], 
        output_dir: str, 
        logger: Optional[logging.Logger] = None,
        max_features: int = 20
    ):
        """
        Create a correlation heatmap of the selected features.
        
        Args:
            X: Feature dataframe
            selected_features: List of feature names to include
            output_dir: Directory to save the plot
            logger: Logger instance
            max_features: Maximum number of features to include in the plot
        """
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # Use up to max_features to keep the plot readable
            if len(selected_features) > max_features:
                plot_features = selected_features[:max_features]
                if logger:
                    logger.info(f"Showing correlation heatmap with top {max_features} features")
            else:
                plot_features = selected_features
            
            # Ensure all selected features exist in X
            plot_features = [f for f in plot_features if f in X.columns]
            
            if not plot_features:
                if logger:
                    logger.warning("No valid features for correlation heatmap")
                return
                
            # Calculate correlation matrix
            corr = X[plot_features].corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                       square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .7})
            
            plt.title(f'Correlation Matrix of Top {len(plot_features)} Features', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            corr_plot_file = os.path.join(plots_dir, 'feature_correlation_heatmap.png')
            plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if logger:
                logger.info(f"Saved correlation heatmap to {corr_plot_file}")
        except Exception as e:
            if logger:
                logger.error(f"Error creating correlation heatmap: {str(e)}")
            plt.close()
    
    def _plot_pca_visualization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        output_dir: str,
        n_features: int,
        task_type: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Create a PCA visualization of the selected features.
        
        Args:
            X: Feature dataframe
            y: Target variable
            selected_features: List of selected feature names
            output_dir: Directory to save the plot
            n_features: Number of features selected
            task_type: 'classification' or 'regression'
            logger: Logger instance
        """
        try:
            # Filter selected features that exist in X
            valid_features = [f for f in selected_features if f in X.columns]
            
            if len(valid_features) < 2:
                if logger:
                    logger.warning(f"Not enough valid features for PCA visualization (need at least 2, got {len(valid_features)})")
                return
            
            # Only use selected features
            X_selected = X[valid_features]
            
            # Apply PCA for dimensionality reduction to 2D
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_selected)
            
            # Create dataframe for plotting
            pca_df = pd.DataFrame({
                'PCA1': X_pca[:, 0],
                'PCA2': X_pca[:, 1],
                'target': y
            })
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            # Handle both categorical and numerical targets
            if task_type == 'classification' or (pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 10):
                # Categorical target - use hue
                ax = sns.scatterplot(x='PCA1', y='PCA2', hue='target', 
                                 data=pca_df, palette='viridis', alpha=0.7, s=60)
                plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Continuous target - use a scatter plot with colormap
                scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['target'], 
                                    cmap='viridis', alpha=0.6, s=50)
                plt.colorbar(scatter, label='Target Value')
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            
            plt.title(f'PCA of Top {n_features} Features\nExplained Variance: {sum(explained_variance):.2%}', 
                    fontsize=14)
            plt.xlabel(f'PCA1 ({explained_variance[0]:.2%})', fontsize=12)
            plt.ylabel(f'PCA2 ({explained_variance[1]:.2%})', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            pca_plot_file = os.path.join(output_dir, f'pca_visualization_top_{n_features}.png')
            plt.savefig(pca_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if logger:
                logger.info(f"Saved PCA visualization to {pca_plot_file}")
        except Exception as e:
            if logger:
                logger.error(f"Error creating PCA visualization: {str(e)}")
            plt.close()
    
    def _plot_tsne_visualization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        output_dir: str,
        n_features: int,
        task_type: str,
        logger: Optional[logging.Logger] = None,
        perplexity: int = 30
    ):
        """
        Create a t-SNE visualization of the selected features.
        
        Args:
            X: Feature dataframe
            y: Target variable
            selected_features: List of selected feature names
            output_dir: Directory to save the plot
            n_features: Number of features selected
            task_type: 'classification' or 'regression'
            logger: Logger instance
            perplexity: Perplexity parameter for t-SNE
        """
        try:
            # Filter selected features that exist in X
            valid_features = [f for f in selected_features if f in X.columns]
            
            if len(valid_features) < 2:
                if logger:
                    logger.warning(f"Not enough valid features for t-SNE visualization (need at least 2, got {len(valid_features)})")
                return
            
            # Only use selected features
            X_selected = X[valid_features]
            
            # Apply t-SNE for dimensionality reduction to 2D
            perplexity = min(perplexity, len(X_selected) - 1)  # Perplexity must be less than n_samples - 1
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
            X_tsne = tsne.fit_transform(X_selected)
            
            # Create dataframe for plotting
            tsne_df = pd.DataFrame({
                'TSNE1': X_tsne[:, 0],
                'TSNE2': X_tsne[:, 1],
                'target': y
            })
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            # Handle both categorical and numerical targets
            if task_type == 'classification' or (pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 10):
                # Categorical target - use hue
                ax = sns.scatterplot(x='TSNE1', y='TSNE2', hue='target', 
                                 data=tsne_df, palette='viridis', alpha=0.7, s=60)
                plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Continuous target - use a scatter plot with colormap
                scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['target'], 
                                    cmap='viridis', alpha=0.6, s=50)
                plt.colorbar(scatter, label='Target Value')
            
            plt.title(f't-SNE Visualization of Top {n_features} Features', fontsize=14)
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            tsne_plot_file = os.path.join(output_dir, f'tsne_visualization_top_{n_features}.png')
            plt.savefig(tsne_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if logger:
                logger.info(f"Saved t-SNE visualization to {tsne_plot_file}")
        except Exception as e:
            if logger:
                logger.error(f"Error creating t-SNE visualization: {str(e)}")
            plt.close()

#nnUNet Training and Inference Tool

class NNUNetTrainingTool(Tool):
    name = "nnunet_training"
    description = """
    This tool trains a segmentation model using the nnUNet framework.
    It first preprocesses the dataset and then trains the model.
    The tool returns the path to the trained model and performance metrics.
    """
    inputs = {
        "dataset_id": {
            "type": "integer",
            "description": "Dataset ID to train with (e.g., 50 for Dataset050, will be zero-padded to 3 digits)"
        },
        "configuration": {
            "type": "string",
            "description": "nnUNet configuration to use ('2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres')",
            "required": False,
            "nullable": True
        },
        "fold": {
            "type": "string",  # Use string type which can handle both integers and "all"
            "description": "Fold of the 5-fold cross-validation. Should be an int between 0 and 4, or 'all' to train all folds.",
            "required": False,
            "nullable": True
        },
        "trainer": {
            "type": "string",
            "description": "Use a custom trainer. Default: nnUNetTrainer",
            "required": False,
            "nullable": True
        },
        "plans_identifier": {
            "type": "string",
            "description": "Custom plans identifier. Default: nnUNetPlans",
            "required": False,
            "nullable": True
        },
        "pretrained_weights": {
            "type": "string",
            "description": "Path to nnU-Net checkpoint file to be used as pretrained model",
            "required": False,
            "nullable": True
        },
        "num_gpus": {
            "type": "integer",
            "description": "Number of GPUs to use for training",
            "required": False,
            "nullable": True
        },
        "device": {
            "type": "string",
            "description": "Device to run training on ('cuda', 'cpu', 'mps')",
            "required": False,
            "nullable": True
        },
        "use_compressed": {
            "type": "boolean",
            "description": "If set, the training cases will not be decompressed",
            "required": False,
            "nullable": True
        },
        "npz": {
            "type": "boolean",
            "description": "Save softmax predictions from final validation as npz files",
            "required": False,
            "nullable": True
        },
        "continue_training": {
            "type": "boolean",
            "description": "Continue training from latest checkpoint",
            "required": False,
            "nullable": True
        },
        "validation_only": {
            "type": "boolean",
            "description": "Only run validation (training must have finished)",
            "required": False,
            "nullable": True
        },
        "val_best": {
            "type": "boolean",
            "description": "Use checkpoint_best instead of checkpoint_final for validation",
            "required": False,
            "nullable": True
        },
        "disable_checkpointing": {
            "type": "boolean",
            "description": "Disable checkpointing during training",
            "required": False,
            "nullable": True
        },
        "verify_dataset_integrity": {
            "type": "boolean",
            "description": "Verify dataset integrity during preprocessing",
            "required": False,
            "nullable": True
        },
        "no_preprocessing": {
            "type": "boolean",
            "description": "Skip preprocessing step (use only if data is already preprocessed)",
            "required": False,
            "nullable": True
        }
    }
    output_type = "object"

    def forward(
        self, 
        dataset_id: int,  # Will be formatted as 3 digits, e.g. 50 → 050
        configuration: Optional[str] = "3d_fullres",
        fold: Optional[str] = None,  # None will be converted to "all", can also be "0", "1", "2", "3", "4", or "all"
        trainer: Optional[str] = None,
        plans_identifier: Optional[str] = None,
        pretrained_weights: Optional[str] = None,
        num_gpus: Optional[int] = None,
        device: Optional[str] = None,
        use_compressed: Optional[bool] = False,
        npz: Optional[bool] = False,
        continue_training: Optional[bool] = False,
        validation_only: Optional[bool] = False,
        val_best: Optional[bool] = False,
        disable_checkpointing: Optional[bool] = False,
        verify_dataset_integrity: Optional[bool] = True,
        no_preprocessing: Optional[bool] = False
    ):
        """
        Train a segmentation model using nnUNet framework with preprocessing.
        
        Args:
            dataset_id: Dataset ID to train with
            configuration: nnUNet configuration
            fold: Cross-validation fold (0-4)
            trainer: Custom trainer
            plans_identifier: Custom plans identifier
            pretrained_weights: Path to pretrained model
            num_gpus: Number of GPUs to use
            device: Device to run on ('cuda', 'cpu', 'mps')
            use_compressed: Use compressed data
            npz: Save softmax predictions
            continue_training: Continue from latest checkpoint
            validation_only: Only run validation
            val_best: Use checkpoint_best for validation
            disable_checkpointing: Disable checkpointing
            verify_dataset_integrity: Verify dataset integrity during preprocessing
            no_preprocessing: Skip preprocessing step
            
        Returns:
            Dictionary with training results including model path and performance metrics
        """
        try:
            # First, preprocess the data unless no_preprocessing is True
            if not no_preprocessing:
                preprocess_result = self._preprocess_data(
                    dataset_id=dataset_id,
                    verify_dataset_integrity=verify_dataset_integrity,
                    configuration=configuration
                )
                
                if preprocess_result.get("status") == "error":
                    return preprocess_result
            
            # Then train the model
            model_path, metrics = self._train_model(
                dataset_id=dataset_id,
                configuration=configuration,
                fold=fold,
                trainer=trainer,
                plans_identifier=plans_identifier,
                pretrained_weights=pretrained_weights,
                num_gpus=num_gpus,
                device=device,
                use_compressed=use_compressed,
                npz=npz,
                continue_training=continue_training,
                validation_only=validation_only,
                val_best=val_best,
                disable_checkpointing=disable_checkpointing
            )
            
            return {
                "status": "success",
                "model_path": model_path,
                "dataset_id": dataset_id,
                "configuration": configuration,
                "fold": fold,
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "dataset_id": dataset_id
            }

    def _preprocess_data(self, dataset_id: int, verify_dataset_integrity: bool = True, configuration: Optional[str] = None):
        """
        Preprocess the data using nnUNetv2_plan_and_preprocess.
        
        Args:
            dataset_id: Dataset ID to preprocess
            verify_dataset_integrity: Whether to verify dataset integrity
            configuration: nnUNet configuration to preprocess for
            
        Returns:
            Dictionary with preprocessing results
        """
        try:
            # Build command for preprocessing
            cmd = ["nnUNetv2_plan_and_preprocess", "-d", f"{dataset_id:03d}"]
            
            # Add configuration-specific option if provided
            if configuration:
                # Use the -c flag to specify which configuration to preprocess
                cmd.extend(["-c", configuration])
            
            if verify_dataset_integrity:
                cmd.append("--verify_dataset_integrity")
            
            print(f"Running preprocessing command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                return {
                    "status": "error",
                    "error_message": f"Preprocessing failed: {process.stderr}",
                    "dataset_id": dataset_id
                }
            
            return {
                "status": "success",
                "message": "Preprocessing completed successfully",
                "dataset_id": dataset_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error during preprocessing: {str(e)}",
                "dataset_id": dataset_id
            }
    
    def _train_model(
        self, 
        dataset_id: int, 
        configuration: str = "3d_fullres",
        fold: Optional[str] = None,
        trainer: Optional[str] = None,
        plans_identifier: Optional[str] = None,
        pretrained_weights: Optional[str] = None,
        num_gpus: Optional[int] = None,
        device: Optional[str] = None,
        use_compressed: bool = False,
        npz: bool = False,
        continue_training: bool = False,
        validation_only: bool = False,
        val_best: bool = False,
        disable_checkpointing: bool = False
    ):
        """
        Train the nnUNet model.
        
        Args:
            dataset_id: Dataset ID to train with
            configuration: nnUNet configuration
            fold: Cross-validation fold (0-4)
            trainer: Custom trainer
            plans_identifier: Custom plans identifier
            pretrained_weights: Path to pretrained model
            num_gpus: Number of GPUs to use
            device: Device to run on ('cuda', 'cpu', 'mps')
            use_compressed: Use compressed data
            npz: Save softmax predictions
            continue_training: Continue from latest checkpoint
            validation_only: Only run validation
            val_best: Use checkpoint_best for validation
            disable_checkpointing: Disable checkpointing
            
        Returns:
            Tuple of (model_path, metrics)
        """
        # Build command for training - start with required positional arguments
        cmd = [
            "nnUNetv2_train",
            f"{dataset_id:03d}",
            configuration,
        ]
        
        # Add fold parameter
        if fold is None:
            # Default to 'all' when fold is not specified
            cmd.append("all")
        else:
            # Use the provided fold value (could be an integer or 'all')
            cmd.append(str(fold))
        
        # Add optional arguments
        if trainer is not None:
            cmd.extend(["-tr", trainer])
            
        if plans_identifier is not None:
            cmd.extend(["-p", plans_identifier])
            
        if pretrained_weights is not None:
            cmd.extend(["-pretrained_weights", pretrained_weights])
            
        if num_gpus is not None:
            cmd.extend(["-num_gpus", str(num_gpus)])
            
        if device is not None:
            cmd.extend(["-device", device])
            
        # Add boolean flags
        if use_compressed:
            cmd.append("--use_compressed")
            
        if npz:
            cmd.append("--npz")
            
        if continue_training:
            cmd.append("--c")
            
        if validation_only:
            cmd.append("--val")
            
        if val_best:
            cmd.append("--val_best")
            
        if disable_checkpointing:
            cmd.append("--disable_checkpointing")
        
        # Run the training
        print(f"Running training command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise RuntimeError(f"Training failed: {process.stderr}")
        
        # Get model path
        results_folder = os.environ["RESULTS_FOLDER"]
        
        # Determine the fold directory name
        if fold is None or fold == "all":
            fold_dir = "fold_all"
        else:
            fold_dir = f"fold_{fold}"
            
        model_path = os.path.join(
            results_folder, 
            f"nnUNetv2_{configuration}", 
            f"Dataset{dataset_id:03d}",
            fold_dir
        )
        
        # Parse metrics from validation output
        metrics = self._parse_metrics(process.stdout)
        
        return model_path, metrics
    
    def _parse_metrics(self, output_text: str):
        """
        Parse metrics from nnUNet training output.
        
        Args:
            output_text: Output text from the training process
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "dice": None,
            "iou": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "validation_loss": None
        }
        
        # Extract metrics from output_text
        import re
        
        # Extract Dice score
        dice_match = re.search(r"mean\s+dice\s*:\s*([0-9.]+)", output_text, re.IGNORECASE)
        if dice_match:
            metrics["dice"] = float(dice_match.group(1))
            
        # Extract IoU/Jaccard if available
        iou_match = re.search(r"mean\s+iou\s*:\s*([0-9.]+)", output_text, re.IGNORECASE)
        if iou_match:
            metrics["iou"] = float(iou_match.group(1))
            
        # Extract validation loss if available
        loss_match = re.search(r"validation\s+loss\s*:\s*([0-9.]+)", output_text, re.IGNORECASE)
        if loss_match:
            metrics["validation_loss"] = float(loss_match.group(1))
        
        return metrics


class NNUNetInferenceTool(Tool):
    name = "nnunet_inference"
    description = """
    This tool runs inference with a trained nnUNet model.
    It applies the model to input images to generate segmentation masks.
    The tool returns the path to the output segmentations.
    """
    inputs = {
        "input_folder": {
            "type": "string",
            "description": "Input folder containing images to segment. Files should use correct channel numberings (_0000 etc)"
        },
        "output_folder": {
            "type": "string",
            "description": "Output folder where segmentations will be saved"
        },
        "dataset_id": {
            "type": "integer",
            "description": "Dataset ID used for training the model (e.g., 50 for Dataset050, will be zero-padded to 3 digits)"
        },
        "configuration": {
            "type": "string",
            "description": "nnUNet configuration to use ('2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres')"
        },
        "model_folder": {
            "type": "string",
            "description": "Path to the directory containing the trained model. For example, the path to 'nnUNet_results/Dataset135_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres' or a subdirectory containing the fold",
            "required": False,
            "nullable": True
        },
        "results_dir": {
            "type": "string",
            "description": "Base directory containing nnUNet results (overrides nnUNet_results environment variable)",
            "required": False,
            "nullable": True
        },
        "folds": {
            "type": "string",
            "description": "Comma-separated list of folds to use for prediction (e.g., '0,1,2,3,4' or 'all')",
            "required": False,
            "nullable": True
        },
        "plans_identifier": {
            "type": "string",
            "description": "Plans identifier. Default: nnUNetPlans",
            "required": False,
            "nullable": True
        },
        "trainer": {
            "type": "string",
            "description": "Trainer class used for training. Default: nnUNetTrainer",
            "required": False,
            "nullable": True
        },
        "step_size": {
            "type": "number",
            "description": "Step size for sliding window prediction (0-1). Default: 0.5",
            "required": False,
            "nullable": True
        },
        "disable_tta": {
            "type": "boolean",
            "description": "Disable test time augmentation (mirroring). Faster but less accurate",
            "required": False,
            "nullable": True
        },
        "save_probabilities": {
            "type": "boolean",
            "description": "Export predicted class probabilities (needed for ensembling)",
            "required": False,
            "nullable": True
        },
        "continue_prediction": {
            "type": "boolean",
            "description": "Continue an aborted previous prediction",
            "required": False,
            "nullable": True
        },
        "checkpoint": {
            "type": "string",
            "description": "Checkpoint name to use. Default: checkpoint_final.pth",
            "required": False,
            "nullable": True
        },
        "num_processes_preprocessing": {
            "type": "integer",
            "description": "Number of processes for preprocessing",
            "required": False,
            "nullable": True
        },
        "num_processes_segmentation": {
            "type": "integer",
            "description": "Number of processes for segmentation export",
            "required": False,
            "nullable": True
        },
        "prev_stage_predictions": {
            "type": "string",
            "description": "Folder with predictions from previous stage (for cascade models)",
            "required": False,
            "nullable": True
        },
        "num_parts": {
            "type": "integer",
            "description": "Number of separate inference calls for parallelization",
            "required": False,
            "nullable": True
        },
        "part_id": {
            "type": "integer",
            "description": "Which part of the parallel inference is this (0 to num_parts-1)",
            "required": False,
            "nullable": True
        },
        "device": {
            "type": "string",
            "description": "Device for inference: 'cuda' (GPU), 'cpu', or 'mps' (Apple)",
            "required": False,
            "nullable": True
        },
        "verbose": {
            "type": "boolean",
            "description": "Enable verbose output",
            "required": False,
            "nullable": True
        }
    }
    output_type = "object"

    def forward(
        self, 
        input_folder: str,
        output_folder: str,
        dataset_id: int,  # Will be formatted as 3 digits, e.g. 50 → 050
        configuration: str,
        model_folder: Optional[str] = None,
        results_dir: Optional[str] = None,
        folds: Optional[str] = None,  # "0,1,2,3,4" or "all"
        plans_identifier: Optional[str] = None,
        trainer: Optional[str] = None,
        step_size: Optional[float] = None,
        disable_tta: Optional[bool] = False,
        save_probabilities: Optional[bool] = False,
        continue_prediction: Optional[bool] = False,
        checkpoint: Optional[str] = None,
        num_processes_preprocessing: Optional[int] = None,
        num_processes_segmentation: Optional[int] = None,
        prev_stage_predictions: Optional[str] = None,
        num_parts: Optional[int] = None,
        part_id: Optional[int] = None,
        device: Optional[str] = None,
        verbose: Optional[bool] = False
    ):
        """
        Run inference with a trained nnUNet model.
        
        Args:
            input_folder: Folder containing images to segment
            output_folder: Folder to save segmentation results
            dataset_id: Dataset ID used for training
            configuration: nnUNet configuration
            folds: Comma-separated list of folds to use
            plans_identifier: Plans identifier
            trainer: Trainer class used
            step_size: Step size for sliding window
            disable_tta: Disable test time augmentation
            save_probabilities: Save softmax outputs
            continue_prediction: Continue previous prediction
            checkpoint: Checkpoint name to use
            num_processes_preprocessing: Processes for preprocessing
            num_processes_segmentation: Processes for segmentation export
            prev_stage_predictions: Previous stage predictions folder
            num_parts: Total number of parallel parts
            part_id: ID of this parallel part
            device: Device for inference
            verbose: Enable verbose output
            
        Returns:
            Dictionary with inference results and paths
        """
        try:
            # Run the inference
            output_files = self._run_inference(
                input_folder=input_folder,
                output_folder=output_folder,
                dataset_id=dataset_id,
                configuration=configuration,
                model_folder=model_folder,
                results_dir=results_dir,
                folds=folds,
                plans_identifier=plans_identifier,
                trainer=trainer,
                step_size=step_size,
                disable_tta=disable_tta,
                save_probabilities=save_probabilities,
                continue_prediction=continue_prediction,
                checkpoint=checkpoint,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation=num_processes_segmentation,
                prev_stage_predictions=prev_stage_predictions,
                num_parts=num_parts,
                part_id=part_id,
                device=device,
                verbose=verbose
            )
            
            return {
                "status": "success",
                "output_folder": output_folder,
                "dataset_id": dataset_id,
                "configuration": configuration,
                "num_segmentations": len(output_files),
                "segmentation_files": output_files
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "dataset_id": dataset_id,
                "input_folder": input_folder
            }
    
    def _run_inference(
        self,
        input_folder: str,
        output_folder: str,
        dataset_id: int,
        configuration: str,
        model_folder: Optional[str] = None,
        results_dir: Optional[str] = None,
        folds: Optional[str] = None,
        plans_identifier: Optional[str] = None,
        trainer: Optional[str] = None,
        step_size: Optional[float] = None,
        disable_tta: Optional[bool] = False,
        save_probabilities: Optional[bool] = False,
        continue_prediction: Optional[bool] = False,
        checkpoint: Optional[str] = None,
        num_processes_preprocessing: Optional[int] = None,
        num_processes_segmentation: Optional[int] = None,
        prev_stage_predictions: Optional[str] = None,
        num_parts: Optional[int] = None,
        part_id: Optional[int] = None,
        device: Optional[str] = None,
        verbose: Optional[bool] = False
    ) -> List[str]:
        """
        Run nnUNet inference.
        
        Args:
            Same as forward method
            
        Returns:
            List of output segmentation file paths
        """
        # Build command for inference - required arguments
        cmd = [
            "nnUNetv2_predict",
            "-i", input_folder,
            "-o", output_folder,
            "-d", f"{dataset_id:03d}",
            "-c", configuration
        ]
        
        # Set RESULTS_FOLDER environment variable if provided
        if results_dir:
            print(f"Setting RESULTS_FOLDER to: {results_dir}")
            os.environ["RESULTS_FOLDER"] = results_dir
            
        # Handle model folder path
        if model_folder is not None:
            # nnUNetv2_predict looks for models in a specific structure:
            # RESULTS_FOLDER/Dataset{dataset_id}_{dataset_name}/{trainer}__{plans}__{configuration}/fold_{fold}/checkpoint_final.pth
            
            # We need to extract the correct RESULTS_FOLDER from the model_folder path
            model_path = os.path.abspath(model_folder)
            
            # Navigate up the folder structure to find the proper RESULTS_FOLDER
            # Typical path pattern: .../nnUNet_results/Dataset{dataset_id}_{dataset_name}/{trainer}__{plans}__{configuration}/fold_{fold}
            # We need to go up to the directory containing "Dataset{dataset_id}..."
            
            folder_parts = model_path.split(os.sep)
            results_folder = None
            
            # Look for a directory pattern like "Dataset{digits}_*"
            for i in range(len(folder_parts)-1, 0, -1):
                if folder_parts[i].startswith("Dataset") and "_" in folder_parts[i]:
                    # Found the dataset folder, set results_folder to its parent
                    results_folder = os.sep.join(folder_parts[:i])
                    break
            
            if results_folder:
                print(f"Setting nnUNet_results to: {results_folder}")
                os.environ["RESULTS_FOLDER"] = results_folder
            else:
                # If we couldn't find the pattern, just set RESULTS_FOLDER to parent of model_folder
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
                print(f"Could not detect standard nnUNet folder structure. Setting RESULTS_FOLDER to: {parent_dir}")
                os.environ["RESULTS_FOLDER"] = parent_dir
        
        # Add optional arguments
        if folds is not None:
            # Handle comma-separated list of folds
            if folds.lower() != 'all':
                fold_list = folds.split(',')
                cmd.extend(["-f"] + fold_list)
            else:
                # If 'all' is specified, use the 'all' keyword
                cmd.extend(["-f", "all"])
        
        if plans_identifier is not None:
            cmd.extend(["-p", plans_identifier])
            
        if trainer is not None:
            cmd.extend(["-tr", trainer])
        
        if step_size is not None:
            cmd.extend(["-step_size", str(step_size)])
            
        if disable_tta:
            cmd.append("--disable_tta")
            
        if verbose:
            cmd.append("--verbose")
            
        if save_probabilities:
            cmd.append("--save_probabilities")
            
        if continue_prediction:
            cmd.append("--continue_prediction")
            
        if checkpoint is not None:
            cmd.extend(["-chk", checkpoint])
            
        if num_processes_preprocessing is not None:
            cmd.extend(["-npp", str(num_processes_preprocessing)])
            
        if num_processes_segmentation is not None:
            cmd.extend(["-nps", str(num_processes_segmentation)])
            
        if prev_stage_predictions is not None:
            cmd.extend(["-prev_stage_predictions", prev_stage_predictions])
            
        if num_parts is not None:
            cmd.extend(["-num_parts", str(num_parts)])
            
        if part_id is not None:
            cmd.extend(["-part_id", str(part_id)])
            
        if device is not None:
            cmd.extend(["-device", device])
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Run the inference command
        print(f"Running inference command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise RuntimeError(f"Inference failed: {process.stderr}")
        
        # Get output segmentation files
        output_files = self._get_output_files(output_folder)
        
        return output_files
    
    def _get_output_files(self, output_folder: str) -> List[str]:
        """
        Get list of segmentation files in the output folder.
        
        Args:
            output_folder: Path to the output folder
            
        Returns:
            List of segmentation file paths
        """
        files = []
        
        # Get all files in the output folder
        for root, _, filenames in os.walk(output_folder):
            for filename in filenames:
                if filename.endswith(('.nii.gz', '.nii')):
                    files.append(os.path.join(root, filename))
        
        return files

#TotalSegmentator Tool

class TotalSegmentatorTool(Tool):
    name = "totalsegmentator"
    description = """
    This tool uses TotalSegmentator to segment anatomical structures in CT and MR images.
    It can process NIFTI files or DICOM slices and supports various tasks and options.
    Tasks ending with '_mr' are designed for MR images, while other tasks are for CT images.
    """
    
    # All available task options
    AVAILABLE_TASKS = [
        # Main tasks
        "total", "total_mr",
        # Subtasks
        "lung_vessels", "body", "body_mr", "vertebrae_mr", "cerebral_bleed", 
        "hip_implant", "pleural_pericard_effusion", "head_glands_cavities",
        "head_muscles", "headneck_bones_vessels", "headneck_muscles", 
        "liver_vessels", "oculomotor_muscles", "lung_nodules", "kidney_cysts",
        "breasts", "liver_segments", "liver_segments_mr", "heartchambers_highres",
        "appendicular_bones", "appendicular_bones_mr", "tissue_types", 
        "tissue_types_mr", "tissue_4_types", "brain_structures", "vertebrae_body",
        "face", "face_mr", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr",
        "coronary_arteries"
    ]
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input CT nifti image or folder of dicom slices"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory for segmentation masks"
        },
        "output_type": {
            "type": "string",
            "description": "Select if segmentations shall be saved as Nifti or as Dicom RT Struct image",
            "required": False,
            "nullable": True
        },
        "multilabel": {
            "type": "boolean",
            "description": "Save one multilabel image for all classes instead of separate binary masks",
            "required": False,
            "nullable": True
        },
        "nr_threads_resampling": {
            "type": "integer",
            "description": "Number of threads for resampling",
            "required": False,
            "nullable": True
        },
        "nr_threads_saving": {
            "type": "integer",
            "description": "Number of threads for saving segmentations",
            "required": False,
            "nullable": True
        },
        "fast": {
            "type": "boolean",
            "description": "Run faster lower resolution model (3mm)",
            "required": False,
            "nullable": True
        },
        "fastest": {
            "type": "boolean",
            "description": "Run even faster lower resolution model (6mm)",
            "required": False,
            "nullable": True
        },
        "nora_tag": {
            "type": "string",
            "description": "Tag in nora as mask. Pass nora project id as argument",
            "required": False,
            "nullable": True
        },
        "preview": {
            "type": "boolean",
            "description": "Generate a png preview of segmentation",
            "required": False,
            "nullable": True
        },
        "task": {
            "type": "string",
            "description": "Select which model to use. Tasks ending with '_mr' are for MR images. Default is 'total' for CT images.",
            "required": False,
            "nullable": True
        },
        "roi_subset": {
            "type": "string",
            "description": "Define a subset of classes to save (comma separated list of class names). If running 1.5mm model, will only run the appropriate models for these rois",
            "required": False,
            "nullable": True
        },
        "roi_subset_robust": {
            "type": "string",
            "description": "Like roi_subset but uses a slower but more robust model to find the rois",
            "required": False,
            "nullable": True
        },
        "statistics": {
            "type": "boolean",
            "description": "Calculate volume (in mm3) and mean intensity. Results will be in statistics.json",
            "required": False,
            "nullable": True
        },
        "radiomics": {
            "type": "boolean",
            "description": "Calculate radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
            "required": False,
            "nullable": True
        },
        "stats_include_incomplete": {
            "type": "boolean",
            "description": "Normally statistics are only calculated for ROIs which are not cut off by the beginning or end of image. Use this option to calc anyways",
            "required": False,
            "nullable": True
        },
        "crop_path": {
            "type": "string",
            "description": "Custom path to masks used for cropping. If not set will use output directory",
            "required": False,
            "nullable": True
        },
        "body_seg": {
            "type": "boolean",
            "description": "Do initial rough body segmentation and crop image to body region",
            "required": False,
            "nullable": True
        },
        "force_split": {
            "type": "boolean",
            "description": "Process image in 3 chunks for less memory consumption",
            "required": False,
            "nullable": True
        },
        "skip_saving": {
            "type": "boolean",
            "description": "Skip saving of segmentations for faster runtime if you are only interested in statistics",
            "required": False,
            "nullable": True
        },
        "no_derived_masks": {
            "type": "boolean",
            "description": "Do not create derived masks (e.g. skin from body mask)",
            "required": False,
            "nullable": True
        },
        "device": {
            "type": "string",
            "description": "Device to use for inference ('gpu', 'cpu', or 'mps')",
            "required": False,
            "nullable": True
        },
        "quiet": {
            "type": "boolean",
            "description": "Suppress console output",
            "required": False,
            "nullable": True
        },
        "verbose": {
            "type": "boolean",
            "description": "Verbose output",
            "required": False,
            "nullable": True
        },
        "license_number": {
            "type": "string",
            "description": "License number for TotalSegmentator",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        output_dir: str,
        output_type: Optional[str] = None,
        multilabel: Optional[bool] = False,
        nr_threads_resampling: Optional[int] = None,
        nr_threads_saving: Optional[int] = None,
        fast: Optional[bool] = False,
        fastest: Optional[bool] = False,
        nora_tag: Optional[str] = None,
        preview: Optional[bool] = False,
        task: Optional[str] = None,
        roi_subset: Optional[str] = None,
        roi_subset_robust: Optional[str] = None,
        statistics: Optional[bool] = False,
        radiomics: Optional[bool] = False,
        stats_include_incomplete: Optional[bool] = False,
        crop_path: Optional[str] = None,
        body_seg: Optional[bool] = False,
        force_split: Optional[bool] = False,
        skip_saving: Optional[bool] = False,
        no_derived_masks: Optional[bool] = False,
        device: Optional[str] = None,
        quiet: Optional[bool] = False,
        verbose: Optional[bool] = False,
        license_number: Optional[str] = None
    ):
        """
        Run TotalSegmentator to segment anatomical structures in CT or MR images.
        
        Args:
            input_path: Path to input CT/MR nifti image or folder of dicom slices
            output_dir: Output directory for segmentation masks
            output_type: Save segmentations as 'nifti' or 'dicom'
            multilabel: Save one multilabel image for all classes
            nr_threads_resampling: Number of threads for resampling
            nr_threads_saving: Number of threads for saving segmentations
            fast: Run faster lower resolution model (3mm)
            fastest: Run even faster lower resolution model (6mm)
            nora_tag: Tag in nora as mask
            preview: Generate a png preview of segmentation
            task: Select which model to use. Tasks ending with '_mr' are for MR images
            roi_subset: Define a subset of classes to save (comma separated)
            roi_subset_robust: Like roi_subset but uses a more robust model
            statistics: Calculate volume and mean intensity
            radiomics: Calculate radiomics features
            stats_include_incomplete: Calculate statistics for incomplete ROIs
            crop_path: Custom path to masks used for cropping
            body_seg: Do initial rough body segmentation
            force_split: Process image in 3 chunks for less memory consumption
            skip_saving: Skip saving of segmentations
            no_derived_masks: Do not create derived masks
            device: Device to use for inference ('gpu', 'cpu', or 'mps')
            quiet: Suppress console output
            verbose: Verbose output
            license_number: License number for TotalSegmentator
            
        Returns:
            Dictionary with segmentation results and stats
        """
        try:
            # Run TotalSegmentator
            result = self._run_totalsegmentator(
                input_path=input_path,
                output_dir=output_dir,
                output_type=output_type,
                multilabel=multilabel,
                nr_threads_resampling=nr_threads_resampling,
                nr_threads_saving=nr_threads_saving,
                fast=fast,
                fastest=fastest,
                nora_tag=nora_tag,
                preview=preview,
                task=task,
                roi_subset=roi_subset,
                roi_subset_robust=roi_subset_robust,
                statistics=statistics,
                radiomics=radiomics,
                stats_include_incomplete=stats_include_incomplete,
                crop_path=crop_path,
                body_seg=body_seg,
                force_split=force_split,
                skip_saving=skip_saving,
                no_derived_masks=no_derived_masks,
                device=device,
                quiet=quiet,
                verbose=verbose,
                license_number=license_number
            )
            
            # Collect segmentation files
            segmentation_files = []
            if not skip_saving:
                segmentation_files = self._get_segmentation_files(output_dir)
            
            # Collect statistics if requested
            stats = None
            if statistics:
                stats_file = os.path.join(output_dir, "statistics.json")
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
            
            # Collect radiomics features if requested
            radiomics_stats = None
            if radiomics:
                radiomics_file = os.path.join(output_dir, "statistics_radiomics.json")
                if os.path.exists(radiomics_file):
                    with open(radiomics_file, 'r') as f:
                        radiomics_stats = json.load(f)
            
            # Add task information to results
            task_info = {
                "task": task if task else "total",
                "is_mr_task": bool(task and task.endswith("_mr")) if task else False,
                "image_type": "MR" if (task and task.endswith("_mr")) else "CT"
            }
            
            return {
                "status": "success",
                "output_dir": output_dir,
                "command_output": result,
                "segmentation_files": segmentation_files,
                "statistics": stats,
                "radiomics": radiomics_stats,
                "preview_file": os.path.join(output_dir, "preview.png") if preview and os.path.exists(os.path.join(output_dir, "preview.png")) else None,
                "task_info": task_info
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "input_path": input_path,
                "output_dir": output_dir
            }
    
    def _run_totalsegmentator(
        self,
        input_path: str,
        output_dir: str,
        output_type: Optional[str] = None,
        multilabel: Optional[bool] = False,
        nr_threads_resampling: Optional[int] = None,
        nr_threads_saving: Optional[int] = None,
        fast: Optional[bool] = False,
        fastest: Optional[bool] = False,
        nora_tag: Optional[str] = None,
        preview: Optional[bool] = False,
        task: Optional[str] = None,
        roi_subset: Optional[str] = None,
        roi_subset_robust: Optional[str] = None,
        statistics: Optional[bool] = False,
        radiomics: Optional[bool] = False,
        stats_include_incomplete: Optional[bool] = False,
        crop_path: Optional[str] = None,
        body_seg: Optional[bool] = False,
        force_split: Optional[bool] = False,
        skip_saving: Optional[bool] = False,
        no_derived_masks: Optional[bool] = False,
        device: Optional[str] = None,
        quiet: Optional[bool] = False,
        verbose: Optional[bool] = False,
        license_number: Optional[str] = None
    ) -> str:
        """
        Execute TotalSegmentator command.
        
        Args:
            Same as forward method
            
        Returns:
            Command output as string
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command for TotalSegmentator
        cmd = ["TotalSegmentator", "-i", input_path, "-o", output_dir]
        
        # Add optional arguments
        if output_type:
            if output_type.lower() in ["nifti", "dicom"]:
                cmd.extend(["-ot", output_type.lower()])
            else:
                raise ValueError("output_type must be 'nifti' or 'dicom'")
        
        if multilabel:
            cmd.append("-ml")
            
        if nr_threads_resampling is not None:
            cmd.extend(["-nr", str(nr_threads_resampling)])
            
        if nr_threads_saving is not None:
            cmd.extend(["-ns", str(nr_threads_saving)])
        
        if fast:
            cmd.append("-f")
            
        if fastest:
            cmd.append("-ff")
            
        if nora_tag:
            cmd.extend(["-t", nora_tag])
            
        if preview:
            cmd.append("-p")
            
        if task:
            if task in self.AVAILABLE_TASKS:
                cmd.extend(["-ta", task])
            else:
                valid_tasks = ", ".join(self.AVAILABLE_TASKS)
                raise ValueError(f"task must be one of: {valid_tasks}")
                
        if roi_subset:
            # Convert comma-separated string to space-separated list for command line
            roi_list = roi_subset.split(",")
            roi_list = [roi.strip() for roi in roi_list]
            cmd.extend(["-rs"] + roi_list)
            
        if roi_subset_robust:
            # Convert comma-separated string to space-separated list for command line
            roi_robust_list = roi_subset_robust.split(",")
            roi_robust_list = [roi.strip() for roi in roi_robust_list]
            cmd.extend(["-rsr"] + roi_robust_list)
            
        if statistics:
            cmd.append("-s")
            
        if radiomics:
            cmd.append("-r")
            
        if stats_include_incomplete:
            cmd.append("-sii")
            
        if crop_path:
            cmd.extend(["-cp", crop_path])
            
        if body_seg:
            cmd.append("-bs")
            
        if force_split:
            cmd.append("-fs")
            
        if skip_saving:
            cmd.append("-ss")
            
        if no_derived_masks:
            cmd.append("-ndm")
            
        if device:
            if device.lower() in ["gpu", "cpu", "mps"]:
                cmd.extend(["-d", device.lower()])
            else:
                raise ValueError("device must be 'gpu', 'cpu', or 'mps'")
                
        if quiet:
            cmd.append("-q")
            
        if verbose:
            cmd.append("-v")
            
        if license_number:
            cmd.extend(["-l", license_number])
        
        # Run TotalSegmentator
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise RuntimeError(f"TotalSegmentator failed: {process.stderr}")
        
        return process.stdout
    
    def _get_segmentation_files(self, output_dir: str) -> List[str]:
        """
        Get list of segmentation files in the output directory.
        
        Args:
            output_dir: Path to the output directory
            
        Returns:
            List of segmentation file paths
        """
        files = []
        
        # Get all files in the output directory
        for root, _, filenames in os.walk(output_dir):
            for filename in filenames:
                # Skip statistics and preview files
                if filename in ["statistics.json", "statistics_radiomics.json", "preview.png"]:
                    continue
                
                # Include nifti and dicom files
                if filename.endswith(('.nii.gz', '.nii', '.dcm')):
                    files.append(os.path.join(root, filename))
        
        return files

#Classification Model Training and Inference Tool (for tabulated data)

class PyCaretClassificationTool(Tool):
    name = "pycaret_classification"
    description = """
    This tool uses PyCaret to train and evaluate classification models.
    It compares multiple models, tunes the best ones, creates a blended model,
    and generates various visualizations and interpretations.
    Results are saved to the specified output directory.
    """
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input data file (CSV format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory where model and results will be saved"
        },
        "target_column": {
            "type": "string",
            "description": "Name of the target column for classification"
        },
        "experiment_name": {
            "type": "string",
            "description": "Name of the experiment",
            "required": False,
            "nullable": True
        },
        "fold": {
            "type": "integer",
            "description": "Number of cross-validation folds",
            "required": False,
            "nullable": True
        },
        "session_id": {
            "type": "integer",
            "description": "Random seed for reproducibility",
            "required": False,
            "nullable": True
        },
        "use_gpu": {
            "type": "boolean",
            "description": "Whether to use GPU for training (if available)",
            "required": False,
            "nullable": True
        },
        "fix_imbalance": {
            "type": "boolean",
            "description": "Whether to fix class imbalance",
            "required": False,
            "nullable": True
        },
        "data_split_stratify": {
            "type": "boolean",
            "description": "Whether to use stratified sampling for data splitting",
            "required": False,
            "nullable": True
        },
        "data_split_shuffle": {
            "type": "boolean",
            "description": "Whether to shuffle data before splitting",
            "required": False,
            "nullable": True
        },
        "preprocess": {
            "type": "boolean",
            "description": "Whether to apply preprocessing steps",
            "required": False,
            "nullable": True
        },
        "ignore_features": {
            "type": "string",
            "description": "Comma-separated list of features to ignore during training",
            "required": False,
            "nullable": True
        },
        "numeric_features": {
            "type": "string",
            "description": "Comma-separated list of numeric features",
            "required": False,
            "nullable": True
        },
        "categorical_features": {
            "type": "string",
            "description": "Comma-separated list of categorical features",
            "required": False,
            "nullable": True
        },
        "date_features": {
            "type": "string",
            "description": "Comma-separated list of date features",
            "required": False,
            "nullable": True
        },
        "n_select": {
            "type": "integer",
            "description": "Number of top models to select for blending",
            "required": False,
            "nullable": True
        },
        "normalize": {
            "type": "boolean",
            "description": "Whether to normalize numeric features",
            "required": False,
            "nullable": True
        },
        "transformation": {
            "type": "boolean",
            "description": "Whether to apply transformation to numeric features",
            "required": False,
            "nullable": True
        },
        "pca": {
            "type": "boolean",
            "description": "Whether to apply PCA for dimensionality reduction",
            "required": False,
            "nullable": True
        },
        "pca_components": {
            "type": "number",
            "description": "Number of PCA components (float between 0-1 or int > 1)",
            "required": False,
            "nullable": True
        },
        "include_models": {
            "type": "string",
            "description": "Comma-separated list of models to include in comparison",
            "required": False,
            "nullable": True
        },
        "exclude_models": {
            "type": "string",
            "description": "Comma-separated list of models to exclude from comparison",
            "required": False,
            "nullable": True
        },
        "test_data_path": {
            "type": "string",
            "description": "Path to test/holdout data for independent evaluation",
            "required": False,
            "nullable": True
        },
        "ignore_gpu_errors": {
            "type": "boolean",
            "description": "Whether to ignore GPU-related errors and fall back to CPU",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        output_dir: str,
        target_column: str,
        experiment_name: Optional[str] = None,
        fold: Optional[int] = 10,
        session_id: Optional[int] = None,
        use_gpu: Optional[bool] = False,
        fix_imbalance: Optional[bool] = True,
        data_split_stratify: Optional[bool] = True,
        data_split_shuffle: Optional[bool] = True,
        preprocess: Optional[bool] = True,
        ignore_features: Optional[str] = None,
        numeric_features: Optional[str] = None,
        categorical_features: Optional[str] = None,
        date_features: Optional[str] = None,
        n_select: Optional[int] = 3,
        normalize: Optional[bool] = False,
        transformation: Optional[bool] = False,
        pca: Optional[bool] = False,
        pca_components: Optional[float] = None,
        include_models: Optional[str] = None,
        exclude_models: Optional[str] = None,
        test_data_path: Optional[str] = None,
        ignore_gpu_errors: Optional[bool] = True
    ):
        """
        Train and evaluate classification models using PyCaret.
        
        Args:
            input_path: Path to input CSV file with training data
            output_dir: Directory to save outputs
            target_column: Target column name for classification
            experiment_name: Name of the experiment
            fold: Number of cross-validation folds
            session_id: Random seed for reproducibility
            use_gpu: Whether to use GPU for training
            fix_imbalance: Whether to fix class imbalance
            data_split_stratify: Whether to use stratified sampling
            data_split_shuffle: Whether to shuffle data before splitting
            preprocess: Whether to apply preprocessing steps
            ignore_features: Comma-separated list of features to ignore
            numeric_features: Comma-separated list of numeric features
            categorical_features: Comma-separated list of categorical features
            date_features: Comma-separated list of date features
            n_select: Number of top models to select for blending
            normalize: Whether to normalize numeric features
            transformation: Whether to apply transformation to numeric features
            pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components
            include_models: Comma-separated list of models to include
            exclude_models: Comma-separated list of models to exclude
            test_data_path: Path to test/holdout data for independent evaluation
            ignore_gpu_errors: Whether to ignore GPU errors and fallback to CPU
            
        Returns:
            Dictionary with model training results and file paths
        """
        try:
            # Setup logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "pycaret_classification.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Generate experiment name if not provided
            if experiment_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                experiment_name = f"classification_exp_{timestamp}"
            
            # Generate session_id if not provided
            if session_id is None:
                import random
                session_id = random.randint(1, 10000)
            
            logging.info(f"Starting classification experiment: {experiment_name}")
            logging.info(f"Input data: {input_path}")
            logging.info(f"Output directory: {output_dir}")
            logging.info(f"Target column: {target_column}")
            
            # Import PyCaret's classification module
            try:
                from pycaret.classification import (
                    setup, compare_models, tune_model, blend_models, 
                    pull, predict_model, save_model, load_model,
                    plot_model, interpret_model
                )
                
                # Check PyCaret version for compatibility adjustments
                import pycaret
                pycaret_version = getattr(pycaret, "__version__", "unknown")
                logging.info(f"PyCaret version: {pycaret_version}")
                
                logging.info("PyCaret imported successfully")
            except ImportError as e:
                logging.error(f"Error importing PyCaret: {str(e)}")
                logging.error("Please install PyCaret: pip install pycaret")
                raise ImportError("PyCaret is required for this tool. Please install it with: pip install pycaret")
            
            # Load data
            logging.info(f"Loading data from {input_path}")
            data = pd.read_csv(input_path)
            logging.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Verify target column exists
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset")
            
            # Prepare parameters for setup
            setup_params = {
                'data': data,
                'target': target_column,
                'session_id': session_id,
                'experiment_name': experiment_name,
                'fold': fold,
                'use_gpu': use_gpu,
                'fix_imbalance': fix_imbalance,
                'preprocess': preprocess,
                'data_split_stratify': data_split_stratify,
                'data_split_shuffle': data_split_shuffle,
                'log_experiment': False  # Set to False to avoid MLflow errors
            }
            
            logging.info("Disabling experiment logging to avoid MLflow-related errors")
                
            # Add optional parameters if provided
            if ignore_features:
                setup_params['ignore_features'] = [f.strip() for f in ignore_features.split(',')]
            
            if numeric_features:
                setup_params['numeric_features'] = [f.strip() for f in numeric_features.split(',')]
            
            if categorical_features:
                setup_params['categorical_features'] = [f.strip() for f in categorical_features.split(',')]
            
            if date_features:
                setup_params['date_features'] = [f.strip() for f in date_features.split(',')]
            
            if normalize is not None:
                setup_params['normalize'] = normalize
                
            if transformation is not None:
                setup_params['transformation'] = transformation
                
            if pca is not None:
                setup_params['pca'] = pca
                
            if pca_components is not None:
                setup_params['pca_components'] = pca_components
            
            # Check GPU availability and cuml installation if use_gpu is requested
            if use_gpu:
                try:
                    import cuml
                    logging.info("RAPIDS cuML is available for GPU acceleration")
                except ImportError:
                    logging.warning("'cuml' is not installed but use_gpu=True was specified.")
                    if ignore_gpu_errors:
                        logging.warning("Running on CPU instead. To use GPU, install cuml: pip install cuml")
                        use_gpu = False
                        setup_params['use_gpu'] = False
                    else:
                        raise ImportError("GPU acceleration requested but cuml is not installed. Install with: pip install cuml")
            
            # Set up the experiment with error handling for parameter compatibility
            logging.info("Setting up PyCaret experiment")
            logging.info(f"Setup parameters: {setup_params}")
            
            # Try to create setup with default parameters first
            try:
                s = setup(**setup_params)
                logging.info("PyCaret setup completed successfully")
            except TypeError as e:
                logging.warning(f"Setup error: {str(e)}")
                if "unexpected keyword argument" in str(e):
                    # Handle incompatible parameters by removing them and retrying
                    error_param = str(e).split("argument ")[-1].split("'")[1] if "'" in str(e) else str(e).split("argument ")[-1].strip()
                    logging.warning(f"Incompatible parameter detected: {error_param}")
                    logging.warning(f"Removing parameter and retrying setup")
                    
                    if error_param in setup_params:
                        del setup_params[error_param]
                        try:
                            s = setup(**setup_params)
                            logging.info("PyCaret setup completed successfully after parameter adjustment")
                        except Exception as inner_e:
                            logging.error(f"Setup failed even after removing parameter: {str(inner_e)}")
                            # Try a minimal setup as last resort
                            minimal_params = {
                                'data': data,
                                'target': target_column,
                                'session_id': session_id
                            }
                            logging.warning(f"Attempting minimal setup with just: {minimal_params.keys()}")
                            s = setup(**minimal_params)
                            logging.info("PyCaret setup completed with minimal parameters")
                    else:
                        raise
                else:
                    raise
            except Exception as e:
                logging.error(f"Unexpected setup error: {str(e)}")
                # Try a minimal setup as last resort
                minimal_params = {
                    'data': data,
                    'target': target_column,
                    'session_id': session_id
                }
                logging.warning(f"Attempting minimal setup with just: {minimal_params.keys()}")
                s = setup(**minimal_params)
                logging.info("PyCaret setup completed with minimal parameters")
            
            # Prepare parameters for compare_models
            compare_params = {'n_select': n_select}
            
            if include_models:
                compare_params['include'] = [m.strip() for m in include_models.split(',')]
                
            if exclude_models:
                compare_params['exclude'] = [m.strip() for m in exclude_models.split(',')]
            
            # Compare baseline models with compatibility handling
            logging.info(f"Comparing models (selecting top {n_select} models)")
            try:
                best = compare_models(**compare_params)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Handle incompatible parameters
                    error_param = str(e).split("argument ")[-1].split("'")[1] if "'" in str(e) else str(e).split("argument ")[-1].strip()
                    logging.warning(f"Incompatible parameter detected in compare_models: {error_param}")
                    logging.warning(f"Removing parameter and retrying")
                    
                    if error_param in compare_params:
                        del compare_params[error_param]
                        best = compare_models(**compare_params)
                    else:
                        raise
                else:
                    raise
            except Exception as e:
                logging.error(f"Error in compare_models: {str(e)}")
                # Try with minimal parameters
                logging.warning("Attempting compare_models with minimal parameters")
                best = compare_models()
            
            # If only one model is returned, wrap it in a list
            if not isinstance(best, list):
                best = [best]
                logging.info("Only one model was selected")
            
            logging.info(f"Top {len(best)} models selected")
            
            # Initialize list to track created files
            created_files = []
            
            # Compare models results
            compare_results = pull()
            compare_results_path = os.path.join(output_dir, 'model_comparison_results.csv')
            compare_results.to_csv(compare_results_path, index=True)
            created_files.append(('Model Comparison Results', compare_results_path))
            logging.info(f"Saved model comparison results to {compare_results_path}")
            
            # create_plots_for_model function 
            def create_plots_for_model(model, plots_dir, created_files):
                plot_types = [
                    'auc', 'confusion_matrix', 'class_report', 'feature', 'boundary',
                    'pr', 'error', 'learning', 'manifold', 'calibration', 'vc', 'dimension', 
                    'feature_all', 'parameter', 'lift', 'gain']
                
                model_plots = []
                
                # Store the original working directory
                original_dir = os.getcwd()
                
                try:
                    # Change to the plots directory before creating plots
                    os.chdir(plots_dir)
                    logging.info(f"Changed working directory to {plots_dir}")
                    
                    # Try to create each plot type
                    for plot_type in plot_types:
                        try:
                            logging.info(f"Attempting to create {plot_type} plot in {plots_dir}")
                            
                            # Try to create the plot with save=True
                            plot_model(model, plot=plot_type, save=True)
                            
                            # Check if file was created
                            expected_filename = f"{plot_type}.png"
                            if os.path.exists(expected_filename):
                                full_path = os.path.join(plots_dir, expected_filename)
                                model_plots.append((f'Plot: {plot_type}', full_path))
                                logging.info(f"Successfully created {plot_type} plot at {full_path}")
                            else:
                                logging.warning(f"Plot {plot_type} was not found after creation")
                            
                        except Exception as e:
                            logging.warning(f"Failed to create {plot_type} plot: {str(e)}")
                
                finally:
                    # Always restore the original working directory
                    os.chdir(original_dir)
                    logging.info(f"Restored working directory to {original_dir}")
                
                return model_plots

            #create_interpretations_for_model function
            def create_interpretations_for_model(model, plots_dir, created_files):
                model_interpretations = []
                
                # Store the original working directory
                original_dir = os.getcwd()
                
                try:
                    # Change to the plots directory before creating interpretations
                    os.chdir(plots_dir)
                    logging.info(f"Changed working directory to {plots_dir}")
                    
                    # Try to create summary interpretation
                    try:
                        logging.info("Attempting to create model interpretation summary")
                        interpret_model(model, plot='summary', save=True)
                        
                        # Check if file was created
                        expected_filename = "SHAP Summary.png"
                        if os.path.exists(expected_filename):
                            full_path = os.path.join(plots_dir, expected_filename)
                            model_interpretations.append(('Interpretation: summary', full_path))
                            logging.info(f"Successfully created interpretation summary at {full_path}")
                        else:
                            logging.warning("Interpretation summary file was not found after creation")
                            
                    except Exception as e:
                        logging.warning(f"Failed to create summary interpretation: {str(e)}")
                    
                    # Try to create reason plot
                    try:
                        logging.info("Attempting to create model interpretation reason")
                        interpret_model(model, plot='reason', observation=1, save=True)
                        
                        # Check if file was created
                        expected_filename = "SHAP Reason Code.png"
                        if os.path.exists(expected_filename):
                            full_path = os.path.join(plots_dir, expected_filename)
                            model_interpretations.append(('Interpretation: reason', full_path))
                            logging.info(f"Successfully created interpretation reason at {full_path}")
                        else:
                            logging.warning("Interpretation reason file was not found after creation")
                            
                    except Exception as e:
                        logging.warning(f"Failed to create reason interpretation: {str(e)}")
                
                finally:
                    # Always restore the original working directory
                    os.chdir(original_dir)
                    logging.info(f"Restored working directory to {original_dir}")
                
                return model_interpretations
            
            # Function to evaluate model on test data 
            def evaluate_model_on_test_data(model, output_path, model_name):
                try:
                    logging.info(f"Evaluating {model_name} on test data using PyCaret's inherent workflow")
                    
                    # Use PyCaret's inherent workflow for holdout predictions
                    # This uses the test split that was created during setup()
                    holdout_pred = predict_model(model)
                    
                    # Save the predictions
                    holdout_pred.to_csv(output_path, index=False)
                    logging.info(f"Saved {model_name} test predictions to {output_path}")
                    return output_path
                except Exception as e:
                    logging.error(f"Error evaluating {model_name} on test data: {str(e)}")
                    return None

            
            # Create a top-level models directory
            models_dir = os.path.join(output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Dictionary to track all individual model paths
            individual_model_paths = {}
            model_test_predictions = {}
            
            # Tune and save each model individually
            tuned_models = []
            for i, model in enumerate(best):
                model_index = i + 1
                model_name = f"tuned_model_{model_index}"
                logging.info(f"Tuning model {model_index}/{len(best)}")
                
                try:
                    # Create a dedicated directory for this model
                    model_dir = os.path.join(models_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Tune the model
                    tuned_model = tune_model(model)
                    tuned_models.append(tuned_model)
                    
                    # Save tuned model results
                    tuned_results = pull()
                    tuned_results_path = os.path.join(model_dir, f'{model_name}_results.csv')
                    tuned_results.to_csv(tuned_results_path, index=True)
                    created_files.append((f'{model_name} Results', tuned_results_path))
                    logging.info(f"Saved {model_name} results to {tuned_results_path}")
                    
                    # Save the model itself
                    model_path = os.path.join(model_dir, f'{model_name}')
                    try:
                        save_model(tuned_model, model_path)
                        created_files.append((f'{model_name} Saved Model', model_path))
                        individual_model_paths[model_name] = model_path
                        logging.info(f"Saved {model_name} to {model_path}")
                    except Exception as e:
                        logging.error(f"Error saving {model_name}: {str(e)}")
                    
                    # Create plots directory for this model
                    plots_dir = os.path.join(model_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Generate plots for this model
                    logging.info(f"Generating plots for {model_name}")
                    model_plots = create_plots_for_model(tuned_model, plots_dir, created_files)
                    created_files.extend(model_plots)
                    
                    # Generate interpretations for this model
                    logging.info(f"Generating interpretations for {model_name}")
                    model_interpretations = create_interpretations_for_model(tuned_model, plots_dir, created_files)
                    created_files.extend(model_interpretations)
                    
                    # Evaluate on holdout set using PyCaret's inherent workflow
                    test_pred_path = os.path.join(model_dir, f'independent_eval_results_{model_name}.csv')
                    test_result = evaluate_model_on_test_data(tuned_model, test_pred_path, model_name)
                    if test_result:
                        created_files.append((f'{model_name} Test Predictions', test_result))
                        model_test_predictions[model_name] = test_result
                    
                except Exception as e:
                    logging.error(f"Error processing {model_name}: {str(e)}")
            
            # Create a blended model if we have multiple tuned models
            blended_model = None
            blended_model_path = None
            blended_test_pred_path = None
            
            if len(tuned_models) > 1:
                logging.info("Creating blended model from tuned models")
                try:
                    # Create dedicated directory for blended model
                    blended_dir = os.path.join(models_dir, "blended_model")
                    os.makedirs(blended_dir, exist_ok=True)
                    
                    # Create the blended model
                    blended_model = blend_models(tuned_models)
                    
                    # Save blended model results
                    blended_results = pull()
                    blended_results_path = os.path.join(blended_dir, 'blended_model_results.csv')
                    blended_results.to_csv(blended_results_path, index=True)
                    created_files.append(('Blended Model Results', blended_results_path))
                    logging.info(f"Saved blended model results to {blended_results_path}")
                    
                    # Save the blended model
                    blended_model_path = os.path.join(blended_dir, 'blended_model')
                    try:
                        save_model(blended_model, blended_model_path)
                        created_files.append(('Blended Model Saved Model', blended_model_path))
                        individual_model_paths["blended_model"] = blended_model_path
                        logging.info(f"Saved blended model to {blended_model_path}")
                    except Exception as e:
                        logging.error(f"Error saving blended model: {str(e)}")
                    
                    # Create plots directory for blended model
                    plots_dir = os.path.join(blended_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Generate plots for blended model
                    logging.info("Generating plots for blended model")
                    blended_plots = create_plots_for_model(blended_model, plots_dir, created_files)
                    created_files.extend(blended_plots)
                    
                    # Generate interpretations for blended model
                    logging.info("Generating interpretations for blended model")
                    blended_interpretations = create_interpretations_for_model(blended_model, plots_dir, created_files)
                    created_files.extend(blended_interpretations)
                    
                    # Evaluate on holdout set using PyCaret's inherent workflow
                    blended_test_pred_path = os.path.join(blended_dir, 'independent_eval_results_blended_model.csv')
                    test_result = evaluate_model_on_test_data(blended_model, blended_test_pred_path, "blended_model")
                    if test_result:
                        created_files.append(('Blended Model Test Predictions', test_result))
                        model_test_predictions["blended_model"] = test_result
                    
                    final_model = blended_model
                except Exception as e:
                    logging.error(f"Error creating blended model: {str(e)}")
                    logging.info("Using the best tuned model as the final model")
                    final_model = tuned_models[0]
            else:
                logging.info("Only one model available, using it as the final model")
                final_model = tuned_models[0] if tuned_models else best[0]
            
            # Initialize default paths
            final_model_path = None
            summary_path = None
            
            # Save the final model to the main output directory as well
            final_model_path = os.path.join(output_dir, f'{experiment_name}_final_model')
            try:
                save_model(final_model, final_model_path)
                created_files.append(('Final Model', final_model_path))
                logging.info(f"Saved final model to {final_model_path}")
            except Exception as e:
                logging.error(f"Error saving final model: {str(e)}")
                final_model_path = None
            
            # Create summary report with links to all models and evaluations
            try:
                summary_path = os.path.join(output_dir, "model_summary.csv")
                summary_data = []
                
                # Add individual models to summary
                for i, model in enumerate(tuned_models):
                    model_name = f"tuned_model_{i+1}"
                    model_path = individual_model_paths.get(model_name, "Not saved")
                    test_pred_path = model_test_predictions.get(model_name, "Not available")
                    
                    summary_data.append({
                        "Model": model_name,
                        "Type": "Tuned Individual Model",
                        "Model Path": model_path,
                        "Test Predictions": test_pred_path
                    })
                
                # Add blended model to summary if it exists
                if blended_model is not None:
                    model_path = individual_model_paths.get("blended_model", "Not saved")
                    test_pred_path = model_test_predictions.get("blended_model", "Not available")
                    
                    summary_data.append({
                        "Model": "blended_model",
                        "Type": "Blended Model",
                        "Model Path": model_path,
                        "Test Predictions": test_pred_path
                    })
                
                # Write summary to CSV
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(summary_path, index=False)
                created_files.append(('Model Summary', summary_path))
                logging.info(f"Created model summary at {summary_path}")
            except Exception as e:
                logging.error(f"Error creating model summary: {str(e)}")
                summary_path = None
            
            logging.info("Classification modeling completed successfully")
            
            return {
                "status": "success",
                "experiment_name": experiment_name,
                "input_path": input_path,
                "output_dir": output_dir,
                "final_model_path": final_model_path,
                "individual_model_paths": individual_model_paths,
                "model_test_predictions": model_test_predictions,
                "model_summary": summary_path,
                "log_file": log_file,
                "created_files": created_files
            }
        
        except Exception as e:
            logging.error(f"Error in PyCaret classification: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "input_path": input_path,
                "output_dir": output_dir
            }

class PyCaretInferenceTool(Tool):
    name = "pycaret_inference"
    description = """
    This tool uses a saved PyCaret classification model to make predictions on new data.
    It can also calculate performance metrics if ground truth data is available.
    """
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input data file (CSV format) for inference"
        },
        "model_path": {
            "type": "string",
            "description": "Path to the saved PyCaret model"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where prediction results will be saved (if not specified, uses same directory as input file)",
            "required": False,
            "nullable": True
        },
        "ground_truth_column": {
            "type": "string",
            "description": "Name of the column containing ground truth values (if available for metrics calculation)",
            "required": False,
            "nullable": True
        },
        "prediction_filename": {
            "type": "string",
            "description": "Name for the output prediction file (default: predictions.csv)",
            "required": False,
            "nullable": True
        },
        "verbose": {
            "type": "boolean",
            "description": "Whether to print detailed logs",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        model_path: str,
        output_dir: Optional[str] = None,
        ground_truth_column: Optional[str] = None,
        prediction_filename: Optional[str] = "predictions.csv",
        verbose: Optional[bool] = True
    ):
        """
        Run inference using a saved PyCaret model on new data.
        
        Args:
            input_path: Path to input CSV file with data for inference
            model_path: Path to the saved PyCaret model
            output_dir: Directory to save prediction outputs (default: same directory as input file)
            ground_truth_column: Name of the column containing ground truth values (if available)
            prediction_filename: Name for the output prediction file
            verbose: Whether to print detailed logs
            
        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set output directory (default to input file directory if not specified)
            if output_dir is None:
                output_dir = os.path.dirname(input_path)
                if not output_dir:  # If input_path doesn't have a directory component
                    output_dir = "."
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up logging with fixed configuration
            log_file = os.path.join(output_dir, "pycaret_inference.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO if verbose else logging.WARNING)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            logging.info(f"Starting inference using model: {model_path}")
            logging.info(f"Input data: {input_path}")
            logging.info(f"Output directory: {output_dir}")
            
            # Import PyCaret's classification module
            try:
                from pycaret.classification import load_model, predict_model
                logging.info("PyCaret imported successfully")
            except ImportError as e:
                logging.error(f"Error importing PyCaret: {str(e)}")
                logging.error("Please install PyCaret: pip install pycaret")
                raise ImportError("PyCaret is required for this tool. Please install it with: pip install pycaret")
            
            # Load the data
            logging.info(f"Loading data from {input_path}")
            try:
                data = pd.read_csv(input_path)
                logging.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
                logging.info(f"Data columns: {data.columns.tolist()}")
            except Exception as e:
                logging.error(f"Error loading data: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load data: {str(e)}",
                    "input_path": input_path
                }
            
            # Check if ground truth column exists if specified
            has_ground_truth = False
            original_ground_truth = None
            if ground_truth_column:
                if ground_truth_column in data.columns:
                    has_ground_truth = True
                    logging.info(f"Ground truth column '{ground_truth_column}' found in dataset")
                    # Store original ground truth data before any modifications
                    original_ground_truth = data[ground_truth_column].copy()
                else:
                    logging.warning(f"Ground truth column '{ground_truth_column}' not found in the dataset. Metrics won't be calculated.")
            
            # Fix model path - remove .pkl extension if present since PyCaret adds it automatically
            fixed_model_path = model_path
            if fixed_model_path.endswith('.pkl'):
                fixed_model_path = fixed_model_path[:-4]
                logging.info(f"Removed .pkl extension from model path, using: {fixed_model_path}")
            
            # Check if the model directory exists
            model_dir = os.path.dirname(fixed_model_path)
            if not os.path.exists(model_dir):
                logging.error(f"Model directory does not exist: {model_dir}")
                return {
                    "status": "error",
                    "error_message": f"Model directory not found: {model_dir}",
                    "model_path": model_path
                }
            
            # Check if model file or directory exists
            if not os.path.exists(fixed_model_path) and not os.path.exists(fixed_model_path + '.pkl'):
                # Try to find the model in the directory by listing all files
                model_dir = os.path.dirname(fixed_model_path)
                model_basename = os.path.basename(fixed_model_path)
                
                logging.warning(f"Model not found at: {fixed_model_path} or {fixed_model_path + '.pkl'}")
                logging.info(f"Searching for model in directory: {model_dir}")
                
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logging.info(f"Files in directory: {files}")
                    
                    # Try to find files that match the model basename
                    potential_models = [f for f in files if f.startswith(model_basename)]
                    if potential_models:
                        potential_model = os.path.join(model_dir, potential_models[0])
                        logging.info(f"Found potential model: {potential_model}")
                        fixed_model_path = potential_model
                        if fixed_model_path.endswith('.pkl'):
                            fixed_model_path = fixed_model_path[:-4]
                
            # Load the model with enhanced error reporting
            logging.info(f"Loading model from {fixed_model_path}")
            try:
                model = load_model(fixed_model_path)
                logging.info("Model loaded successfully")
            except FileNotFoundError as e:
                logging.error(f"File not found error: {str(e)}")
                # Try to list available files in the directory
                model_dir = os.path.dirname(fixed_model_path)
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logging.error(f"Files available in the directory: {files}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model (file not found): {str(e)}",
                    "model_path": model_path,
                    "fixed_model_path": fixed_model_path
                }
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model: {str(e)}",
                    "model_path": model_path,
                    "fixed_model_path": fixed_model_path
                }
            
            # Make predictions
            logging.info("Making predictions")
            try:
                # Try to get the raw scores (probabilities) for metrics
                try:
                    predictions = predict_model(model, data=data, raw_score=True)
                    logging.info("Predictions with raw scores generated successfully")
                except Exception as e:
                    logging.warning(f"Error generating raw scores: {str(e)}. Falling back to standard predictions.")
                    predictions = predict_model(model, data=data)
                    logging.info("Standard predictions generated successfully")
            except Exception as e:
                logging.error(f"Error making predictions: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to make predictions: {str(e)}"
                }
            
            # Log the column names to help debug
            logging.info(f"Prediction columns: {predictions.columns.tolist()}")
            
            # Save predictions
            predictions_path = os.path.join(output_dir, prediction_filename)
            try:
                predictions.to_csv(predictions_path, index=False)
                logging.info(f"Predictions saved to {predictions_path}")
            except Exception as e:
                logging.error(f"Error saving predictions: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to save predictions: {str(e)}",
                    "predictions": predictions  # Still return predictions in memory even if saving failed
                }
            
            # Initialize variables for metrics calculation
            metrics = {}
            metrics_df = None
            comparison_path = None
            confusion_matrix_path = None
            metrics_csv_path = None
            metrics_json_path = None
            
            # Calculate metrics if ground truth is available
            if has_ground_truth and original_ground_truth is not None:
                logging.info(f"Calculating performance metrics using '{ground_truth_column}' as ground truth")
                try:
                    # Get actual values using the stored ground truth
                    y_true = original_ground_truth
                    
                    # Determine prediction column (PyCaret usually adds 'Label' column)
                    pred_col_candidates = ['Label', 'prediction_label', 'predicted_y']
                    pred_col = None
                    
                    for col in pred_col_candidates:
                        if col in predictions.columns:
                            pred_col = col
                            break
                    
                    if pred_col is None:
                        # Fall back to the last column if no standard name is found
                        pred_col = predictions.columns[-1]
                        logging.warning(f"No standard prediction column found. Using '{pred_col}' as the prediction column.")
                    
                    logging.info(f"Using '{pred_col}' as prediction column for metrics")
                    
                    y_pred = predictions[pred_col]
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(y_true, y_pred)
                    metrics['accuracy'] = accuracy
                    logging.info(f"Accuracy: {accuracy:.4f}")
                    
                    # Create a DataFrame for prediction vs actual comparison
                    comparison_df = pd.DataFrame({
                        'actual': y_true,
                        'predicted': y_pred
                    })
                    
                    # Get class labels to determine binary or multiclass
                    unique_classes = np.unique(y_true)
                    is_binary = len(unique_classes) == 2
                    
                    # Calculate confusion matrix and save it
                    cm = confusion_matrix(y_true, y_pred)
                    cm_df = pd.DataFrame(cm)
                    
                    # Add class labels if available
                    if hasattr(model, 'classes_'):
                        cm_df.index = model.classes_
                        cm_df.columns = model.classes_
                        cm_df.index.name = 'Actual'
                        cm_df.columns.name = 'Predicted'
                    else:
                        cm_df.index = [f'Actual_{i}' for i in range(len(cm_df))]
                        cm_df.columns = [f'Predicted_{i}' for i in range(len(cm_df.columns))]
                    
                    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.csv")
                    cm_df.to_csv(confusion_matrix_path)
                    logging.info(f"Confusion matrix saved to {confusion_matrix_path}")
                    
                    # For binary classification, calculate additional metrics
                    if is_binary:
                        logging.info("Binary classification detected, calculating additional metrics")
                        
                        # Get unique classes and determine positive class
                        classes = sorted(unique_classes)
                        pos_class = classes[1]  # Usually the higher value is considered positive class
                        logging.info(f"Classes: {classes}, using {pos_class} as positive class")
                        
                        # Find probability columns
                        prob_cols = [col for col in predictions.columns if col.startswith('Score_')]
                        
                        # If no Score_ columns, look for other probability columns
                        if not prob_cols:
                            prob_cols = [col for col in predictions.columns 
                                        if 'probability' in col.lower() or 'score' in col.lower()]
                        
                        # Log available probability columns to help debugging
                        if prob_cols:
                            logging.info(f"Available probability columns: {prob_cols}")
                            
                            # Try to find column for positive class probability
                            pos_class_cols = [col for col in prob_cols 
                                            if str(pos_class) in col or '1' in col or 'positive' in col.lower()]
                            
                            if pos_class_cols:
                                # Prefer columns that seem to match positive class
                                prob_col = pos_class_cols[0]
                                logging.info(f"Found column matching positive class: {prob_col}")
                            else:
                                # Fallback to first probability column
                                prob_col = prob_cols[0]
                                logging.info(f"No column matching positive class found, using first probability column: {prob_col}")
                                
                            logging.info(f"Using '{prob_col}' for AUC calculation (positive class: {pos_class})")
                            
                            try:
                                y_prob = predictions[prob_col]
                                
                                # Add probability to comparison DataFrame
                                comparison_df['probability'] = y_prob
                                
                                # Calculate AUC
                                auc = roc_auc_score(y_true, y_prob)
                                metrics['auc'] = auc
                                logging.info(f"AUC: {auc:.4f}")
                            except Exception as e:
                                logging.warning(f"AUC calculation failed: {str(e)}")
                                # Try alternative approach with predict_proba if available
                                try:
                                    if hasattr(model, 'predict_proba'):
                                        logging.info("Attempting AUC calculation using model.predict_proba")
                                        proba = model.predict_proba(data)
                                        # Find the index of the positive class
                                        if hasattr(model, 'classes_'):
                                            pos_idx = np.where(model.classes_ == pos_class)[0][0]
                                            y_prob = proba[:, pos_idx]
                                        else:
                                            # If classes_ not available, assume second column is positive class
                                            y_prob = proba[:, 1]
                                            
                                        # Add probability to comparison DataFrame
                                        comparison_df['probability'] = y_prob
                                        
                                        auc = roc_auc_score(y_true, y_prob)
                                        metrics['auc'] = auc
                                        logging.info(f"AUC (using predict_proba): {auc:.4f}")
                                except Exception as inner_e:
                                    logging.warning(f"Alternative AUC calculation also failed: {str(inner_e)}")
                        else:
                            logging.warning("No probability columns found for AUC calculation")
                        
                        # Calculate other binary classification metrics
                        try:
                            # Extract confusion matrix values for binary case
                            tn, fp, fn, tp = cm.ravel()
                            
                            # Calculate precision, recall, f1
                            precision = precision_score(y_true, y_pred)
                            recall = recall_score(y_true, y_pred)  # Same as sensitivity
                            f1 = f1_score(y_true, y_pred)
                            
                            # Calculate specificity
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            
                            # Store metrics
                            metrics['precision'] = precision
                            metrics['recall'] = recall
                            metrics['sensitivity'] = recall  # Sensitivity is the same as recall
                            metrics['specificity'] = specificity
                            metrics['f1'] = f1
                            
                            # Log all metrics
                            logging.info(f"Precision: {precision:.4f}")
                            logging.info(f"Recall/Sensitivity: {recall:.4f}")
                            logging.info(f"Specificity: {specificity:.4f}")
                            logging.info(f"F1 Score: {f1:.4f}")
                            
                            # Add confusion matrix details for easier interpretation
                            metrics['true_positives'] = int(tp)
                            metrics['false_positives'] = int(fp)
                            metrics['true_negatives'] = int(tn)
                            metrics['false_negatives'] = int(fn)
                            
                        except Exception as e:
                            logging.warning(f"Could not calculate some binary metrics: {str(e)}")
                    else:
                        # For multiclass, add appropriate metrics
                        logging.info("Multiclass classification detected")
                        
                        try:
                            # Multiclass precision, recall, f1 with average
                            precision_macro = precision_score(y_true, y_pred, average='macro')
                            recall_macro = recall_score(y_true, y_pred, average='macro')
                            f1_macro = f1_score(y_true, y_pred, average='macro')
                            
                            metrics['precision_macro'] = precision_macro
                            metrics['recall_macro'] = recall_macro
                            metrics['f1_macro'] = f1_macro
                            
                            logging.info(f"Precision (macro): {precision_macro:.4f}")
                            logging.info(f"Recall (macro): {recall_macro:.4f}")
                            logging.info(f"F1 Score (macro): {f1_macro:.4f}")
                            
                            # Try to calculate multiclass AUC if applicable
                            try:
                                # Check for probability columns
                                prob_cols = [col for col in predictions.columns if col.startswith('Score_')]
                                if len(prob_cols) > 1:
                                    # Get probabilities for each class
                                    class_probs = predictions[prob_cols].values
                                    
                                    # Calculate one-vs-rest AUC
                                    from sklearn.preprocessing import label_binarize
                                    
                                    # Convert y_true to one-hot encoding
                                    classes = sorted(unique_classes)
                                    y_true_bin = label_binarize(y_true, classes=classes)
                                    
                                    # If only two classes, reshape to work with roc_auc_score
                                    if len(classes) == 2:
                                        y_true_bin = y_true_bin.reshape(-1, 1)
                                    
                                    # Calculate AUC for multiclass
                                    if len(classes) > 2:
                                        auc_multiclass = roc_auc_score(y_true_bin, class_probs, multi_class='ovr', average='macro')
                                        metrics['auc_macro'] = auc_multiclass
                                        logging.info(f"AUC (macro): {auc_multiclass:.4f}")
                            except Exception as e:
                                logging.warning(f"Could not calculate multiclass AUC: {str(e)}")
                                
                        except Exception as e:
                            logging.warning(f"Could not calculate some multiclass metrics: {str(e)}")
                    
                    # Save the prediction vs actual comparison
                    comparison_path = os.path.join(output_dir, "prediction_vs_actual.csv")
                    comparison_df.to_csv(comparison_path, index=False)
                    logging.info(f"Prediction vs actual comparison saved to {comparison_path}")
                    
                    # Create a DataFrame for the metrics report
                    metrics_df = pd.DataFrame([metrics])
                    
                    # Save metrics to CSV file
                    metrics_csv_path = os.path.join(output_dir, "metrics_report.csv")
                    metrics_df.to_csv(metrics_csv_path, index=False)
                    logging.info(f"Metrics report saved to CSV: {metrics_csv_path}")
                    
                    # Save metrics to JSON file (for compatibility)
                    metrics_json_path = os.path.join(output_dir, "metrics.json")
                    with open(metrics_json_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    logging.info(f"Performance metrics saved to JSON: {metrics_json_path}")
                    
                    logging.info(f"Metrics summary: {metrics}")
                    
                except Exception as e:
                    logging.error(f"Error calculating metrics: {str(e)}", exc_info=True)
                    # Still continue, as predictions were made
            
            return {
                "status": "success",
                "predictions_path": predictions_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_df": metrics_df.to_dict('records')[0] if metrics_df is not None else None,
                "metrics_csv_path": metrics_csv_path,
                "metrics_json_path": metrics_json_path,
                "comparison_path": comparison_path,
                "confusion_matrix_path": confusion_matrix_path,
                "input_path": input_path,
                "model_path": model_path,
                "fixed_model_path": fixed_model_path,
                "output_dir": output_dir,
                "num_predictions": len(predictions) if predictions is not None else 0,
                "has_ground_truth": has_ground_truth
            }
        
        except Exception as e:
            logging.error(f"Unhandled error in PyCaret inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "input_path": input_path,
                "model_path": model_path
            }

#Regression Model Training and Inference Tool (for tabulated data)

class PyCaretRegressionInferenceTool(Tool):
    name = "pycaret_regression_inference"
    description = """
    This tool uses a saved PyCaret regression model to make predictions on new data.
    It can also calculate performance metrics if ground truth data is available.
    """
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input data file (CSV format) for inference"
        },
        "model_path": {
            "type": "string",
            "description": "Path to the saved PyCaret model"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where prediction results will be saved (if not specified, uses same directory as input file)",
            "required": False,
            "nullable": True
        },
        "ground_truth_column": {
            "type": "string",
            "description": "Name of the column containing ground truth values (if available for metrics calculation)",
            "required": False,
            "nullable": True
        },
        "prediction_filename": {
            "type": "string",
            "description": "Name for the output prediction file (default: predictions.csv)",
            "required": False,
            "nullable": True
        },
        "verbose": {
            "type": "boolean",
            "description": "Whether to print detailed logs",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        model_path: str,
        output_dir: Optional[str] = None,
        ground_truth_column: Optional[str] = None,
        prediction_filename: Optional[str] = "predictions.csv",
        verbose: Optional[bool] = True
    ):
        """
        Run inference using a saved PyCaret regression model on new data.
        
        Args:
            input_path: Path to input CSV file with data for inference
            model_path: Path to the saved PyCaret model
            output_dir: Directory to save prediction outputs (default: same directory as input file)
            ground_truth_column: Name of the column containing ground truth values (if available)
            prediction_filename: Name for the output prediction file
            verbose: Whether to print detailed logs
            
        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set output directory (default to input file directory if not specified)
            if output_dir is None:
                output_dir = os.path.dirname(input_path)
                if not output_dir:  # If input_path doesn't have a directory component
                    output_dir = "."
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up logging with fixed configuration
            log_file = os.path.join(output_dir, "pycaret_regression_inference.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO if verbose else logging.WARNING)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            logging.info(f"Starting inference using model: {model_path}")
            logging.info(f"Input data: {input_path}")
            logging.info(f"Output directory: {output_dir}")
            
            # Import PyCaret's regression module
            try:
                from pycaret.regression import load_model, predict_model
                logging.info("PyCaret imported successfully")
            except ImportError as e:
                logging.error(f"Error importing PyCaret: {str(e)}")
                logging.error("Please install PyCaret: pip install pycaret")
                raise ImportError("PyCaret is required for this tool. Please install it with: pip install pycaret")
            
            # Load the data
            logging.info(f"Loading data from {input_path}")
            try:
                data = pd.read_csv(input_path)
                logging.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
                logging.info(f"Data columns: {data.columns.tolist()}")
            except Exception as e:
                logging.error(f"Error loading data: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load data: {str(e)}",
                    "input_path": input_path
                }
            
            # Check if ground truth column exists if specified
            has_ground_truth = False
            original_ground_truth = None
            if ground_truth_column:
                if ground_truth_column in data.columns:
                    has_ground_truth = True
                    logging.info(f"Ground truth column '{ground_truth_column}' found in dataset")
                    # Store original ground truth data before any modifications
                    original_ground_truth = data[ground_truth_column].copy()
                else:
                    logging.warning(f"Ground truth column '{ground_truth_column}' not found in the dataset. Metrics won't be calculated.")
            
            # Fix model path - remove .pkl extension if present since PyCaret adds it automatically
            fixed_model_path = model_path
            if fixed_model_path.endswith('.pkl'):
                fixed_model_path = fixed_model_path[:-4]
                logging.info(f"Removed .pkl extension from model path, using: {fixed_model_path}")
            
            # Check if the model directory exists
            model_dir = os.path.dirname(fixed_model_path)
            if not os.path.exists(model_dir):
                logging.error(f"Model directory does not exist: {model_dir}")
                return {
                    "status": "error",
                    "error_message": f"Model directory not found: {model_dir}",
                    "model_path": model_path
                }
            
            # Check if model file or directory exists
            if not os.path.exists(fixed_model_path) and not os.path.exists(fixed_model_path + '.pkl'):
                # Try to find the model in the directory by listing all files
                model_dir = os.path.dirname(fixed_model_path)
                model_basename = os.path.basename(fixed_model_path)
                
                logging.warning(f"Model not found at: {fixed_model_path} or {fixed_model_path + '.pkl'}")
                logging.info(f"Searching for model in directory: {model_dir}")
                
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logging.info(f"Files in directory: {files}")
                    
                    # Try to find files that match the model basename
                    potential_models = [f for f in files if f.startswith(model_basename)]
                    if potential_models:
                        potential_model = os.path.join(model_dir, potential_models[0])
                        logging.info(f"Found potential model: {potential_model}")
                        fixed_model_path = potential_model
                        if fixed_model_path.endswith('.pkl'):
                            fixed_model_path = fixed_model_path[:-4]
                
            # Load the model with enhanced error reporting
            logging.info(f"Loading model from {fixed_model_path}")
            try:
                model = load_model(fixed_model_path)
                logging.info("Model loaded successfully")
            except FileNotFoundError as e:
                logging.error(f"File not found error: {str(e)}")
                # Try to list available files in the directory
                model_dir = os.path.dirname(fixed_model_path)
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logging.error(f"Files available in the directory: {files}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model (file not found): {str(e)}",
                    "model_path": model_path,
                    "fixed_model_path": fixed_model_path
                }
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model: {str(e)}",
                    "model_path": model_path,
                    "fixed_model_path": fixed_model_path
                }
            
            # Make predictions
            logging.info("Making predictions")
            try:
                predictions = predict_model(model, data=data)
                logging.info("Predictions generated successfully")
            except Exception as e:
                logging.error(f"Error making predictions: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to make predictions: {str(e)}"
                }
            
            # Log the column names to help debug
            logging.info(f"Prediction columns: {predictions.columns.tolist()}")
            
            # Save predictions
            predictions_path = os.path.join(output_dir, prediction_filename)
            try:
                predictions.to_csv(predictions_path, index=False)
                logging.info(f"Predictions saved to {predictions_path}")
            except Exception as e:
                logging.error(f"Error saving predictions: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to save predictions: {str(e)}",
                    "predictions": predictions  # Still return predictions in memory even if saving failed
                }
            
            # Initialize variables for metrics calculation
            metrics = {}
            metrics_df = None
            comparison_path = None
            residuals_path = None
            metrics_csv_path = None
            metrics_json_path = None
            
            # Calculate metrics if ground truth is available
            if has_ground_truth and original_ground_truth is not None:
                logging.info(f"Calculating performance metrics using '{ground_truth_column}' as ground truth")
                try:
                    # Get actual values using the stored ground truth
                    y_true = original_ground_truth
                    
                    # Determine prediction column (PyCaret usually adds 'Label' or 'prediction_label' column)
                    pred_col_candidates = ['Label', 'prediction_label', 'predicted_y', 'Prediction']
                    pred_col = None
                    
                    for col in pred_col_candidates:
                        if col in predictions.columns:
                            pred_col = col
                            break
                    
                    if pred_col is None:
                        # Fall back to the last column if no standard name is found
                        pred_col = predictions.columns[-1]
                        logging.warning(f"No standard prediction column found. Using '{pred_col}' as the prediction column.")
                    
                    logging.info(f"Using '{pred_col}' as prediction column for metrics")
                    
                    y_pred = predictions[pred_col]
                    
                    # Create a DataFrame for prediction vs actual comparison
                    comparison_df = pd.DataFrame({
                        'actual': y_true,
                        'predicted': y_pred
                    })
                    
                    # Calculate residuals
                    comparison_df['residual'] = y_true - y_pred
                    comparison_df['abs_residual'] = np.abs(comparison_df['residual'])
                    comparison_df['squared_residual'] = comparison_df['residual'] ** 2
                    
                    # Calculate percent error where actual != 0
                    comparison_df['percent_error'] = np.nan
                    nonzero_idx = y_true != 0
                    if np.any(nonzero_idx):
                        comparison_df.loc[nonzero_idx, 'percent_error'] = (
                            np.abs(y_true[nonzero_idx] - y_pred[nonzero_idx]) / np.abs(y_true[nonzero_idx])
                        ) * 100
                    
                    # Calculate regression metrics
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    explained_var = explained_variance_score(y_true, y_pred)
                    
                    # Try to calculate MAPE with handling for zero values
                    try:
                        # Calculate MAPE only on non-zero actual values
                        if np.any(nonzero_idx):
                            mape = mean_absolute_percentage_error(y_true[nonzero_idx], y_pred[nonzero_idx]) * 100
                        else:
                            mape = np.nan
                    except Exception as e:
                        logging.warning(f"Could not calculate MAPE: {str(e)}")
                        mape = np.nan
                    
                    # Store metrics
                    metrics['mse'] = mse
                    metrics['rmse'] = rmse
                    metrics['mae'] = mae
                    metrics['r2'] = r2
                    metrics['explained_variance'] = explained_var
                    
                    if not np.isnan(mape):
                        metrics['mape'] = mape
                    
                    # Additional descriptive statistics on residuals
                    metrics['mean_residual'] = np.mean(comparison_df['residual'])
                    metrics['median_residual'] = np.median(comparison_df['residual'])
                    metrics['min_residual'] = np.min(comparison_df['residual'])
                    metrics['max_residual'] = np.max(comparison_df['residual'])
                    metrics['std_residual'] = np.std(comparison_df['residual'])
                    
                    # Log all metrics
                    logging.info(f"MSE: {mse:.4f}")
                    logging.info(f"RMSE: {rmse:.4f}")
                    logging.info(f"MAE: {mae:.4f}")
                    logging.info(f"R²: {r2:.4f}")
                    logging.info(f"Explained Variance: {explained_var:.4f}")
                    if not np.isnan(mape):
                        logging.info(f"MAPE: {mape:.4f}%")
                    
                    # Create a residual analysis table
                    residual_bins = pd.cut(comparison_df['residual'], bins=10)
                    residual_analysis = comparison_df.groupby(residual_bins)['residual'].agg(['count', 'mean', 'min', 'max'])
                    residual_analysis = residual_analysis.reset_index()
                    residual_analysis['bin'] = residual_analysis['residual'].astype(str)
                    residual_analysis = residual_analysis.drop('residual', axis=1)
                    
                    # Save residual analysis
                    residuals_path = os.path.join(output_dir, "residual_analysis.csv")
                    residual_analysis.to_csv(residuals_path, index=False)
                    logging.info(f"Residual analysis saved to {residuals_path}")
                    
                    # Save the prediction vs actual comparison
                    comparison_path = os.path.join(output_dir, "prediction_vs_actual.csv")
                    comparison_df.to_csv(comparison_path, index=False)
                    logging.info(f"Prediction vs actual comparison saved to {comparison_path}")
                    
                    # Create a DataFrame for the metrics report
                    metrics_df = pd.DataFrame([metrics])
                    
                    # Save metrics to CSV file
                    metrics_csv_path = os.path.join(output_dir, "metrics_report.csv")
                    metrics_df.to_csv(metrics_csv_path, index=False)
                    logging.info(f"Metrics report saved to CSV: {metrics_csv_path}")
                    
                    # Save metrics to JSON file (for compatibility)
                    metrics_json_path = os.path.join(output_dir, "metrics.json")
                    with open(metrics_json_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    logging.info(f"Performance metrics saved to JSON: {metrics_json_path}")
                    
                    logging.info(f"Metrics summary: {metrics}")
                    
                except Exception as e:
                    logging.error(f"Error calculating metrics: {str(e)}", exc_info=True)
                    # Still continue, as predictions were made
            
            return {
                "status": "success",
                "predictions_path": predictions_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_df": metrics_df.to_dict('records')[0] if metrics_df is not None else None,
                "metrics_csv_path": metrics_csv_path,
                "metrics_json_path": metrics_json_path,
                "comparison_path": comparison_path,
                "residuals_path": residuals_path,
                "input_path": input_path,
                "model_path": model_path,
                "fixed_model_path": fixed_model_path,
                "output_dir": output_dir,
                "num_predictions": len(predictions) if predictions is not None else 0,
                "has_ground_truth": has_ground_truth
            }
        
        except Exception as e:
            logging.error(f"Unhandled error in PyCaret regression inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "input_path": input_path,
                "model_path": model_path
            }


class PyCaretRegressionTool(Tool):
    name = "pycaret_regression"
    description = """
    This tool uses PyCaret to train and evaluate regression models.
    It compares multiple models, tunes the best ones, creates a blended model,
    and generates various visualizations and interpretations.
    Results are saved to the specified output directory.
    """
    
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to input data file (CSV format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory where model and results will be saved"
        },
        "target_column": {
            "type": "string",
            "description": "Name of the target column for regression"
        },
        "experiment_name": {
            "type": "string",
            "description": "Name of the experiment",
            "required": False,
            "nullable": True
        },
        "fold": {
            "type": "integer",
            "description": "Number of cross-validation folds",
            "required": False,
            "nullable": True
        },
        "session_id": {
            "type": "integer",
            "description": "Random seed for reproducibility",
            "required": False,
            "nullable": True
        },
        "use_gpu": {
            "type": "boolean",
            "description": "Whether to use GPU for training (if available)",
            "required": False,
            "nullable": True
        },
        "data_split_stratify": {
            "type": "boolean",
            "description": "Whether to use stratified sampling for data splitting",
            "required": False,
            "nullable": True
        },
        "data_split_shuffle": {
            "type": "boolean",
            "description": "Whether to shuffle data before splitting",
            "required": False,
            "nullable": True
        },
        "preprocess": {
            "type": "boolean",
            "description": "Whether to apply preprocessing steps",
            "required": False,
            "nullable": True
        },
        "ignore_features": {
            "type": "string",
            "description": "Comma-separated list of features to ignore during training",
            "required": False,
            "nullable": True
        },
        "numeric_features": {
            "type": "string",
            "description": "Comma-separated list of numeric features",
            "required": False,
            "nullable": True
        },
        "categorical_features": {
            "type": "string",
            "description": "Comma-separated list of categorical features",
            "required": False,
            "nullable": True
        },
        "date_features": {
            "type": "string",
            "description": "Comma-separated list of date features",
            "required": False,
            "nullable": True
        },
        "n_select": {
            "type": "integer",
            "description": "Number of top models to select for blending",
            "required": False,
            "nullable": True
        },
        "normalize": {
            "type": "boolean",
            "description": "Whether to normalize numeric features",
            "required": False,
            "nullable": True
        },
        "transformation": {
            "type": "boolean",
            "description": "Whether to apply transformation to numeric features",
            "required": False,
            "nullable": True
        },
        "pca": {
            "type": "boolean",
            "description": "Whether to apply PCA for dimensionality reduction",
            "required": False,
            "nullable": True
        },
        "pca_components": {
            "type": "number",
            "description": "Number of PCA components (float between 0-1 or int > 1)",
            "required": False,
            "nullable": True
        },
        "include_models": {
            "type": "string",
            "description": "Comma-separated list of models to include in comparison",
            "required": False,
            "nullable": True
        },
        "exclude_models": {
            "type": "string",
            "description": "Comma-separated list of models to exclude from comparison",
            "required": False,
            "nullable": True
        },
        "test_data_path": {
            "type": "string",
            "description": "Path to test/holdout data for independent evaluation",
            "required": False,
            "nullable": True
        },
        "ignore_gpu_errors": {
            "type": "boolean",
            "description": "Whether to ignore GPU-related errors and fall back to CPU",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"

    def forward(
        self,
        input_path: str,
        output_dir: str,
        target_column: str,
        experiment_name: Optional[str] = None,
        fold: Optional[int] = 10,
        session_id: Optional[int] = None,
        use_gpu: Optional[bool] = False,
        data_split_stratify: Optional[bool] = False,  # Usually False for regression
        data_split_shuffle: Optional[bool] = True,
        preprocess: Optional[bool] = True,
        ignore_features: Optional[str] = None,
        numeric_features: Optional[str] = None,
        categorical_features: Optional[str] = None,
        date_features: Optional[str] = None,
        n_select: Optional[int] = 3,
        normalize: Optional[bool] = True,  # Usually True for regression
        transformation: Optional[bool] = True,  # Usually True for regression
        pca: Optional[bool] = False,
        pca_components: Optional[float] = None,
        include_models: Optional[str] = None,
        exclude_models: Optional[str] = None,
        test_data_path: Optional[str] = None,
        ignore_gpu_errors: Optional[bool] = True
    ):
        """
        Train and evaluate regression models using PyCaret.
        
        Args:
            input_path: Path to input CSV file with training data
            output_dir: Directory to save outputs
            target_column: Target column name for regression
            experiment_name: Name of the experiment
            fold: Number of cross-validation folds
            session_id: Random seed for reproducibility
            use_gpu: Whether to use GPU for training
            data_split_stratify: Whether to use stratified sampling
            data_split_shuffle: Whether to shuffle data before splitting
            preprocess: Whether to apply preprocessing steps
            ignore_features: Comma-separated list of features to ignore
            numeric_features: Comma-separated list of numeric features
            categorical_features: Comma-separated list of categorical features
            date_features: Comma-separated list of date features
            n_select: Number of top models to select for blending
            normalize: Whether to normalize numeric features
            transformation: Whether to apply transformation to numeric features
            pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components
            include_models: Comma-separated list of models to include
            exclude_models: Comma-separated list of models to exclude
            test_data_path: Path to test/holdout data for independent evaluation
            ignore_gpu_errors: Whether to ignore GPU errors and fallback to CPU
            
        Returns:
            Dictionary with model training results and file paths
        """
        try:
            # Setup logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "pycaret_regression.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Generate experiment name if not provided
            if experiment_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                experiment_name = f"regression_exp_{timestamp}"
            
            # Generate session_id if not provided
            if session_id is None:
                import random
                session_id = random.randint(1, 10000)
            
            logging.info(f"Starting regression experiment: {experiment_name}")
            logging.info(f"Input data: {input_path}")
            logging.info(f"Output directory: {output_dir}")
            logging.info(f"Target column: {target_column}")
            
            # Import PyCaret's regression module
            try:
                from pycaret.regression import (
                    setup, compare_models, tune_model, blend_models, 
                    pull, predict_model, save_model, load_model,
                    plot_model, interpret_model
                )
                
                # Check PyCaret version for compatibility adjustments
                import pycaret
                pycaret_version = getattr(pycaret, "__version__", "unknown")
                logging.info(f"PyCaret version: {pycaret_version}")
                
                logging.info("PyCaret imported successfully")
            except ImportError as e:
                logging.error(f"Error importing PyCaret: {str(e)}")
                logging.error("Please install PyCaret: pip install pycaret")
                raise ImportError("PyCaret is required for this tool. Please install it with: pip install pycaret")
            
            # Load data
            logging.info(f"Loading data from {input_path}")
            data = pd.read_csv(input_path)
            logging.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Verify target column exists
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset")
            
            # Prepare parameters for setup
            setup_params = {
                'data': data,
                'target': target_column,
                'session_id': session_id,
                'experiment_name': experiment_name,
                'fold': fold,
                'use_gpu': use_gpu,
                'preprocess': preprocess,
                'data_split_shuffle': data_split_shuffle,
                'log_experiment': False  # Set to False to avoid MLflow errors
            }
            
            # Add stratify only if it's supported in regression context
            if data_split_stratify is not None:
                setup_params['data_split_stratify'] = data_split_stratify
            
            # Add optional parameters if provided
            if ignore_features:
                setup_params['ignore_features'] = [f.strip() for f in ignore_features.split(',')]
            
            if numeric_features:
                setup_params['numeric_features'] = [f.strip() for f in numeric_features.split(',')]
            
            if categorical_features:
                setup_params['categorical_features'] = [f.strip() for f in categorical_features.split(',')]
            
            if date_features:
                setup_params['date_features'] = [f.strip() for f in date_features.split(',')]
            
            if normalize is not None:
                setup_params['normalize'] = normalize
                
            if transformation is not None:
                setup_params['transformation'] = transformation
                
            if pca is not None:
                setup_params['pca'] = pca
                
            if pca_components is not None:
                setup_params['pca_components'] = pca_components
            
            # Check GPU availability and cuml installation if use_gpu is requested
            if use_gpu:
                try:
                    import cuml
                    logging.info("RAPIDS cuML is available for GPU acceleration")
                except ImportError:
                    logging.warning("'cuml' is not installed but use_gpu=True was specified.")
                    if ignore_gpu_errors:
                        logging.warning("Running on CPU instead. To use GPU, install cuml: pip install cuml")
                        use_gpu = False
                        setup_params['use_gpu'] = False
                    else:
                        raise ImportError("GPU acceleration requested but cuml is not installed. Install with: pip install cuml")
            
            # Set up the experiment with error handling for parameter compatibility
            logging.info("Setting up PyCaret experiment")
            logging.info(f"Setup parameters: {setup_params}")
            
            # Try to create setup with default parameters first
            try:
                s = setup(**setup_params)
                logging.info("PyCaret setup completed successfully")
            except TypeError as e:
                logging.warning(f"Setup error: {str(e)}")
                if "unexpected keyword argument" in str(e):
                    # Handle incompatible parameters by removing them and retrying
                    error_param = str(e).split("argument ")[-1].split("'")[1] if "'" in str(e) else str(e).split("argument ")[-1].strip()
                    logging.warning(f"Incompatible parameter detected: {error_param}")
                    logging.warning(f"Removing parameter and retrying setup")
                    
                    if error_param in setup_params:
                        del setup_params[error_param]
                        try:
                            s = setup(**setup_params)
                            logging.info("PyCaret setup completed successfully after parameter adjustment")
                        except Exception as inner_e:
                            logging.error(f"Setup failed even after removing parameter: {str(inner_e)}")
                            # Try a minimal setup as last resort
                            minimal_params = {
                                'data': data,
                                'target': target_column,
                                'session_id': session_id
                            }
                            logging.warning(f"Attempting minimal setup with just: {minimal_params.keys()}")
                            s = setup(**minimal_params)
                            logging.info("PyCaret setup completed with minimal parameters")
                    else:
                        raise
                else:
                    raise
            except Exception as e:
                logging.error(f"Unexpected setup error: {str(e)}")
                # Try a minimal setup as last resort
                minimal_params = {
                    'data': data,
                    'target': target_column,
                    'session_id': session_id
                }
                logging.warning(f"Attempting minimal setup with just: {minimal_params.keys()}")
                s = setup(**minimal_params)
                logging.info("PyCaret setup completed with minimal parameters")
            
            # Prepare parameters for compare_models
            compare_params = {'n_select': n_select}
            
            if include_models:
                compare_params['include'] = [m.strip() for m in include_models.split(',')]
                
            if exclude_models:
                compare_params['exclude'] = [m.strip() for m in exclude_models.split(',')]
            
            # Compare baseline models with compatibility handling
            logging.info(f"Comparing models (selecting top {n_select} models)")
            try:
                best = compare_models(**compare_params)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Handle incompatible parameters
                    error_param = str(e).split("argument ")[-1].split("'")[1] if "'" in str(e) else str(e).split("argument ")[-1].strip()
                    logging.warning(f"Incompatible parameter detected in compare_models: {error_param}")
                    logging.warning(f"Removing parameter and retrying")
                    
                    if error_param in compare_params:
                        del compare_params[error_param]
                        best = compare_models(**compare_params)
                    else:
                        raise
                else:
                    raise
            except Exception as e:
                logging.error(f"Error in compare_models: {str(e)}")
                # Try with minimal parameters
                logging.warning("Attempting compare_models with minimal parameters")
                best = compare_models()
            
            # If only one model is returned, wrap it in a list
            if not isinstance(best, list):
                best = [best]
                logging.info("Only one model was selected")
            
            logging.info(f"Top {len(best)} models selected")
            
            # Initialize list to track created files
            created_files = []
            
            # Compare models results
            compare_results = pull()
            compare_results_path = os.path.join(output_dir, 'model_comparison_results.csv')
            compare_results.to_csv(compare_results_path, index=True)
            created_files.append(('Model Comparison Results', compare_results_path))
            logging.info(f"Saved model comparison results to {compare_results_path}")
            
            # Define a function to create plots for a model
            def create_plots_for_model(model, plots_dir, created_files):
                # Store the original working directory
                original_dir = os.getcwd()
                
                try:
                    # Change to the plots directory before creating plots
                    os.chdir(plots_dir)
                    logging.info(f"Changed working directory to {plots_dir}")
                    
                    # Regression-specific plot types
                    plot_types = [
                        'residuals', 'error', 'cooks', 'learning',
                        'vc', 'feature', 'feature_all', 'parameter']
                    
                    model_plots = []
                    
                    # Try to create each plot type
                    for plot_type in plot_types:
                        try:
                            logging.info(f"Attempting to create {plot_type} plot in {plots_dir}")
                            
                            # Try to create the plot with save=True
                            plot_model(model, plot=plot_type, save=True)
                            
                            # Check if file was created
                            expected_filename = f"{plot_type}.png"
                            if os.path.exists(expected_filename):
                                full_path = os.path.join(plots_dir, expected_filename)
                                model_plots.append((f'Plot: {plot_type}', full_path))
                                logging.info(f"Successfully created {plot_type} plot at {full_path}")
                            else:
                                logging.warning(f"Plot {plot_type} was not found after creation")
                            
                        except Exception as e:
                            logging.warning(f"Failed to create {plot_type} plot: {str(e)}")
                    
                    return model_plots
                
                finally:
                    # Always restore the original working directory
                    os.chdir(original_dir)
                    logging.info(f"Restored working directory to {original_dir}")
            
            # Function to generate model interpretations
            def create_interpretations_for_model(model, plots_dir, created_files):
                # Store the original working directory
                original_dir = os.getcwd()
                
                try:
                    # Change to the plots directory before creating interpretations
                    os.chdir(plots_dir)
                    logging.info(f"Changed working directory to {plots_dir}")
                    
                    model_interpretations = []
                    
                    # Try to create summary interpretation
                    try:
                        logging.info("Attempting to create model interpretation summary")
                        interpret_model(model, plot='summary', save=True)
                        
                        # Check if file was created
                        expected_filename = "SHAP Summary.png"
                        if os.path.exists(expected_filename):
                            full_path = os.path.join(plots_dir, expected_filename)
                            model_interpretations.append(('Interpretation: summary', full_path))
                            logging.info(f"Successfully created interpretation summary at {full_path}")
                        else:
                            logging.warning("Interpretation summary file was not found after creation")
                            
                    except Exception as e:
                        logging.warning(f"Failed to create summary interpretation: {str(e)}")
                    
                    # Try to create correlation interpretation
                    try:
                        logging.info("Attempting to create model interpretation correlation")
                        interpret_model(model, plot='correlation', save=True)
                        
                        # Check if file was created
                        expected_filename = "SHAP Correlation.png"
                        if os.path.exists(expected_filename):
                            full_path = os.path.join(plots_dir, expected_filename)
                            model_interpretations.append(('Interpretation: correlation', full_path))
                            logging.info(f"Successfully created interpretation correlation at {full_path}")
                        else:
                            logging.warning("Interpretation correlation file was not found after creation")
                            
                    except Exception as e:
                        logging.warning(f"Failed to create correlation interpretation: {str(e)}")
                    
                    return model_interpretations
                
                finally:
                    # Always restore the original working directory
                    os.chdir(original_dir)
                    logging.info(f"Restored working directory to {original_dir}")
            
            # Function to evaluate model on test data
            def evaluate_model_on_test_data(model, output_path, model_name):
                try:
                    logging.info(f"Evaluating {model_name} on test data using PyCaret's inherent workflow")
                    
                    # Use PyCaret's inherent workflow for holdout predictions
                    # This uses the test split that was created during setup()
                    holdout_pred = predict_model(model)
                    
                    # Save the predictions
                    holdout_pred.to_csv(output_path, index=False)
                    logging.info(f"Saved {model_name} test predictions to {output_path}")
                    return output_path
                except Exception as e:
                    logging.error(f"Error evaluating {model_name} on test data: {str(e)}")
                    return None
            
            # Create a top-level models directory
            models_dir = os.path.join(output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Dictionary to track all individual model paths
            individual_model_paths = {}
            model_test_predictions = {}
            
            # Tune and save each model individually
            tuned_models = []
            for i, model in enumerate(best):
                model_index = i + 1
                model_name = f"tuned_model_{model_index}"
                logging.info(f"Tuning model {model_index}/{len(best)}")
                
                try:
                    # Create a dedicated directory for this model
                    model_dir = os.path.join(models_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Tune the model
                    tuned_model = tune_model(model)
                    tuned_models.append(tuned_model)
                    
                    # Save tuned model results
                    tuned_results = pull()
                    tuned_results_path = os.path.join(model_dir, f'{model_name}_results.csv')
                    tuned_results.to_csv(tuned_results_path, index=True)
                    created_files.append((f'{model_name} Results', tuned_results_path))
                    logging.info(f"Saved {model_name} results to {tuned_results_path}")
                    
                    # Save the model itself
                    model_path = os.path.join(model_dir, f'{model_name}')
                    try:
                        save_model(tuned_model, model_path)
                        created_files.append((f'{model_name} Saved Model', model_path))
                        individual_model_paths[model_name] = model_path
                        logging.info(f"Saved {model_name} to {model_path}")
                    except Exception as e:
                        logging.error(f"Error saving {model_name}: {str(e)}")
                    
                    # Create plots directory for this model
                    plots_dir = os.path.join(model_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Generate plots for this model
                    logging.info(f"Generating plots for {model_name}")
                    model_plots = create_plots_for_model(tuned_model, plots_dir, created_files)
                    created_files.extend(model_plots)
                    
                    # Generate interpretations for this model
                    logging.info(f"Generating interpretations for {model_name}")
                    model_interpretations = create_interpretations_for_model(tuned_model, plots_dir, created_files)
                    created_files.extend(model_interpretations)
                    
                    # Evaluate on holdout set using PyCaret's inherent workflow
                    test_pred_path = os.path.join(model_dir, f'independent_eval_results_{model_name}.csv')
                    test_result = evaluate_model_on_test_data(tuned_model, test_pred_path, model_name)
                    if test_result:
                        created_files.append((f'{model_name} Test Predictions', test_result))
                        model_test_predictions[model_name] = test_result
                    
                except Exception as e:
                    logging.error(f"Error processing {model_name}: {str(e)}")
            
            # Create a blended model if we have multiple tuned models
            blended_model = None
            
            if len(tuned_models) > 1:
                logging.info("Creating blended model from tuned models")
                try:
                    # Create dedicated directory for blended model
                    blended_dir = os.path.join(models_dir, "blended_model")
                    os.makedirs(blended_dir, exist_ok=True)
                    
                    # Create the blended model
                    blended_model = blend_models(tuned_models)
                    
                    # Save blended model results
                    blended_results = pull()
                    blended_results_path = os.path.join(blended_dir, 'blended_model_results.csv')
                    blended_results.to_csv(blended_results_path, index=True)
                    created_files.append(('Blended Model Results', blended_results_path))
                    logging.info(f"Saved blended model results to {blended_results_path}")
                    
                    # Save the blended model
                    blended_model_path = os.path.join(blended_dir, 'blended_model')
                    try:
                        save_model(blended_model, blended_model_path)
                        created_files.append(('Blended Model Saved Model', blended_model_path))
                        individual_model_paths["blended_model"] = blended_model_path
                        logging.info(f"Saved blended model to {blended_model_path}")
                    except Exception as e:
                        logging.error(f"Error saving blended model: {str(e)}")
                    
                    # Create plots directory for blended model
                    plots_dir = os.path.join(blended_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Generate plots for blended model
                    logging.info("Generating plots for blended model")
                    blended_plots = create_plots_for_model(blended_model, plots_dir, created_files)
                    created_files.extend(blended_plots)
                    
                    # Generate interpretations for blended model
                    logging.info("Generating interpretations for blended model")
                    blended_interpretations = create_interpretations_for_model(blended_model, plots_dir, created_files)
                    created_files.extend(blended_interpretations)
                    
                    # Evaluate on holdout set using PyCaret's inherent workflow
                    blended_test_pred_path = os.path.join(blended_dir, 'independent_eval_results_blended_model.csv')
                    test_result = evaluate_model_on_test_data(blended_model, blended_test_pred_path, "blended_model")
                    if test_result:
                        created_files.append(('Blended Model Test Predictions', test_result))
                        model_test_predictions["blended_model"] = test_result
                    
                    final_model = blended_model
                except Exception as e:
                    logging.error(f"Error creating blended model: {str(e)}")
                    logging.info("Using the best tuned model as the final model")
                    final_model = tuned_models[0]
            else:
                logging.info("Only one model available, using it as the final model")
                final_model = tuned_models[0] if tuned_models else best[0]
            
            # Initialize final model path
            final_model_path = None
            
            # Save the final model to the main output directory as well
            final_model_path = os.path.join(output_dir, f'{experiment_name}_final_model')
            try:
                save_model(final_model, final_model_path)
                created_files.append(('Final Model', final_model_path))
                logging.info(f"Saved final model to {final_model_path}")
            except Exception as e:
                logging.error(f"Error saving final model: {str(e)}")
                final_model_path = None
            
            # Initialize summary path
            summary_path = None
            
            # Create summary report with links to all models and evaluations
            try:
                summary_path = os.path.join(output_dir, "model_summary.csv")
                summary_data = []
                
                # Add individual models to summary
                for i, model in enumerate(tuned_models):
                    model_name = f"tuned_model_{i+1}"
                    model_path = individual_model_paths.get(model_name, "Not saved")
                    test_pred_path = model_test_predictions.get(model_name, "Not available")
                    
                    summary_data.append({
                        "Model": model_name,
                        "Type": "Tuned Individual Model",
                        "Model Path": model_path,
                        "Test Predictions": test_pred_path
                    })
                
                # Add blended model to summary if it exists
                if blended_model is not None:
                    model_path = individual_model_paths.get("blended_model", "Not saved")
                    test_pred_path = model_test_predictions.get("blended_model", "Not available")
                    
                    summary_data.append({
                        "Model": "blended_model",
                        "Type": "Blended Model",
                        "Model Path": model_path,
                        "Test Predictions": test_pred_path
                    })
                
                # Write summary to CSV
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(summary_path, index=False)
                created_files.append(('Model Summary', summary_path))
                logging.info(f"Created model summary at {summary_path}")
            except Exception as e:
                logging.error(f"Error creating model summary: {str(e)}")
                summary_path = None

            logging.info("Regression modeling completed successfully")

            return {
                "status": "success",
                "experiment_name": experiment_name,
                "input_path": input_path,
                "output_dir": output_dir,
                "final_model_path": final_model_path,
                "individual_model_paths": individual_model_paths,
                "model_test_predictions": model_test_predictions,
                "model_summary": summary_path,
                "log_file": log_file,
                "created_files": created_files
            }

        except Exception as e:
            logging.error(f"Error in PyCaret regression: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "input_path": input_path,
                "output_dir": output_dir
            }

#Image Classification Model Training and Inference Tool

class MedicalImageDataset(Dataset):
    """Dataset class for medical images that works with both ResNet and Inception V3 models."""
    
    def __init__(self, image_dir, labels_file=None, transform=None, is_test=False, image_size=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing the images.
            labels_file (str, optional): Path to CSV file with image names and labels.
            transform (callable, optional): Transform to apply to the images.
            is_test (bool, optional): Whether this is a test dataset without labels.
            image_size (tuple, optional): Size for fallback images when loading fails.
                                         Default: (224, 224) for ResNet, (299, 299) for Inception V3.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        # Default image sizes based on architecture
        if image_size is None:
            # Default to ResNet size as it's more common
            self.image_size = (224, 224)
        else:
            self.image_size = image_size
        
        if not is_test and labels_file is not None:
            self.labels_df = pd.read_csv(labels_file)
            
            # Handle both naming conventions (filename and image_name)
            if 'filename' in self.labels_df.columns:
                self.filename_col = 'filename'
            elif 'image_name' in self.labels_df.columns:
                self.filename_col = 'image_name'
            else:
                raise ValueError("CSV file must contain either 'filename' or 'image_name' column")
            
            # Ensure filenames don't have directory paths
            self.labels_df[self.filename_col] = self.labels_df[self.filename_col].apply(
                lambda x: os.path.basename(x) if isinstance(x, str) else x
            )
        elif is_test:
            # For test set without labels, just list all images in the directory
            self.image_files = [f for f in os.listdir(image_dir) 
                            if os.path.isfile(os.path.join(image_dir, f)) and 
                            f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
    
    def __len__(self):
        if self.is_test:
            return len(self.image_files)
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.is_test:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logging.warning(f"Error opening image {img_path}: {e}")
                image = Image.new('RGB', self.image_size, color='black')
                
            if self.transform:
                image = self.transform(image)
                
            return image, img_name  # Return filename for prediction output
        else:
            img_name = self.labels_df.iloc[idx][self.filename_col]
            img_path = os.path.join(self.image_dir, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                # If image can't be opened, create a black image as placeholder
                logging.warning(f"Error opening image {img_path}: {e}")
                image = Image.new('RGB', self.image_size, color='black')
                
            label = self.labels_df.iloc[idx]['label'] 
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
               
class PyTorchResNetTrainingTool(Tool):
    name = "pytorch_resnet_training"
    description = """
    This tool trains a ResNet model using PyTorch for medical image classification.
    It can train from scratch or fine-tune a pre-trained model, and includes validation metrics.
    """
    
    inputs = {
        "data_dir": {
            "type": "string",
            "description": "Directory containing dataset with training and validation folders"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where the trained model and results will be saved"
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes for classification"
        },
        "model_type": {
            "type": "string",
            "description": "ResNet model type: resnet18, resnet34, resnet50, resnet101, or resnet152",
            "required": False,
            "nullable": True
        },
        "num_epochs": {
            "type": "integer",
            "description": "Number of training epochs",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for training",
            "required": False,
            "nullable": True
        },
        "pretrained": {
            "type": "boolean",
            "description": "Whether to use pretrained weights",
            "required": False,
            "nullable": True
        },
        "early_stopping": {
            "type": "boolean", 
            "description": "Whether to use early stopping",
            "required": False,
            "nullable": True
        },
        "patience": {
            "type": "integer",
            "description": "Number of epochs to wait before early stopping",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        data_dir: str,
        output_dir: str,
        num_classes: int,
        model_type: Optional[str] = "resnet50",
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 16,
        pretrained: Optional[bool] = True,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 5
    ):
        """
        Train a ResNet model for medical image classification.
        
        Args:
            data_dir: Directory containing dataset with train and val subdirectories
            output_dir: Directory to save model and results
            num_classes: Number of classes for classification
            model_type: ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            pretrained: Whether to use pretrained weights
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary with training results and model paths
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "training.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Log configuration
            logging.info(f"Starting ResNet training with configuration:")
            logging.info(f"Model type: {model_type}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Pretrained: {pretrained}")
            logging.info(f"Early stopping: {early_stopping}")
            if early_stopping:
                logging.info(f"Patience: {patience}")
            logging.info(f"Data directory: {data_dir}")
            
            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
            # Define transformations
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Setup directories
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            test_dir = os.path.join(data_dir, 'test')
            
            if not os.path.exists(train_dir):
                raise ValueError(f"Training directory not found: {train_dir}")
            
            # Check if validation directory exists, otherwise use a split from training
            use_train_val_split = not os.path.exists(val_dir)
            
            # Check if data directory has a labels.csv file or if images are in class subdirectories
            train_labels_file = os.path.join(data_dir, 'labels.csv')
            use_csv_labels = os.path.exists(train_labels_file)
            
            # Create datasets based on the data organization
            if use_csv_labels:
                # Using CSV file with image paths and labels
                logging.info(f"Using labels from CSV file: {train_labels_file}")
                train_dataset = MedicalImageDataset(
                    image_dir=train_dir,
                    labels_file=train_labels_file,
                    transform=train_transform
                )
                
                if not use_train_val_split:
                    val_labels_file = os.path.join(data_dir, 'val_labels.csv')
                    if not os.path.exists(val_labels_file):
                        val_labels_file = train_labels_file  # Fallback to same labels file
                    
                    val_dataset = MedicalImageDataset(
                        image_dir=val_dir,
                        labels_file=val_labels_file,
                        transform=val_transform
                    )
            else:
                # Using directory structure (each class in its own subdirectory)
                logging.info("Using directory structure for class labels")
                train_dataset = datasets.ImageFolder(
                    root=train_dir,
                    transform=train_transform
                )
                
                if not use_train_val_split:
                    val_dataset = datasets.ImageFolder(
                        root=val_dir,
                        transform=val_transform
                    )
            
            # Split training data if no validation directory
            if use_train_val_split:
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size]
                )
            
            logging.info(f"Training dataset size: {len(train_dataset)}")
            logging.info(f"Validation dataset size: {len(val_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Create model
            logging.info(f"Creating {model_type} model...")
            
            # Dictionary of available ResNet models and their weights
            resnet_models = {
                'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
                'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
                'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
                'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
                'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1)
            }
            
            if model_type not in resnet_models:
                logging.warning(f"Invalid model type: {model_type}. Using resnet50 instead.")
                model_type = 'resnet50'
            
            # Get model function and weights    
            model_fn, weights = resnet_models[model_type]
            
            # Create model with or without pretrained weights
            if pretrained:
                model = model_fn(weights=weights)
                logging.info(f"Using pretrained weights from ImageNet")
            else:
                model = model_fn(weights=None)
                logging.info(f"Initializing model with random weights")
            
            # Modify the last layer for our number of classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
            # Move model to device
            model = model.to(device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=2, factor=0.5
            )
            
            # Track best model
            best_val_loss = float('inf')
            best_val_acc = 0.0
            patience_counter = 0
            best_epoch = 0
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'learning_rates': []
            }
            
            # Training loop
            logging.info("Starting training...")
            start_time = time.time()
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                epoch_train_loss = train_loss / train_total
                epoch_train_acc = 100 * train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = 100 * val_correct / val_total
                
                # Update learning rate
                scheduler.step(epoch_val_loss)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Save history
                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_val_loss)
                history['train_acc'].append(epoch_train_acc)
                history['val_acc'].append(epoch_val_acc)
                history['learning_rates'].append(current_lr)
                
                # Log progress
                logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Train Acc: {epoch_train_acc:.2f}%, "
                           f"Val Loss: {epoch_val_loss:.4f}, "
                           f"Val Acc: {epoch_val_acc:.2f}%, "
                           f"LR: {current_lr:.6f}")
                
                # Check for improvement
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_val_loss = epoch_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'accuracy': best_val_acc
                    }, best_model_path)
                    logging.info(f"Saved best model with accuracy: {best_val_acc:.2f}%")
                else:
                    patience_counter += 1
                
                # Early stopping if enabled
                if early_stopping and patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Calculate training time
            total_time = time.time() - start_time
            logging.info(f"Training completed in {total_time:.2f} seconds")
            
            # Save final model
            final_model_path = os.path.join(output_dir, "final_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc
            }, final_model_path)
            
            # Variables to track test results
            test_acc = None
            metrics_path = None
            cm_path = None
            
            # If test directory exists, evaluate on test set using best model
            if os.path.exists(test_dir):
                logging.info("Evaluating model on test set...")
                
                # Load best model for testing
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded best model from epoch {checkpoint['epoch']} for testing")
                
                # Create test dataset and dataloader
                if use_csv_labels:
                    test_labels_file = os.path.join(data_dir, 'test_labels.csv')
                    if not os.path.exists(test_labels_file):
                        test_labels_file = train_labels_file  # Fallback to same labels file
                    
                    test_dataset = MedicalImageDataset(
                        image_dir=test_dir,
                        labels_file=test_labels_file,
                        transform=val_transform  # Use the same transform as validation
                    )
                else:
                    test_dataset = datasets.ImageFolder(
                        root=test_dir,
                        transform=val_transform
                    )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )
                
                # Evaluate on test set
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        test_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                        
                        # Store predictions and labels for metrics
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                test_acc = 100 * test_correct / test_total
                logging.info(f"Test accuracy: {test_acc:.2f}%")
                
                # Calculate and log detailed metrics
                metrics = {}
                metrics['accuracy'] = accuracy_score(all_labels, all_preds)
                metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                
                # Add per-class metrics if there are multiple classes
                if num_classes > 1:
                    metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    metrics['f1_per_class'] = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                
                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "test_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Generate confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
            
            # Save training history
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Create plots
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')
            
            plt.subplot(1, 3, 2)
            plt.plot(history['train_acc'], label='Training Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Curves')
            
            plt.subplot(1, 3, 3)
            plt.plot(history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            
            plt.tight_layout()
            plots_path = os.path.join(output_dir, "training_plots.png")
            plt.savefig(plots_path)
            plt.close()
            
            # Save model configuration
            config = {
                'model_type': model_type,
                'num_classes': num_classes,
                'image_size': 224,
                'pretrained': pretrained,
                'early_stopping': early_stopping,
                'patience': patience if early_stopping else None,
                'best_epoch': best_epoch,
                'best_accuracy': best_val_acc,
                'training_epochs': epoch + 1,
                'early_stopped': early_stopping and patience_counter >= patience,
                'batch_size': batch_size
            }
            
            config_path = os.path.join(output_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Return results
            return {
                "status": "success",
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                "config_path": config_path,
                "plots_path": plots_path,
                "history_path": history_path,
                "test_metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "best_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "training_time_seconds": total_time,
                "epochs_completed": epoch + 1,
                "early_stopped": early_stopping and patience_counter >= patience,
                "early_stopping_used": early_stopping
            }
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "data_dir": data_dir,
                "output_dir": output_dir
            }

class PyTorchResNetInferenceTool(Tool):
    name = "pytorch_resnet_inference"
    description = """
    This tool uses a trained PyTorch ResNet model to perform inference on new images.
    It can also calculate performance metrics if ground truth labels are provided.
    """
    
    inputs = {
        "image_dir": {
            "type": "string",
            "description": "Directory containing images for inference"
        },
        "model_path": {
            "type": "string",
            "description": "Path to the trained model file (.pt format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where prediction results will be saved"
        },
        "config_path": {
            "type": "string",
            "description": "Path to model configuration JSON file (optional)",
            "required": False,
            "nullable": True
        },
        "ground_truth_file": {
            "type": "string",
            "description": "CSV file with image filenames and ground truth labels for evaluation (optional)",
            "required": False,
            "nullable": True
        },
        "class_names": {
            "type": "array",
            "description": "List of class names corresponding to model output indices (optional)",
            "required": False,
            "nullable": True
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes (needed if not provided in config file)",
            "required": False,
            "nullable": True
        },
        "model_type": {
            "type": "string",
            "description": "ResNet model type",
            "required": True,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for inference",
            "required": False,
            "nullable": True
        },
        "case_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains case/file identifiers",
            "required": False,
            "nullable": True
        },
        "label_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains the labels",
            "required": False,
            "nullable": True
        },
        "file_extension": {
            "type": "string",
            "description": "File extension to add to case IDs if they don't already have one",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        image_dir: str,
        model_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        model_type: Optional[str] = None,
        batch_size: Optional[int] = 32,
        case_column: Optional[str] = "case",
        label_column: Optional[str] = "label",
        file_extension: Optional[str] = ".png"
    ):
        """
        Run inference using a trained PyTorch ResNet model on new images.
        
        Args:
            image_dir: Directory containing images for inference
            model_path: Path to the trained model file (.pt format)
            output_dir: Directory to save prediction outputs
            config_path: Path to model configuration JSON file (optional)
            ground_truth_file: CSV file with filenames and ground truth labels (optional)
            class_names: List of class names corresponding to model output indices
            num_classes: Number of classes (needed if not in config file)
            model_type: ResNet model type (will use config value if not specified)
            batch_size: Batch size for inference
            case_column: Name of the column in ground truth file with case/file IDs
            label_column: Name of the column in ground truth file with labels
            file_extension: File extension to add to case IDs if needed
            
        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "inference.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Load model configuration if provided
            model_config = {}
            if config_path and os.path.exists(config_path):
                logging.info(f"Loading model configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        model_config = json.load(f)
                    logging.info(f"Model configuration loaded: {model_config}")
                except Exception as e:
                    logging.warning(f"Error loading model configuration: {str(e)}. Will use provided parameters.")
            
            # Set parameters, with priority to explicit parameters over config values
            if model_type is None:
                model_type = model_config.get('model_type', 'resnet50')
            
            if num_classes is None:
                num_classes = model_config.get('num_classes')
                if num_classes is None:
                    if class_names:
                        num_classes = len(class_names)
                    else:
                        raise ValueError("Number of classes must be provided either directly, in config file, or through class_names")
            
            image_size = model_config.get('image_size', 224)
            
            # Log inference settings
            logging.info(f"Starting ResNet inference with settings:")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Model type: {model_type}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Image directory: {image_dir}")
            logging.info(f"Has ground truth: {ground_truth_file is not None}")
            
            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
            # Define image transformation
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Create model
            logging.info(f"Creating {model_type} model architecture...")
            
            # Dictionary of available ResNet models
            resnet_models = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152
            }
            
            if model_type not in resnet_models:
                logging.warning(f"Invalid model type: {model_type}. Using resnet50 instead.")
                model_type = 'resnet50'
                
            model_fn = resnet_models[model_type]
            model = model_fn(pretrained=False)
            
            # Modify the last layer for our number of classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
            # Load trained weights
            logging.info(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    logging.info("Model loaded directly from state dict")
            except Exception as e:
                logging.error(f"Error loading model weights: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model weights: {str(e)}",
                    "model_path": model_path
                }
            
            # Move model to device and set to evaluation mode
            model = model.to(device)
            model.eval()
            
            # Initialize variables for ground truth processing
            ground_truth = {}
            filename_to_case = {}
            
            # Load ground truth labels if provided
            if ground_truth_file and os.path.exists(ground_truth_file):
                logging.info(f"Loading ground truth data from {ground_truth_file}")
                try:
                    gt_df = pd.read_csv(ground_truth_file)
                    
                    # Check required columns
                    if case_column not in gt_df.columns or label_column not in gt_df.columns:
                        raise ValueError(f"Ground truth file must contain '{case_column}' and '{label_column}' columns")
                    
                    # Create mapping of case ID to filename
                    case_to_filename = {}
                    
                    # Create dictionary mapping filename to label
                    for _, row in gt_df.iterrows():
                        # Get case ID and convert to string
                        case_id = str(row[case_column])
                        
                        # Add file extension if needed
                        if not any(case_id.lower().endswith(ext) for ext in 
                                  ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                            filename = f"{case_id}{file_extension}"
                        else:
                            filename = case_id
                        
                        # Store the mapping and ground truth
                        case_to_filename[case_id] = filename
                        ground_truth[filename] = int(row[label_column])
                        
                    logging.info(f"Loaded {len(ground_truth)} ground truth labels")
                    
                    # Create a reverse mapping from filename to case ID for results
                    filename_to_case = {v: k for k, v in case_to_filename.items()}
                    
                except Exception as e:
                    logging.error(f"Error loading ground truth file: {str(e)}")
                    logging.warning("Will continue without ground truth evaluation")
                    ground_truth = {}
                    filename_to_case = {}
            
            # Get list of all image files
            image_files = [f for f in os.listdir(image_dir) 
                         if os.path.isfile(os.path.join(image_dir, f)) and 
                         f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
            
            if not image_files:
                logging.error(f"No valid image files found in {image_dir}")
                return {
                    "status": "error",
                    "error_message": f"No valid image files found in directory",
                    "image_dir": image_dir
                }
            
            logging.info(f"Found {len(image_files)} images for inference")
            
            # Process images in batches for efficiency
            all_predictions = []
            batch_idx = 0
            start_time = time.time()
            
            with torch.no_grad():  # No gradient computation needed for inference
                for batch_start in range(0, len(image_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(image_files))
                    batch_files = image_files[batch_start:batch_end]
                    batch_imgs = []
                    valid_files = []
                    
                    # Load and preprocess images in this batch
                    for img_file in batch_files:
                        try:
                            img_path = os.path.join(image_dir, img_file)
                            image = Image.open(img_path).convert('RGB')
                            image_tensor = transform(image)
                            batch_imgs.append(image_tensor)
                            valid_files.append(img_file)
                        except Exception as e:
                            logging.warning(f"Error processing image {img_file}: {str(e)}")
                            all_predictions.append({
                                'filename': img_file,
                                'error': str(e)
                            })
                    
                    if not batch_imgs:  # Skip if no valid images in batch
                        continue
                    
                    # Stack tensors into batch
                    batch_tensor = torch.stack(batch_imgs).to(device)
                    
                    # Run inference
                    outputs = model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Process results
                    batch_probs = probabilities.cpu().numpy()
                    batch_preds = np.argmax(batch_probs, axis=1)
                    
                    # Store predictions
                    for i, (filename, pred_class, probs) in enumerate(zip(valid_files, batch_preds, batch_probs)):
                        # Get corresponding case ID if available
                        case_id = filename_to_case.get(filename, filename)
                        
                        result = {
                            'filename': filename,
                            'case_id': case_id,
                            'predicted_class': int(pred_class),
                            'confidence': float(probs[pred_class])
                        }
                        
                        # Add ground truth if available
                        if filename in ground_truth:
                            result['true_label'] = int(ground_truth[filename])
                            result['correct'] = result['predicted_class'] == result['true_label']
                        
                        # Add class name if available
                        if class_names and pred_class < len(class_names):
                            result['predicted_class_name'] = class_names[pred_class]
                            
                        # Add probabilities for each class
                        for class_idx, prob in enumerate(probs):
                            if class_names and class_idx < len(class_names):
                                result[f'prob_{class_names[class_idx]}'] = float(prob)
                            else:
                                result[f'prob_class_{class_idx}'] = float(prob)
                        
                        all_predictions.append(result)
                    
                    batch_idx += 1
                    if batch_idx % 10 == 0:
                        logging.info(f"Processed {batch_end}/{len(image_files)} images")
            
            processing_time = time.time() - start_time
            logging.info(f"Inference completed in {processing_time:.2f} seconds")
            
            # Create results dataframe
            predictions_df = pd.DataFrame(all_predictions)
            
            # Save predictions to CSV
            csv_path = os.path.join(output_dir, "predictions.csv")
            predictions_df.to_csv(csv_path, index=False)
            logging.info(f"Saved predictions to {csv_path}")
            
            # Calculate metrics if ground truth is available
            metrics = {}
            metrics_path = None
            cm_path = None
            roc_path = None
            
            if ground_truth and any('true_label' in p for p in all_predictions):
                logging.info("Calculating performance metrics...")
                
                # Filter predictions with ground truth
                valid_preds = [p for p in all_predictions if 'true_label' in p]
                
                # Extract true labels and predictions
                y_true = [p['true_label'] for p in valid_preds]
                y_pred = [p['predicted_class'] for p in valid_preds]
                
                # Get unique classes in sorted order
                unique_classes = sorted(list(set(y_true)))
                
                # Basic metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                
                try:
                    # Multi-class metrics
                    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    # Per-class metrics
                    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    
                    # Extract per-class metrics
                    for class_idx in unique_classes:
                        class_key = str(class_idx)
                        if class_key in class_report:
                            class_metrics = class_report[class_key]
                            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"class_{class_idx}"
                            
                            metrics[f'precision_{class_name}'] = class_metrics['precision']
                            metrics[f'recall_{class_name}'] = class_metrics['recall']
                            metrics[f'f1_{class_name}'] = class_metrics['f1-score']
                            
                            # Calculate sensitivity and specificity for each class
                            # Sensitivity = recall
                            metrics[f'sensitivity_{class_name}'] = class_metrics['recall']
                            
                            # Specificity = TN / (TN + FP)
                            y_true_binary = [1 if y == class_idx else 0 for y in y_true]
                            y_pred_binary = [1 if y == class_idx else 0 for y in y_pred]
                            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            metrics[f'specificity_{class_name}'] = specificity
                    
                    # Calculate AUC and ROC curves if we have probability outputs
                    if len(unique_classes) == 2:  # Binary classification
                        # Extract probabilities for positive class
                        class_idx = max(unique_classes)  # Assuming 1 is positive class in binary case
                        if class_names and class_idx < len(class_names):
                            prob_col = f'prob_{class_names[class_idx]}'
                        else:
                            prob_col = f'prob_class_{class_idx}'
                        
                        if prob_col in predictions_df.columns:
                            y_score = predictions_df.loc[predictions_df['true_label'].notna(), prob_col].values
                            y_true_for_auc = [y for i, y in enumerate(y_true)]
                            
                            # Calculate ROC curve and AUC
                            metrics['auc'] = roc_auc_score(y_true_for_auc, y_score)
                            fpr, tpr, _ = roc_curve(y_true_for_auc, y_score)
                            
                            # Plot ROC curve
                            plt.figure(figsize=(8, 8))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                    label=f'ROC curve (area = {metrics["auc"]:.3f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic')
                            plt.legend(loc="lower right")
                            
                            roc_path = os.path.join(output_dir, "roc_curve.png")
                            plt.savefig(roc_path)
                            plt.close()
                    elif len(unique_classes) > 2:  # Multi-class
                        # For multi-class, calculate one-vs-rest AUC for each class
                        for class_idx in unique_classes:
                            if class_names and class_idx < len(class_names):
                                prob_col = f'prob_{class_names[class_idx]}'
                                class_name = class_names[class_idx]
                            else:
                                prob_col = f'prob_class_{class_idx}'
                                class_name = f"class_{class_idx}"
                            
                            if prob_col in predictions_df.columns:
                                # Create binary labels for this class
                                y_true_binary = [1 if y == class_idx else 0 for y in y_true]
                                y_score = predictions_df.loc[predictions_df['true_label'].notna(), prob_col].values
                                
                                try:
                                    # Calculate AUC
                                    class_auc = roc_auc_score(y_true_binary, y_score)
                                    metrics[f'auc_{class_name}'] = class_auc
                                except Exception as e:
                                    logging.warning(f"Could not calculate AUC for {class_name}: {str(e)}")
                except Exception as e:
                    logging.warning(f"Error calculating some metrics: {str(e)}")
                
                # Log all metrics
                logging.info("Performance metrics:")
                for metric_name, metric_value in metrics.items():
                    logging.info(f"{metric_name}: {metric_value:.4f}")
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
                
                # Set up class labels for the plot
                if class_names:
                    labels = [class_names[i] if i < len(class_names) else f"class_{i}" for i in unique_classes]
                else:
                    labels = [f"class_{i}" for i in unique_classes]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                
                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Create a readable metrics summary as CSV
                metrics_summary = []
                for metric_name, metric_value in metrics.items():
                    metrics_summary.append({
                        'metric': metric_name,
                        'value': metric_value
                    })
                pd.DataFrame(metrics_summary).to_csv(
                    os.path.join(output_dir, "metrics_summary.csv"), index=False
                )
            
            # Return results
            return {
                "status": "success",
                "predictions_path": csv_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "roc_curve_path": roc_path,
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir,
                "num_images_processed": len(all_predictions),
                "num_images_with_errors": sum(1 for p in all_predictions if 'error' in p),
                "has_ground_truth": bool(ground_truth),
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir
            }

class PyTorchInceptionV3TrainingTool(Tool):
    name = "pytorch_inception_v3_training"
    description = """
    This tool trains an Inception V3 model using PyTorch for medical image classification.
    It can train from scratch or fine-tune a pre-trained model, and includes validation metrics.
    """
    
    inputs = {
        "data_dir": {
            "type": "string",
            "description": "Directory containing dataset with training and validation folders"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where the trained model and results will be saved"
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes for classification"
        },
        "num_epochs": {
            "type": "integer",
            "description": "Number of training epochs",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for training",
            "required": False,
            "nullable": True
        },
        "pretrained": {
            "type": "boolean",
            "description": "Whether to use pretrained weights",
            "required": False,
            "nullable": True
        },
        "early_stopping": {
            "type": "boolean", 
            "description": "Whether to use early stopping",
            "required": False,
            "nullable": True
        },
        "patience": {
            "type": "integer",
            "description": "Number of epochs to wait before early stopping",
            "required": False,
            "nullable": True
        },
        "aux_logits": {
            "type": "boolean",
            "description": "Whether to use auxiliary logits during training (Inception specific)",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        data_dir: str,
        output_dir: str,
        num_classes: int,
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 16,
        pretrained: Optional[bool] = True,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 5,
        aux_logits: Optional[bool] = True
    ):
        """
        Train an Inception V3 model for medical image classification.
        
        Args:
            data_dir: Directory containing dataset with train and val subdirectories
            output_dir: Directory to save model and results
            num_classes: Number of classes for classification
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            pretrained: Whether to use pretrained weights
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            aux_logits: Whether to use auxiliary logits during training (Inception specific)
            
        Returns:
            Dictionary with training results and model paths
        """
        try:
            # Define the image size for Inception V3
            inception_image_size = (299, 299)
            
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "training.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Log configuration
            logging.info(f"Starting Inception V3 training with configuration:")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Pretrained: {pretrained}")
            logging.info(f"Early stopping: {early_stopping}")
            logging.info(f"Auxiliary logits: {aux_logits}")
            if early_stopping:
                logging.info(f"Patience: {patience}")
            logging.info(f"Data directory: {data_dir}")
            
            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
            # Define transformations - Inception V3 requires 299x299 input
            train_transform = transforms.Compose([
                transforms.Resize(inception_image_size),  # Inception V3 requires 299x299 input
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize(inception_image_size),  # Inception V3 requires 299x299 input
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Setup directories
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            test_dir = os.path.join(data_dir, 'test')
            
            if not os.path.exists(train_dir):
                raise ValueError(f"Training directory not found: {train_dir}")
            
            # Check if validation directory exists, otherwise use a split from training
            use_train_val_split = not os.path.exists(val_dir)
            
            # Check if data directory has a labels.csv file or if images are in class subdirectories
            train_labels_file = os.path.join(data_dir, 'labels.csv')
            use_csv_labels = os.path.exists(train_labels_file)
            
            # Create datasets based on the data organization
            if use_csv_labels:
                # Using CSV file with image paths and labels
                logging.info(f"Using labels from CSV file: {train_labels_file}")
                train_dataset = MedicalImageDataset(
                    image_dir=train_dir,
                    labels_file=train_labels_file,
                    transform=train_transform
                )
                
                if not use_train_val_split:
                    val_labels_file = os.path.join(data_dir, 'val_labels.csv')
                    if not os.path.exists(val_labels_file):
                        val_labels_file = train_labels_file  # Fallback to same labels file
                    
                    val_dataset = MedicalImageDataset(
                        image_dir=val_dir,
                        labels_file=val_labels_file,
                        transform=val_transform
                    )
            else:
                # Using directory structure (each class in its own subdirectory)
                logging.info("Using directory structure for class labels")
                train_dataset = datasets.ImageFolder(
                    root=train_dir,
                    transform=train_transform
                )
                
                if not use_train_val_split:
                    val_dataset = datasets.ImageFolder(
                        root=val_dir,
                        transform=val_transform
                    )
            
            # Split training data if no validation directory
            if use_train_val_split:
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size]
                )
            
            logging.info(f"Training dataset size: {len(train_dataset)}")
            logging.info(f"Validation dataset size: {len(val_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Create Inception V3 model
            logging.info("Creating Inception V3 model...")
            
            # Initialize model with or without pretrained weights
            if pretrained:
                model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=aux_logits)
                logging.info("Using pretrained weights from ImageNet")
            else:
                model = models.inception_v3(weights=None, aux_logits=aux_logits)
                logging.info("Initializing model with random weights")
            
            # Modify the classifier for our number of classes
            # Inception V3 has two outputs if aux_logits=True
            # Replace the main classifier
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
            # Replace the auxiliary classifier if aux_logits is enabled
            if aux_logits:
                in_features_aux = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)
            
            # Move model to device
            model = model.to(device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=2, factor=0.5
            )
            
            # Track best model
            best_val_loss = float('inf')
            best_val_acc = 0.0
            best_epoch = 0
            patience_counter = 0
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'learning_rates': []
            }
            
            # Training loop
            logging.info("Starting training...")
            start_time = time.time()
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    # Inception v3 returns tuple (output, aux_output) when training if aux_logits=True
                    if aux_logits:
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        # During training, the auxiliary classifier's loss is weighted by 0.3
                        loss = loss1 + 0.3 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                epoch_train_loss = train_loss / train_total
                epoch_train_acc = 100 * train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        # During evaluation, only the main output is used
                        outputs = model(inputs)
                        
                        # Handle the case where model returns a tuple (happens during evaluation with aux_logits=True)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = 100 * val_correct / val_total
                
                # Update learning rate
                scheduler.step(epoch_val_loss)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Save history
                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_val_loss)
                history['train_acc'].append(epoch_train_acc)
                history['val_acc'].append(epoch_val_acc)
                history['learning_rates'].append(current_lr)
                
                # Log progress
                logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Train Acc: {epoch_train_acc:.2f}%, "
                           f"Val Loss: {epoch_val_loss:.4f}, "
                           f"Val Acc: {epoch_val_acc:.2f}%, "
                           f"LR: {current_lr:.6f}")
                
                # Check for improvement
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_val_loss = epoch_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'accuracy': best_val_acc
                    }, best_model_path)
                    logging.info(f"Saved best model with accuracy: {best_val_acc:.2f}%")
                else:
                    patience_counter += 1
                
                # Early stopping if enabled
                if early_stopping and patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Calculate training time
            total_time = time.time() - start_time
            logging.info(f"Training completed in {total_time:.2f} seconds")
            
            # Save final model
            final_model_path = os.path.join(output_dir, "final_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc
            }, final_model_path)
            
            # Variables to track test results
            test_acc = None
            metrics_path = None
            cm_path = None
            
            # If test directory exists, evaluate on test set using best model
            if os.path.exists(test_dir):
                logging.info("Evaluating model on test set...")
                
                # Load best model for testing
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded best model from epoch {checkpoint['epoch']} for testing")
                
                # Create test dataset and dataloader
                if use_csv_labels:
                    test_labels_file = os.path.join(data_dir, 'test_labels.csv')
                    if not os.path.exists(test_labels_file):
                        test_labels_file = train_labels_file  # Fallback to same labels file
                    
                    test_dataset = MedicalImageDataset(
                        image_dir=test_dir,
                        labels_file=test_labels_file,
                        transform=val_transform  # Use the same transform as validation
                    )
                else:
                    test_dataset = datasets.ImageFolder(
                        root=test_dir,
                        transform=val_transform
                    )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )
                
                # Evaluate on test set
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        
                        # Handle the case where model returns a tuple (happens during evaluation with aux_logits=True)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                            
                        loss = criterion(outputs, labels)
                        
                        test_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                        
                        # Store predictions and labels for metrics
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                test_acc = 100 * test_correct / test_total
                logging.info(f"Test accuracy: {test_acc:.2f}%")
                
                # Calculate and log detailed metrics
                metrics = {}
                metrics['accuracy'] = accuracy_score(all_labels, all_preds)
                metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                
                # Add per-class metrics if there are multiple classes
                if num_classes > 1:
                    metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    metrics['f1_per_class'] = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                
                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "test_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Generate confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
            
            # Save training history
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Create plots
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')
            
            plt.subplot(1, 3, 2)
            plt.plot(history['train_acc'], label='Training Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Curves')
            
            plt.subplot(1, 3, 3)
            plt.plot(history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            
            plt.tight_layout()
            plots_path = os.path.join(output_dir, "training_plots.png")
            plt.savefig(plots_path)
            plt.close()
            
            # Save model configuration
            config = {
                'model_type': 'inception_v3',
                'num_classes': num_classes,
                'image_size': 299,  # Inception V3 uses 299x299 images
                'pretrained': pretrained,
                'early_stopping': early_stopping,
                'patience': patience if early_stopping else None,
                'best_epoch': best_epoch,
                'best_accuracy': best_val_acc,
                'training_epochs': epoch + 1,
                'early_stopped': early_stopping and patience_counter >= patience,
                'batch_size': batch_size,
                'aux_logits': aux_logits
            }
            
            config_path = os.path.join(output_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Return results
            return {
                "status": "success",
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                "config_path": config_path,
                "plots_path": plots_path,
                "history_path": history_path,
                "test_metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "best_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "training_time_seconds": total_time,
                "epochs_completed": epoch + 1,
                "early_stopped": early_stopping and patience_counter >= patience,
                "early_stopping_used": early_stopping,
                "aux_logits_used": aux_logits
            }
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "data_dir": data_dir,
                "output_dir": output_dir
            }

class PyTorchInceptionV3InferenceTool(Tool):
    name = "pytorch_inception_v3_inference"
    description = """
    This tool uses a trained PyTorch Inception V3 model to perform inference on new images.
    It can also calculate performance metrics if ground truth labels are provided.
    """
    
    inputs = {
        "image_dir": {
            "type": "string",
            "description": "Directory containing images for inference"
        },
        "model_path": {
            "type": "string",
            "description": "Path to the trained model file (.pt format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where prediction results will be saved"
        },
        "config_path": {
            "type": "string",
            "description": "Path to model configuration JSON file (optional)",
            "required": False,
            "nullable": True
        },
        "ground_truth_file": {
            "type": "string",
            "description": "CSV file with image filenames and ground truth labels for evaluation (optional)",
            "required": False,
            "nullable": True
        },
        "class_names": {
            "type": "array",
            "description": "List of class names corresponding to model output indices (optional)",
            "required": False,
            "nullable": True
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes (needed if not provided in config file)",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for inference",
            "required": False,
            "nullable": True
        },
        "case_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains case/file identifiers",
            "required": False,
            "nullable": True
        },
        "label_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains the labels",
            "required": False,
            "nullable": True
        },
        "file_extension": {
            "type": "string",
            "description": "File extension to add to case IDs if they don't already have one",
            "required": False,
            "nullable": True
        },
        "aux_logits": {
            "type": "boolean",
            "description": "Whether the model was trained with auxiliary logits (Inception specific)",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        image_dir: str,
        model_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        batch_size: Optional[int] = 32,
        case_column: Optional[str] = "case",
        label_column: Optional[str] = "label",
        file_extension: Optional[str] = ".png",
        aux_logits: Optional[bool] = True
    ):
        """
        Run inference using a trained PyTorch Inception V3 model on new images.
        
        Args:
            image_dir: Directory containing images for inference
            model_path: Path to the trained model file (.pt format)
            output_dir: Directory to save prediction outputs
            config_path: Path to model configuration JSON file (optional)
            ground_truth_file: CSV file with filenames and ground truth labels (optional)
            class_names: List of class names corresponding to model output indices
            num_classes: Number of classes (needed if not in config file)
            batch_size: Batch size for inference
            case_column: Name of the column in ground truth file with case/file IDs
            label_column: Name of the column in ground truth file with labels
            file_extension: File extension to add to case IDs if needed
            aux_logits: Whether the model was trained with auxiliary logits
            
        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "inference.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Load model configuration if provided
            model_config = {}
            if config_path and os.path.exists(config_path):
                logging.info(f"Loading model configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        model_config = json.load(f)
                    logging.info(f"Model configuration loaded: {model_config}")
                except Exception as e:
                    logging.warning(f"Error loading model configuration: {str(e)}. Will use provided parameters.")
            
            # Set parameters, with priority to explicit parameters over config values
            model_type = 'inception_v3'  # Fixed for this tool
            
            if num_classes is None:
                num_classes = model_config.get('num_classes')
                if num_classes is None:
                    if class_names:
                        num_classes = len(class_names)
                    else:
                        raise ValueError("Number of classes must be provided either directly, in config file, or through class_names")
            
            # Inception V3 requires 299x299 images
            image_size = model_config.get('image_size', 299)
            
            # Get aux_logits from config if not provided
            if aux_logits is None:
                aux_logits = model_config.get('aux_logits', True)
            
            # Log inference settings
            logging.info(f"Starting Inception V3 inference with settings:")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Model type: {model_type}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Image directory: {image_dir}")
            logging.info(f"Aux logits: {aux_logits}")
            logging.info(f"Has ground truth: {ground_truth_file is not None}")
            
            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
            # Define image transformation (Inception V3 requires 299x299 input)
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Create model
            logging.info(f"Creating Inception V3 model architecture...")
            
            # Create Inception V3 model
            model = models.inception_v3(pretrained=False, aux_logits=aux_logits)
            
            # Modify the classifiers for our number of classes
            # Main classifier
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
            # Auxiliary classifier (if used)
            if aux_logits:
                in_features_aux = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)
            
            # Load trained weights
            logging.info(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    logging.info("Model loaded directly from state dict")
            except Exception as e:
                logging.error(f"Error loading model weights: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model weights: {str(e)}",
                    "model_path": model_path
                }
            
            # Move model to device and set to evaluation mode
            model = model.to(device)
            model.eval()
            
            # Initialize variables for ground truth processing
            ground_truth = {}
            filename_to_case = {}
            
            # Load ground truth labels if provided
            if ground_truth_file and os.path.exists(ground_truth_file):
                logging.info(f"Loading ground truth data from {ground_truth_file}")
                try:
                    gt_df = pd.read_csv(ground_truth_file)
                    
                    # Check required columns
                    if case_column not in gt_df.columns or label_column not in gt_df.columns:
                        raise ValueError(f"Ground truth file must contain '{case_column}' and '{label_column}' columns")
                    
                    # Create mapping of case ID to filename
                    case_to_filename = {}
                    
                    # Create dictionary mapping filename to label
                    for _, row in gt_df.iterrows():
                        # Get case ID and convert to string
                        case_id = str(row[case_column])
                        
                        # Add file extension if needed
                        if not any(case_id.lower().endswith(ext) for ext in 
                                  ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                            filename = f"{case_id}{file_extension}"
                        else:
                            filename = case_id
                        
                        # Store the mapping and ground truth
                        case_to_filename[case_id] = filename
                        ground_truth[filename] = int(row[label_column])
                        
                    logging.info(f"Loaded {len(ground_truth)} ground truth labels")
                    
                    # Create a reverse mapping from filename to case ID for results
                    filename_to_case = {v: k for k, v in case_to_filename.items()}
                    
                except Exception as e:
                    logging.error(f"Error loading ground truth file: {str(e)}")
                    logging.warning("Will continue without ground truth evaluation")
                    ground_truth = {}
                    filename_to_case = {}
            
            # Get list of all image files
            image_files = [f for f in os.listdir(image_dir) 
                         if os.path.isfile(os.path.join(image_dir, f)) and 
                         f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
            
            if not image_files:
                logging.error(f"No valid image files found in {image_dir}")
                return {
                    "status": "error",
                    "error_message": f"No valid image files found in directory",
                    "image_dir": image_dir
                }
            
            logging.info(f"Found {len(image_files)} images for inference")
            
            # Process images in batches for efficiency
            all_predictions = []
            batch_idx = 0
            start_time = time.time()
            
            with torch.no_grad():  # No gradient computation needed for inference
                for batch_start in range(0, len(image_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(image_files))
                    batch_files = image_files[batch_start:batch_end]
                    batch_imgs = []
                    valid_files = []
                    
                    # Load and preprocess images in this batch
                    for img_file in batch_files:
                        try:
                            img_path = os.path.join(image_dir, img_file)
                            image = Image.open(img_path).convert('RGB')
                            image_tensor = transform(image)
                            batch_imgs.append(image_tensor)
                            valid_files.append(img_file)
                        except Exception as e:
                            logging.warning(f"Error processing image {img_file}: {str(e)}")
                            all_predictions.append({
                                'filename': img_file,
                                'error': str(e)
                            })
                    
                    if not batch_imgs:  # Skip if no valid images in batch
                        continue
                    
                    # Stack tensors into batch
                    batch_tensor = torch.stack(batch_imgs).to(device)
                    
                    try:
                        # Run inference - need to handle the aux_logits
                        outputs = model(batch_tensor)
                        
                        # Inception V3 in eval mode returns a tuple if aux_logits=True
                        if isinstance(outputs, tuple):
                            # During evaluation, we only use the main output (index 0)
                            outputs = outputs[0]
                            
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # Process results
                        batch_probs = probabilities.cpu().numpy()
                        batch_preds = np.argmax(batch_probs, axis=1)
                        
                        # Store predictions
                        for i, (filename, pred_class, probs) in enumerate(zip(valid_files, batch_preds, batch_probs)):
                            # Get corresponding case ID if available
                            case_id = filename_to_case.get(filename, filename)
                            
                            result = {
                                'filename': filename,
                                'case_id': case_id,
                                'predicted_class': int(pred_class),
                                'confidence': float(probs[pred_class])
                            }
                            
                            # Add ground truth if available
                            if filename in ground_truth:
                                result['true_label'] = int(ground_truth[filename])
                                result['correct'] = result['predicted_class'] == result['true_label']
                            
                            # Add class name if available
                            if class_names and pred_class < len(class_names):
                                result['predicted_class_name'] = class_names[pred_class]
                                
                            # Add probabilities for each class
                            for class_idx, prob in enumerate(probs):
                                if class_names and class_idx < len(class_names):
                                    result[f'prob_{class_names[class_idx]}'] = float(prob)
                                else:
                                    result[f'prob_class_{class_idx}'] = float(prob)
                            
                            all_predictions.append(result)
                    except Exception as e:
                        logging.error(f"Error during inference for batch {batch_idx}: {str(e)}")
                        for filename in valid_files:
                            all_predictions.append({
                                'filename': filename,
                                'error': f"Batch inference error: {str(e)}"
                            })
                    
                    batch_idx += 1
                    if batch_idx % 10 == 0:
                        logging.info(f"Processed {batch_end}/{len(image_files)} images")
            
            processing_time = time.time() - start_time
            logging.info(f"Inference completed in {processing_time:.2f} seconds")
            
            # Create results dataframe
            predictions_df = pd.DataFrame(all_predictions)
            
            # Save predictions to CSV
            csv_path = os.path.join(output_dir, "predictions.csv")
            predictions_df.to_csv(csv_path, index=False)
            logging.info(f"Saved predictions to {csv_path}")
            
            # Calculate metrics if ground truth is available
            metrics = {}
            metrics_path = None
            cm_path = None
            roc_path = None
            
            if ground_truth and any('true_label' in p for p in all_predictions):
                logging.info("Calculating performance metrics...")
                
                # Filter predictions with ground truth
                valid_preds = [p for p in all_predictions if 'true_label' in p]
                
                # Extract true labels and predictions
                y_true = [p['true_label'] for p in valid_preds]
                y_pred = [p['predicted_class'] for p in valid_preds]
                
                # Get unique classes in sorted order
                unique_classes = sorted(list(set(y_true)))
                
                # Basic metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                
                try:
                    # Multi-class metrics
                    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    # Per-class metrics
                    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    
                    # Extract per-class metrics
                    for class_idx in unique_classes:
                        class_key = str(class_idx)
                        if class_key in class_report:
                            class_metrics = class_report[class_key]
                            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"class_{class_idx}"
                            
                            metrics[f'precision_{class_name}'] = class_metrics['precision']
                            metrics[f'recall_{class_name}'] = class_metrics['recall']
                            metrics[f'f1_{class_name}'] = class_metrics['f1-score']
                            
                            # Calculate sensitivity and specificity for each class
                            # Sensitivity = recall
                            metrics[f'sensitivity_{class_name}'] = class_metrics['recall']
                            
                            # Specificity = TN / (TN + FP)
                            y_true_binary = [1 if y == class_idx else 0 for y in y_true]
                            y_pred_binary = [1 if y == class_idx else 0 for y in y_pred]
                            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            metrics[f'specificity_{class_name}'] = specificity
                    
                    # Calculate AUC and ROC curves if we have probability outputs
                    if len(unique_classes) == 2:  # Binary classification
                        # Extract probabilities for positive class
                        class_idx = max(unique_classes)  # Assuming 1 is positive class in binary case
                        if class_names and class_idx < len(class_names):
                            prob_col = f'prob_{class_names[class_idx]}'
                        else:
                            prob_col = f'prob_class_{class_idx}'
                        
                        if prob_col in predictions_df.columns:
                            y_score = predictions_df.loc[predictions_df['true_label'].notna(), prob_col].values
                            y_true_for_auc = [y for i, y in enumerate(y_true)]
                            
                            # Calculate ROC curve and AUC
                            metrics['auc'] = roc_auc_score(y_true_for_auc, y_score)
                            fpr, tpr, _ = roc_curve(y_true_for_auc, y_score)
                            
                            # Plot ROC curve
                            plt.figure(figsize=(8, 8))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                    label=f'ROC curve (area = {metrics["auc"]:.3f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic')
                            plt.legend(loc="lower right")
                            
                            roc_path = os.path.join(output_dir, "roc_curve.png")
                            plt.savefig(roc_path)
                            plt.close()
                    elif len(unique_classes) > 2:  # Multi-class
                        # For multi-class, calculate one-vs-rest AUC for each class
                        for class_idx in unique_classes:
                            if class_names and class_idx < len(class_names):
                                prob_col = f'prob_{class_names[class_idx]}'
                                class_name = class_names[class_idx]
                            else:
                                prob_col = f'prob_class_{class_idx}'
                                class_name = f"class_{class_idx}"
                            
                            if prob_col in predictions_df.columns:
                                # Create binary labels for this class
                                y_true_binary = [1 if y == class_idx else 0 for y in y_true]
                                y_score = predictions_df.loc[predictions_df['true_label'].notna(), prob_col].values
                                
                                try:
                                    # Calculate AUC
                                    class_auc = roc_auc_score(y_true_binary, y_score)
                                    metrics[f'auc_{class_name}'] = class_auc
                                except Exception as e:
                                    logging.warning(f"Could not calculate AUC for {class_name}: {str(e)}")
                except Exception as e:
                    logging.warning(f"Error calculating some metrics: {str(e)}")
                
                # Log all metrics
                logging.info("Performance metrics:")
                for metric_name, metric_value in metrics.items():
                    logging.info(f"{metric_name}: {metric_value:.4f}")
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
                
                # Set up class labels for the plot
                if class_names:
                    labels = [class_names[i] if i < len(class_names) else f"class_{i}" for i in unique_classes]
                else:
                    labels = [f"class_{i}" for i in unique_classes]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                
                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Create a readable metrics summary as CSV
                metrics_summary = []
                for metric_name, metric_value in metrics.items():
                    metrics_summary.append({
                        'metric': metric_name,
                        'value': metric_value
                    })
                pd.DataFrame(metrics_summary).to_csv(
                    os.path.join(output_dir, "metrics_summary.csv"), index=False
                )
            
            # Return results
            return {
                "status": "success",
                "predictions_path": csv_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "roc_curve_path": roc_path,
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir,
                "num_images_processed": len(all_predictions),
                "num_images_with_errors": sum(1 for p in all_predictions if 'error' in p),
                "has_ground_truth": bool(ground_truth),
                "processing_time_seconds": processing_time,
                "model_type": "inception_v3",
                "aux_logits_used": aux_logits
            }
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir
            }

class PyTorchVGG16TrainingTool(Tool):
    name = "pytorch_vgg16_training"
    description = """
    This tool trains a VGG16 model using PyTorch for medical image classification.
    It can train from scratch or fine-tune a pre-trained model, and includes validation metrics.
    """
    
    inputs = {
        "data_dir": {
            "type": "string",
            "description": "Directory containing dataset with training and validation folders"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where the trained model and results will be saved"
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes for classification"
        },
        "num_epochs": {
            "type": "integer",
            "description": "Number of training epochs",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for training",
            "required": False,
            "nullable": True
        },
        "pretrained": {
            "type": "boolean",
            "description": "Whether to use pretrained weights",
            "required": False,
            "nullable": True
        },
        "early_stopping": {
            "type": "boolean", 
            "description": "Whether to use early stopping",
            "required": False,
            "nullable": True
        },
        "patience": {
            "type": "integer",
            "description": "Number of epochs to wait before early stopping",
            "required": False,
            "nullable": True
        },
        "use_batch_norm": {
            "type": "boolean",
            "description": "Whether to use VGG16 with batch normalization (VGG16_BN)",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        data_dir: str,
        output_dir: str,
        num_classes: int,
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 16,
        pretrained: Optional[bool] = True,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 5,
        use_batch_norm: Optional[bool] = False
    ):
        """
        Train a VGG16 model for medical image classification.
        
        Args:
            data_dir: Directory containing dataset with train and val subdirectories
            output_dir: Directory to save model and results
            num_classes: Number of classes for classification
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            pretrained: Whether to use pretrained weights
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            use_batch_norm: Whether to use VGG16 with batch normalization
            
        Returns:
            Dictionary with training results and model paths
        """
        try:
            # Define the image size for VGG16 (standard size is 224x224)
            vgg_image_size = (224, 224)
            
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "training.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Log configuration
            logging.info(f"Starting VGG16 training with configuration:")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Pretrained: {pretrained}")
            logging.info(f"Early stopping: {early_stopping}")
            logging.info(f"Use batch normalization: {use_batch_norm}")
            if early_stopping:
                logging.info(f"Patience: {patience}")
            logging.info(f"Data directory: {data_dir}")
            
            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
            # Define transformations for VGG16 (224x224 input)
            train_transform = transforms.Compose([
                transforms.Resize(vgg_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize(vgg_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Setup directories
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            test_dir = os.path.join(data_dir, 'test')
            
            if not os.path.exists(train_dir):
                raise ValueError(f"Training directory not found: {train_dir}")
            
            # Check if validation directory exists, otherwise use a split from training
            use_train_val_split = not os.path.exists(val_dir)
            
            # Check if data directory has a labels.csv file or if images are in class subdirectories
            train_labels_file = os.path.join(data_dir, 'labels.csv')
            use_csv_labels = os.path.exists(train_labels_file)
            
            # Create datasets based on the data organization
            if use_csv_labels:
                # Using CSV file with image paths and labels
                logging.info(f"Using labels from CSV file: {train_labels_file}")
                train_dataset = MedicalImageDataset(
                    image_dir=train_dir,
                    labels_file=train_labels_file,
                    transform=train_transform
                )
                
                if not use_train_val_split:
                    val_labels_file = os.path.join(data_dir, 'val_labels.csv')
                    if not os.path.exists(val_labels_file):
                        val_labels_file = train_labels_file  # Fallback to same labels file
                    
                    val_dataset = MedicalImageDataset(
                        image_dir=val_dir,
                        labels_file=val_labels_file,
                        transform=val_transform
                    )
            else:
                # Using directory structure (each class in its own subdirectory)
                logging.info("Using directory structure for class labels")
                train_dataset = datasets.ImageFolder(
                    root=train_dir,
                    transform=train_transform
                )
                
                if not use_train_val_split:
                    val_dataset = datasets.ImageFolder(
                        root=val_dir,
                        transform=val_transform
                    )
            
            # Split training data if no validation directory
            if use_train_val_split:
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size]
                )
            
            logging.info(f"Training dataset size: {len(train_dataset)}")
            logging.info(f"Validation dataset size: {len(val_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Create VGG16 model
            logging.info("Creating VGG16 model...")
            
            # Choose between VGG16 and VGG16_BN (with batch normalization)
            if use_batch_norm:
                logging.info("Using VGG16 with batch normalization")
                if pretrained:
                    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
                    logging.info("Using pretrained weights from ImageNet")
                else:
                    model = models.vgg16_bn(weights=None)
                    logging.info("Initializing model with random weights")
            else:
                logging.info("Using standard VGG16")
                if pretrained:
                    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                    logging.info("Using pretrained weights from ImageNet")
                else:
                    model = models.vgg16(weights=None)
                    logging.info("Initializing model with random weights")
            
            # Modify the classifier for our number of classes
            # VGG16's classifier is a sequential model with the last layer being the output layer
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            
            # Move model to device
            model = model.to(device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            
            # Using Adam optimizer with weight decay to prevent overfitting
            # VGG16 is prone to overfitting due to its large number of parameters
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=2, factor=0.5
            )
            
            # Track best model
            best_val_loss = float('inf')
            best_val_acc = 0.0
            best_epoch = 0
            patience_counter = 0
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'learning_rates': []
            }
            
            # Training loop
            logging.info("Starting training...")
            start_time = time.time()
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                epoch_train_loss = train_loss / train_total
                epoch_train_acc = 100 * train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = 100 * val_correct / val_total
                
                # Update learning rate
                scheduler.step(epoch_val_loss)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Save history
                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_val_loss)
                history['train_acc'].append(epoch_train_acc)
                history['val_acc'].append(epoch_val_acc)
                history['learning_rates'].append(current_lr)
                
                # Log progress
                logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Train Acc: {epoch_train_acc:.2f}%, "
                           f"Val Loss: {epoch_val_loss:.4f}, "
                           f"Val Acc: {epoch_val_acc:.2f}%, "
                           f"LR: {current_lr:.6f}")
                
                # Check for improvement
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_val_loss = epoch_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'accuracy': best_val_acc
                    }, best_model_path)
                    logging.info(f"Saved best model with accuracy: {best_val_acc:.2f}%")
                else:
                    patience_counter += 1
                
                # Early stopping if enabled
                if early_stopping and patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Calculate training time
            total_time = time.time() - start_time
            logging.info(f"Training completed in {total_time:.2f} seconds")
            
            # Save final model
            final_model_path = os.path.join(output_dir, "final_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc
            }, final_model_path)
            
            # Variables to track test results
            test_acc = None
            metrics_path = None
            cm_path = None
            
            # If test directory exists, evaluate on test set using best model
            if os.path.exists(test_dir):
                logging.info("Evaluating model on test set...")
                
                # Load best model for testing
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded best model from epoch {checkpoint['epoch']} for testing")
                
                # Create test dataset and dataloader
                if use_csv_labels:
                    test_labels_file = os.path.join(data_dir, 'test_labels.csv')
                    if not os.path.exists(test_labels_file):
                        test_labels_file = train_labels_file  # Fallback to same labels file
                    
                    test_dataset = MedicalImageDataset(
                        image_dir=test_dir,
                        labels_file=test_labels_file,
                        transform=val_transform  # Use the same transform as validation
                    )
                else:
                    test_dataset = datasets.ImageFolder(
                        root=test_dir,
                        transform=val_transform
                    )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )
                
                # Evaluate on test set
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        test_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                        
                        # Store predictions and labels for metrics
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                test_acc = 100 * test_correct / test_total
                logging.info(f"Test accuracy: {test_acc:.2f}%")
                
                # Calculate and log detailed metrics
                metrics = {}
                metrics['accuracy'] = accuracy_score(all_labels, all_preds)
                metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                
                # Add per-class metrics if there are multiple classes
                if num_classes > 1:
                    metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    metrics['f1_per_class'] = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                
                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "test_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Generate confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
            
            # Save training history
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Create plots
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')
            
            plt.subplot(1, 3, 2)
            plt.plot(history['train_acc'], label='Training Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Curves')
            
            plt.subplot(1, 3, 3)
            plt.plot(history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            
            plt.tight_layout()
            plots_path = os.path.join(output_dir, "training_plots.png")
            plt.savefig(plots_path)
            plt.close()
            
            # Save model configuration
            model_type = "vgg16_bn" if use_batch_norm else "vgg16"
            config = {
                'model_type': model_type,
                'num_classes': num_classes,
                'image_size': 224,  # VGG16 uses 224x224 images
                'pretrained': pretrained,
                'early_stopping': early_stopping,
                'patience': patience if early_stopping else None,
                'best_epoch': best_epoch,
                'best_accuracy': best_val_acc,
                'training_epochs': epoch + 1,
                'early_stopped': early_stopping and patience_counter >= patience,
                'batch_size': batch_size,
                'use_batch_norm': use_batch_norm
            }
            
            config_path = os.path.join(output_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Return results
            return {
                "status": "success",
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                "config_path": config_path,
                "plots_path": plots_path,
                "history_path": history_path,
                "test_metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "best_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "training_time_seconds": total_time,
                "epochs_completed": epoch + 1,
                "early_stopped": early_stopping and patience_counter >= patience,
                "early_stopping_used": early_stopping,
                "model_type": model_type,
                "use_batch_norm": use_batch_norm
            }
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "data_dir": data_dir,
                "output_dir": output_dir
            }

class PyTorchVGG16InferenceTool(Tool):
    name = "pytorch_vgg16_inference"
    description = """
    This tool uses a trained PyTorch VGG16 model to perform inference on new images.
    It can also calculate performance metrics if ground truth labels are provided.
    """
    
    inputs = {
        "image_dir": {
            "type": "string",
            "description": "Directory containing images for inference"
        },
        "model_path": {
            "type": "string",
            "description": "Path to the trained model file (.pt format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where prediction results will be saved"
        },
        "config_path": {
            "type": "string",
            "description": "Path to model configuration JSON file (optional)",
            "required": False,
            "nullable": True
        },
        "ground_truth_file": {
            "type": "string",
            "description": "CSV file with image filenames and ground truth labels for evaluation (optional)",
            "required": False,
            "nullable": True
        },
        "class_names": {
            "type": "array",
            "description": "List of class names corresponding to model output indices (optional)",
            "required": False,
            "nullable": True
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes (needed if not provided in config file)",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for inference",
            "required": False,
            "nullable": True
        },
        "case_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains case/file identifiers",
            "required": False,
            "nullable": True
        },
        "label_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains the labels",
            "required": False,
            "nullable": True
        },
        "file_extension": {
            "type": "string",
            "description": "File extension to add to case IDs if they don't already have one",
            "required": False,
            "nullable": True
        },
        "use_batch_norm": {
            "type": "boolean",
            "description": "Whether the model uses batch normalization (VGG16_BN)",
            "required": False,
            "nullable": True
        }
    }
    
    output_type = "object"
    
    def forward(
        self,
        image_dir: str,
        model_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        batch_size: Optional[int] = 32,
        case_column: Optional[str] = "case",
        label_column: Optional[str] = "label",
        file_extension: Optional[str] = ".png",
        use_batch_norm: Optional[bool] = False
    ):
        """
        Run inference using a trained PyTorch VGG16 model on new images.
        
        Args:
            image_dir: Directory containing images for inference
            model_path: Path to the trained model file (.pt format)
            output_dir: Directory to save prediction outputs
            config_path: Path to model configuration JSON file (optional)
            ground_truth_file: CSV file with filenames and ground truth labels (optional)
            class_names: List of class names corresponding to model output indices
            num_classes: Number of classes (needed if not in config file)
            batch_size: Batch size for inference
            case_column: Name of the column in ground truth file with case/file IDs
            label_column: Name of the column in ground truth file with labels
            file_extension: File extension to add to case IDs if needed
            use_batch_norm: Whether the model uses batch normalization
            
        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "inference.log")
            
            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Load model configuration if provided
            model_config = {}
            if config_path and os.path.exists(config_path):
                logging.info(f"Loading model configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        model_config = json.load(f)
                    logging.info(f"Model configuration loaded: {model_config}")
                except Exception as e:
                    logging.warning(f"Error loading model configuration: {str(e)}. Will use provided parameters.")
            
            # Set parameters, with priority to explicit parameters over config values
            # Determine if we're using VGG16 with batch normalization
            if use_batch_norm is None:
                use_batch_norm = model_config.get('use_batch_norm', False)
            
            # Set model type based on batch normalization flag
            model_type = "vgg16_bn" if use_batch_norm else "vgg16"
            
            if num_classes is None:
                num_classes = model_config.get('num_classes')
                if num_classes is None:
                    if class_names:
                        num_classes = len(class_names)
                    else:
                        raise ValueError("Number of classes must be provided either directly, in config file, or through class_names")
            
            # VGG16 uses 224x224 images
            image_size = model_config.get('image_size', 224)
            
            # Log inference settings
            logging.info(f"Starting VGG16 inference with settings:")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Model type: {model_type}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Image directory: {image_dir}")
            logging.info(f"Has ground truth: {ground_truth_file is not None}")
            
            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
            # Define image transformation
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Create model
            logging.info(f"Creating {model_type} model architecture...")
            
            # Choose between VGG16 and VGG16_BN
            if use_batch_norm:
                model = models.vgg16_bn(pretrained=False)
            else:
                model = models.vgg16(pretrained=False)
            
            # Modify the classifier for our number of classes
            # VGG16's classifier is a sequential model with the last layer being the output layer
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            
            # Load trained weights
            logging.info(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    logging.info("Model loaded directly from state dict")
            except Exception as e:
                logging.error(f"Error loading model weights: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model weights: {str(e)}",
                    "model_path": model_path
                }
            
            # Move model to device and set to evaluation mode
            model = model.to(device)
            model.eval()
            
            # Initialize variables for ground truth processing
            ground_truth = {}
            filename_to_case = {}
            
            # Load ground truth labels if provided
            if ground_truth_file and os.path.exists(ground_truth_file):
                logging.info(f"Loading ground truth data from {ground_truth_file}")
                try:
                    gt_df = pd.read_csv(ground_truth_file)
                    
                    # Check required columns
                    if case_column not in gt_df.columns or label_column not in gt_df.columns:
                        raise ValueError(f"Ground truth file must contain '{case_column}' and '{label_column}' columns")
                    
                    # Create mapping of case ID to filename
                    case_to_filename = {}
                    
                    # Create dictionary mapping filename to label
                    for _, row in gt_df.iterrows():
                        # Get case ID and convert to string
                        case_id = str(row[case_column])
                        
                        # Add file extension if needed
                        if not any(case_id.lower().endswith(ext) for ext in 
                                  ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                            filename = f"{case_id}{file_extension}"
                        else:
                            filename = case_id
                        
                        # Store the mapping and ground truth
                        case_to_filename[case_id] = filename
                        ground_truth[filename] = int(row[label_column])
                        
                    logging.info(f"Loaded {len(ground_truth)} ground truth labels")
                    
                    # Create a reverse mapping from filename to case ID for results
                    filename_to_case = {v: k for k, v in case_to_filename.items()}
                    
                except Exception as e:
                    logging.error(f"Error loading ground truth file: {str(e)}")
                    logging.warning("Will continue without ground truth evaluation")
                    ground_truth = {}
                    filename_to_case = {}
            
            # Get list of all image files
            image_files = [f for f in os.listdir(image_dir) 
                         if os.path.isfile(os.path.join(image_dir, f)) and 
                         f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
            
            if not image_files:
                logging.error(f"No valid image files found in {image_dir}")
                return {
                    "status": "error",
                    "error_message": f"No valid image files found in directory",
                    "image_dir": image_dir
                }
            
            logging.info(f"Found {len(image_files)} images for inference")
            
            # Process images in batches for efficiency
            all_predictions = []
            batch_idx = 0
            start_time = time.time()
            
            with torch.no_grad():  # No gradient computation needed for inference
                for batch_start in range(0, len(image_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(image_files))
                    batch_files = image_files[batch_start:batch_end]
                    batch_imgs = []
                    valid_files = []
                    
                    # Load and preprocess images in this batch
                    for img_file in batch_files:
                        try:
                            img_path = os.path.join(image_dir, img_file)
                            image = Image.open(img_path).convert('RGB')
                            image_tensor = transform(image)
                            batch_imgs.append(image_tensor)
                            valid_files.append(img_file)
                        except Exception as e:
                            logging.warning(f"Error processing image {img_file}: {str(e)}")
                            all_predictions.append({
                                'filename': img_file,
                                'error': str(e)
                            })
                    
                    if not batch_imgs:  # Skip if no valid images in batch
                        continue
                    
                    # Stack tensors into batch
                    batch_tensor = torch.stack(batch_imgs).to(device)
                    
                    # Run inference
                    outputs = model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Process results
                    batch_probs = probabilities.cpu().numpy()
                    batch_preds = np.argmax(batch_probs, axis=1)
                    
                    # Store predictions
                    for i, (filename, pred_class, probs) in enumerate(zip(valid_files, batch_preds, batch_probs)):
                        # Get corresponding case ID if available
                        case_id = filename_to_case.get(filename, filename)
                        
                        result = {
                            'filename': filename,
                            'case_id': case_id,
                            'predicted_class': int(pred_class),
                            'confidence': float(probs[pred_class])
                        }
                        
                        # Add ground truth if available
                        if filename in ground_truth:
                            result['true_label'] = int(ground_truth[filename])
                            result['correct'] = result['predicted_class'] == result['true_label']
                        
                        # Add class name if available
                        if class_names and pred_class < len(class_names):
                            result['predicted_class_name'] = class_names[pred_class]
                            
                        # Add probabilities for each class
                        for class_idx, prob in enumerate(probs):
                            if class_names and class_idx < len(class_names):
                                result[f'prob_{class_names[class_idx]}'] = float(prob)
                            else:
                                result[f'prob_class_{class_idx}'] = float(prob)
                        
                        all_predictions.append(result)
                    
                    batch_idx += 1
                    if batch_idx % 10 == 0:
                        logging.info(f"Processed {batch_end}/{len(image_files)} images")
            
            processing_time = time.time() - start_time
            logging.info(f"Inference completed in {processing_time:.2f} seconds")
            
            # Create results dataframe
            predictions_df = pd.DataFrame(all_predictions)
            
            # Save predictions to CSV
            csv_path = os.path.join(output_dir, "predictions.csv")
            predictions_df.to_csv(csv_path, index=False)
            logging.info(f"Saved predictions to {csv_path}")
            
            # Calculate metrics if ground truth is available
            metrics = {}
            metrics_path = None
            cm_path = None
            roc_path = None
            
            if ground_truth and any('true_label' in p for p in all_predictions):
                logging.info("Calculating performance metrics...")
                
                # Filter predictions with ground truth
                valid_preds = [p for p in all_predictions if 'true_label' in p]
                
                # Extract true labels and predictions
                y_true = [p['true_label'] for p in valid_preds]
                y_pred = [p['predicted_class'] for p in valid_preds]
                
                # Get unique classes in sorted order
                unique_classes = sorted(list(set(y_true)))
                
                # Basic metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                
                try:
                    # Multi-class metrics
                    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    # Per-class metrics
                    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    
                    # Extract per-class metrics
                    for class_idx in unique_classes:
                        class_key = str(class_idx)
                        if class_key in class_report:
                            class_metrics = class_report[class_key]
                            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"class_{class_idx}"
                            
                            metrics[f'precision_{class_name}'] = class_metrics['precision']
                            metrics[f'recall_{class_name}'] = class_metrics['recall']
                            metrics[f'f1_{class_name}'] = class_metrics['f1-score']
                            
                            # Calculate sensitivity and specificity for each class
                            # Sensitivity = recall
                            metrics[f'sensitivity_{class_name}'] = class_metrics['recall']
                            
                            # Specificity = TN / (TN + FP)
                            y_true_binary = [1 if y == class_idx else 0 for y in y_true]
                            y_pred_binary = [1 if y == class_idx else 0 for y in y_pred]
                            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            metrics[f'specificity_{class_name}'] = specificity
                    
                    # Calculate AUC and ROC curves if we have probability outputs
                    if len(unique_classes) == 2:  # Binary classification
                        # Extract probabilities for positive class
                        class_idx = max(unique_classes)  # Assuming 1 is positive class in binary case
                        if class_names and class_idx < len(class_names):
                            prob_col = f'prob_{class_names[class_idx]}'
                        else:
                            prob_col = f'prob_class_{class_idx}'
                        
                        if prob_col in predictions_df.columns:
                            y_score = predictions_df.loc[predictions_df['true_label'].notna(), prob_col].values
                            y_true_for_auc = [y for i, y in enumerate(y_true)]
                            
                            # Calculate ROC curve and AUC
                            metrics['auc'] = roc_auc_score(y_true_for_auc, y_score)
                            fpr, tpr, _ = roc_curve(y_true_for_auc, y_score)
                            
                            # Plot ROC curve
                            plt.figure(figsize=(8, 8))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                    label=f'ROC curve (area = {metrics["auc"]:.3f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic')
                            plt.legend(loc="lower right")
                            
                            roc_path = os.path.join(output_dir, "roc_curve.png")
                            plt.savefig(roc_path)
                            plt.close()
                    elif len(unique_classes) > 2:  # Multi-class
                        # For multi-class, calculate one-vs-rest AUC for each class
                        for class_idx in unique_classes:
                            if class_names and class_idx < len(class_names):
                                prob_col = f'prob_{class_names[class_idx]}'
                                class_name = class_names[class_idx]
                            else:
                                prob_col = f'prob_class_{class_idx}'
                                class_name = f"class_{class_idx}"
                            
                            if prob_col in predictions_df.columns:
                                # Create binary labels for this class
                                y_true_binary = [1 if y == class_idx else 0 for y in y_true]
                                y_score = predictions_df.loc[predictions_df['true_label'].notna(), prob_col].values
                                
                                try:
                                    # Calculate AUC
                                    class_auc = roc_auc_score(y_true_binary, y_score)
                                    metrics[f'auc_{class_name}'] = class_auc
                                except Exception as e:
                                    logging.warning(f"Could not calculate AUC for {class_name}: {str(e)}")
                except Exception as e:
                    logging.warning(f"Error calculating some metrics: {str(e)}")
                
                # Log all metrics
                logging.info("Performance metrics:")
                for metric_name, metric_value in metrics.items():
                    logging.info(f"{metric_name}: {metric_value:.4f}")
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
                
                # Set up class labels for the plot
                if class_names:
                    labels = [class_names[i] if i < len(class_names) else f"class_{i}" for i in unique_classes]
                else:
                    labels = [f"class_{i}" for i in unique_classes]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                
                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                # Create a readable metrics summary as CSV
                metrics_summary = []
                for metric_name, metric_value in metrics.items():
                    metrics_summary.append({
                        'metric': metric_name,
                        'value': metric_value
                    })
                pd.DataFrame(metrics_summary).to_csv(
                    os.path.join(output_dir, "metrics_summary.csv"), index=False
                )
            
            # Return results
            return {
                "status": "success",
                "predictions_path": csv_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "roc_curve_path": roc_path,
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir,
                "num_images_processed": len(all_predictions),
                "num_images_with_errors": sum(1 for p in all_predictions if 'error' in p),
                "has_ground_truth": bool(ground_truth),
                "processing_time_seconds": processing_time,
                "model_type": model_type,
                "use_batch_norm": use_batch_norm
            }
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir
            }
