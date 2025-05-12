#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:41:47 2025

@author: jcdunne.lab
"""

#!/usr/bin/env python
# coding: utf-8

# # Detectron2 Mask-RCNN on SP images

import sys
print(sys.executable)
import json
import os
import random
import cv2
import matplotlib
# Force matplotlib to not use any Xwindow backend - prevents display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil
import math
from cmath import inf
from collections import defaultdict
from importlib import reload
from time import sleep
import pandas as pd

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging


# Setup logger and save logs to current directory
from detectron2.utils.logger import setup_logger
setup_logger(output=r".")

# Define paths
base_folder = os.path.expanduser("~/PodTracker_study")
data_folder = os.path.join(base_folder, "data/pod_model_data")
detectron2_folder = os.path.join(base_folder, "detectron2")
models_folder = os.path.join(base_folder, "model_checkpoints")
train_data_folder = os.path.join(data_folder, "train/images")
val_data_folder = os.path.join(data_folder, "val/images")
test_data_folder = os.path.join(data_folder, "test/images")
results_folder = os.path.join(base_folder, "results")
plots_folder = os.path.join(results_folder, "plots")
tables_folder = os.path.join(results_folder, "tables")

os.makedirs(base_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)
os.makedirs(detectron2_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)
os.makedirs(train_data_folder, exist_ok=True)
os.makedirs(val_data_folder, exist_ok=True)
os.makedirs(test_data_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(tables_folder, exist_ok=True)

# Save the new annotations files
train_annotations_file = os.path.join(data_folder, 'train/annotations.json')
val_annotations_file = os.path.join(data_folder, 'val/annotations.json')
test_annotations_file = os.path.join(data_folder, 'test/annotations.json')

# Register your dataset
register_coco_instances("my_pod_dataset_train", {}, train_annotations_file, train_data_folder)
register_coco_instances("my_pod_dataset_val", {}, val_annotations_file, val_data_folder)
register_coco_instances("my_pod_dataset_test", {}, test_annotations_file, test_data_folder)

# Get metadata for your dataset
metadata = MetadataCatalog.get("my_pod_dataset_train").set(thing_classes=["Peanut",], evaluator_type="coco")

# Check if model exists (will be referenced later)
model_output_dir = os.path.join(models_folder, "pod_models")
final_model_path = os.path.join(model_output_dir, "model_final.pth")
model_exists = os.path.exists(final_model_path)

if model_exists:
    print(f"Final model already exists at {final_model_path}")
    print("Skipping training phase and proceeding directly to evaluation")
else:
    print("Final model not found. Will proceed with training.")
    
    
# Load the model for inference (whether we just trained it or it already existed)
def load_json_arr(json_path):
    lines = []
    try:
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return []
    return lines


# Prepare Training Configuration
def prepare_cfg(num_epochs=20, debug=True):
    cfg = get_cfg()
    
    config_file = os.path.join(detectron2_folder, "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(config_file)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.DATASETS.TRAIN = ("my_pod_dataset_train",)
    # This is labeled as DATASETS.TEST, but acts as their validation dataset
    cfg.DATASETS.TEST = ("my_pod_dataset_val",)

    num_train_imgs = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))

    # Initialize weights from model zoo
    cfg.MODEL.WEIGHTS = os.path.join(models_folder, "sp_model.pth")

    # Use GPU
    cfg.MODEL.DEVICE = "cuda"
    # Number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Set the confidence threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Base learning rate
    cfg.SOLVER.BASE_LR = 0.002

    # Batch size
    cfg.SOLVER.IMS_PER_BATCH = 1

    # Number of iterations to train
    cfg.SOLVER.MAX_ITER = int(num_train_imgs * num_epochs / cfg.SOLVER.IMS_PER_BATCH)
    # Number of iterations before saving weight
    cfg.SOLVER.CHECKPOINT_PERIOD = int(num_train_imgs / cfg.SOLVER.IMS_PER_BATCH)
    # Number of iterations before validating (evaluating on test set)
    cfg.TEST.EVAL_PERIOD = int(num_train_imgs / cfg.SOLVER.IMS_PER_BATCH)

    # Max number of detections in an image
    cfg.TEST.DETECTIONS_PER_IMAGE = 20

    # Output directory, where weights and output metrics are stored
    cfg.OUTPUT_DIR = model_output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if debug:
        print(f"Number of images: {num_train_imgs}")
        print(f"Number of iterations: {cfg.SOLVER.MAX_ITER}")
        print(f"Number of iterations per weights generated: {cfg.SOLVER.CHECKPOINT_PERIOD}")
        print(f"Number of iterations per validation step: {cfg.TEST.EVAL_PERIOD}")

    return cfg

cfg = prepare_cfg(num_epochs = 10, debug = True)

# Only define and run training if the model doesn't exist
if not model_exists:
    class LossEvalHook(HookBase):
        def __init__(self, eval_period, model, data_loader):
            self._model = model
            self._period = eval_period
            self._data_loader = data_loader
        
        def _do_loss_eval(self):
            # Copying inference_on_dataset from evaluator.py
            total = len(self._data_loader)
            num_warmup = min(5, total - 1)
                
            start_time = time.perf_counter()
            total_compute_time = 0
            losses = []
            for idx, inputs in enumerate(self._data_loader):            
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                loss_batch = self._get_loss(inputs)
                losses.append(loss_batch)
            mean_loss = np.mean(losses)
            self.trainer.storage.put_scalar('validation_loss', mean_loss)
            comm.synchronize()

            return losses
                
        def _get_loss(self, data):
            # How loss is calculated on train_loop 
            metrics_dict = self._model(data)
            metrics_dict = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in metrics_dict.items()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())
            return total_losses_reduced
            
            
        def after_step(self):
            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                self._do_loss_eval()
            self.trainer.storage.put_scalars(timetest=12)


    class CustomTrainer(DefaultTrainer):
        """
        Custom Trainer deriving from the "DefaultTrainer"
        Overloads build_hooks to add a hook to calculate loss on the test set during training.
        """

        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1, LossEvalHook(
                100, # Frequency of calculation - every 100 iterations here
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                )
            ))

            return hooks

    # Set seeds for reproducibility
    def set_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seeds before evaluation
    set_seeds(42)  # You can use any seed value you prefer

    # Train the model
    print("Starting model training...")
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("Model training completed successfully.")

    # Plot losses after training without displaying
    try:
        experiment_metrics = load_json_arr(os.path.join(cfg.OUTPUT_DIR, 'metrics.json'))
        
        plt.figure(figsize=(10, 6))
        plt.title("Total Loss Across Training Iterations", fontdict={'fontsize': 15, 'fontfamily': 'sans-serif'})
        
        if experiment_metrics:
            plt.plot(
                [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
                [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], 'p', color="#ff4d00", markersize=5)
            plt.plot(
                [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
                [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], 's', color="#00ddff", markersize=5)
        
            plt.legend(['Training', 'Validation'], loc='upper right', fontsize=12)
            plt.xlabel("Iterations", fontsize=13)
            plt.ylabel("Loss", fontsize=13)
            
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(alpha=0.2)
            plt.minorticks_on()
            
            # Save the plot without displaying it
            loss_plot_path = os.path.join(plots_folder, 'maskrcnn_training_loss.png')
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()  # Close the figure to prevent display
            print(f"Loss plot saved to {loss_plot_path}")
        else:
            print("No metrics data found for plotting")
    except Exception as e:
        print(f"Error generating loss plot: {e}")

# Configure model for inference
print("Configuring model for evaluation...")
cfg.MODEL.WEIGHTS = final_model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cpu'  # Use CPU for inference to avoid potential CUDA issues

predictor = DefaultPredictor(cfg)

# Get Average Precision Scores - This runs regardless of whether we trained or loaded a model
print("Calculating AP scores...")
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

try:
    evaluator = COCOEvaluator("my_pod_dataset_test", cfg, False, output_dir=os.path.join(models_folder, "output/"))
    test_loader = build_detection_test_loader(cfg, "my_pod_dataset_test")
    print(f"Running inference on test dataset...")
    res = inference_on_dataset(predictor.model, test_loader, evaluator)
    print("Inference completed.")

    # Create a list to hold all metrics data
    data = []

    # Process each task type (bbox, segm, etc.)
    for task, metrics in res.items():
        for metric, value in metrics.items():
            data.append({
                'task': task,
                'metric': metric, 
                'value': value
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    ap_scores_path = os.path.join(tables_folder, "ap_scores.csv")
    df.to_csv(ap_scores_path, index=False)
    print(f"AP scores successfully saved to {ap_scores_path}")
    
    # Also print the scores to console
    print("\nAP Scores Summary:")
    for task, metrics in res.items():
        print(f"\n{task} metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
except Exception as e:
    print(f"Error during AP score calculation: {e}")
    
print("Script completed.")