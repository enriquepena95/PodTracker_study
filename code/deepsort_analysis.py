#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:44:56 2024

@author: jcdunne.lab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:45:37 2023

@author: enrique

"""

import sys
import datetime
import os
import glob
import io
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
#import dropbox
#import dropbox.files
#from access_dropbox import get_entries_in_path, get_folders_in_path
import random
import math
from datetime import datetime
import pickle
import time
import csv
from scipy.stats import kurtosis
#%%
# Set base folder
base_folder = os.path.expanduser("~/PodTracker_study")
sys.path.append(os.path.expanduser("~/PodTracker_study"))
from DeepSortMask.deep_sort import preprocessing, nn_matching
from DeepSortMask.deep_sort.detection import Detection
from DeepSortMask.deep_sort.tracker import Tracker
from DeepSortMask.toolsTracker import generate_detections as gdet


#%% Initialize GPU capabilities
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["QT_QPA_PLATFORM"] = "offscreen"


if torch.cuda.is_available():
    print('cuda is available')
    dev = torch.device('cuda')
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s ,s, device = dev), torch.zeros(s, s, s, s, device = dev))
    
#Initialize configuration for Mask-RCNN
cfg = get_cfg()
cfg.merge_from_file(os.path.join(base_folder, "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.MODEL.WEIGHTS = r'/home/enrique/detectron2/model_final.pth'
#cfg.MODEL.WEIGHTS = r'./enrique/peanut_model_final2.pth'
cfg.MODEL.WEIGHTS = os.path.join(base_folder, 'model_checkpoints/pod_models/model_final.pth')
# model threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
cfg.TEST.DETECTIONS_PER_IMAGE = 20
predictor = DefaultPredictor(cfg)

# Initializing tracking parameters
max_cosine_distance = 1
nn_budget = None
nms_max_overlap = 1.0


# Initialize deep sort
model_filename = os.path.join(base_folder, 'DeepSortMask/model_data/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

# Calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

# Initialize tracker
tracker = Tracker(metric)

#%% Define Functions

def extract_timestamp(filename):
    # Extract the timestamp part from the filename
    timestamp_str = filename.split('-')[1].split('.')[0]
    # Convert the timestamp string to a datetime object
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")


# Function to calculate IoU
def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1: Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        box2: Coordinates of the second bounding box in the format (x1, y1, x2, y2).
        
    Returns:
        IoU: Intersection over Union between the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x3)
    y_intersection = max(y1, y3)
    intersection_width = max(0, min(x2, x4) - x_intersection)
    intersection_height = max(0, min(y2, y4) - y_intersection)
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of each bounding box
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    
    # Calculate IoU
    iou = intersection_area / (area_box1 + area_box2 - intersection_area)
    
    return iou


#%% MAIN 

# Convert raw data to jpg format
#raw_to_jpg(rawdir, jpgdir)

# Initialize Variables
count = 0
center_points_prv_frame = []
max_dims = {} # Dictionary to store the maximum dims according to max area for each track_id
tracking_objects = {}
track_id = 0
distance_threshold = 300
iou_threshold = 0.5

# List of all classes in the maskRCNN model in the same order as used for training
class_names = ["peanut"]
allowed_classes = class_names

#data_path = r"/media/enrique/scanner_data/jpg_data/varieties/Bailey_1007/rep1"
#image_list = sorted([f for f in  os.listdir(data_path) if f.endswith('.jpg')], key=extract_timestamp)

#image_list = glob.glob(os.path.join(data_path, '*.jpg'))
compute_cal_factor = False


# initialize "all unique IDs" set
all_unique_ids = set()


# Initialize an empty list to store all detections after NMS
filtered_detections = []


# Record the start time
start_time = time.time()

# Define base directories
data_folder = os.path.join(base_folder, "data/counts_data") # <- for count analysis
raw_base = "raw_data"
jpg_base = "jpg_data"

# results folder
results_folder = os.path.join(base_folder, "results")


# models_folder
model_folder = os.path.join(base_folder, "model_checkpoints")


counts = ["20_count",
          "40_count", 
          "60_count",
          "80_count",
          "100_count",
          "120_count",
          "140_count",
          "160_count",
          "180_count",
          "200_count",
          "220_count",
          "240_count",
          "260_count",
          "280_count",
          "300_count"] # for count analysis

# Define reps within each variety
reps = ["rep1", "rep2", "rep3"]


# Iterate over each variety and process its raw to jpg conversion
for count_ in counts:
    for rep in reps:
        filtered_max_dims = {}
        max_dims = {}
        class_counts = {}
        y_pred_counts = {}
        data_path = os.path.join(data_folder, jpg_base, count_, rep)
        print(data_path)
        output_path = os.path.join(results_folder, "tables", "counts", count_, rep)
        plot_path = os.path.join(results_folder, "plots", "counts", count_, rep)
        
        # Check if CSV already exists
        csv_output_path = f'{output_path}/deepsortmask_{rep}.csv'
        if os.path.exists(csv_output_path):
            print(f"CSV file already exists at {csv_output_path}. Skipping to the next iteration.")
            continue  # This will skip to the next rep in the inner loop
            
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        image_list = sorted([f for f in  os.listdir(data_path) if f.endswith('.jpg')], key=extract_timestamp)

        compute_cal_factor = False


        for img in image_list:
        
            img_path = os.path.join(data_path, img)
            
            #image_pil = Image.open(img_path)
            image_in = cv2.imread(img_path)
            #image_in = np.asarray(image_pil)
            #image_in = cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB)
            
            cal_factor = (5.042014999389648 + 5.32524154663086) /2
            cal_factor_sq = cal_factor**2
            compute_cal_factor = True
            
        
            width = image_in.shape[1]
        
            # Define the region of interest (ROI) as the bottom part of the image
            roi = image_in[width // 4:-width//4, :]
        
        
            count +=1
            center_points_cur_frame = []     
        
            # Perform inference on image
            outputs = predictor(roi)
            
            predictions = outputs["instances"].to("cpu")
            
            predictions.pred_boxes.tensor[:, 0] += width // 4  # Adjust x_min
            predictions.pred_boxes.tensor[:, 2] += width // 4  # Adjust x_max
            
            
            
            bboxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy() // 1
            scores = outputs['instances'].get_fields()['scores'].cpu().numpy()
            classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
            masks = predictions.get("pred_masks").numpy().astype(np.uint8) * 255
            #print('Detected: ', [class_names[clas] for clas in classes])
            num_objects = len(bboxes)
            
            # Apply NMS to filter out redundant detections
            boxs = np.array([bboxes[i] for i in range(num_objects)])
            scores = np.array([scores[i] for i in range(num_objects)])
            classes = np.array([int(classes[i]) for i in range(num_objects)])
            
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            
            # Initialize figure and axis
            #fig, ax = plt.subplots(figsize=(10, 8))
            #ax.imshow(image_in)
            
            names = []
            deleted_indx = []
            detections = []
            for i in indices:
                bbox = bboxes[i]
                score = scores[i]
                class_name = class_names[classes[i]]
                mask = masks[i]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = [contour + [0, width//4] for contour in contours]
                if len(contours) > 0:
                    contours = [max(contours, key=cv2.contourArea)]
                    x, y, w, h = cv2.boundingRect(contours[0])
                    border_threshold = 10  # Adjust this threshold as needed
                    if (
                        x > border_threshold 
                        and y > border_threshold 
                        and (x + w) < (width - border_threshold)
                        and (x + w) > (width//4)
                    ):
                        mask = np.zeros_like(mask)
                        cv2.drawContours(image_in, contours, -1, (0,255,0), 3)
                        
                        x,y,w,h = cv2.boundingRect(contours[0])
                        cx = int((x + x + w)/2)
                        cy = int((y + y + h)/2)
                        cv2.circle(image_in, (cx,cy), 5, (0,0,255), -1)
                        center_points_cur_frame.append((cx,cy))
                    
                        # Compute the area, width, and length of each mask
                        area_cont = cv2.contourArea(contours[0])/cal_factor_sq
                        rect = cv2.minAreaRect(contours[0]) # Note we are using rotated minimum area
                        w_cont = min(rect[1])/cal_factor # in mm
                        l_cont = length = max(rect[1])/cal_factor # in mm
                        
                        # Update max dims for the track_id 
                        if track_id in max_dims:
                            if area_cont > max_dims[track_id]['area_cont']:
                                max_dims[track_id]['area_cont'] = area_cont
                                max_dims[track_id]['l_cont'] = l_cont
                                if w_cont < max_dims[track_id]['w_cont']:
                                    max_dims[track_id]['w_cont'] = w_cont
                        else:
                            max_dims[track_id] = {'area_cont': area_cont, 'w_cont': w_cont, 'l_cont': l_cont}
                
                
                #bbox_full_img = bbox.copy()
                #bbox_full_img[0::2] += width // 4
                
        
                
                
                feature = encoder(roi, bbox.reshape(1, -1))  # Assuming encoder is already defined
                detection = Detection(bbox, score, class_name, mask, feature)
                detections.append(detection)
                
                #ax.imshow(mask, alpha=0.5, cmap='Greens', interpolation='nearest')
                
                #ax.text(bbox_full_img[0], bbox_full_img[1], f'{class_name}: {score:.2f}', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
            
            
            #ax.set_title(f'Frame: {img}')
            #ax.axis('off')
            #plt.show()
            
            # Update tracker with detections
            tracker.predict()
            
            
            for track in tracker.tracks:
                if  track.is_confirmed() or track.time_since_update > 5:
                    continue
            
                bbox = track.to_tlbr()  # Assuming tracker has method to get bounding box in (top, left, bottom, right) format
                track_id = track.track_id
                
                # Overlay track_id on the image
                #cv2.putText(image_in, str(track_id), (int(bbox[0]), int(bbox[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(image_in, str(track_id), (int(bbox[0]), int(bbox[1]+500)), 0, 1, (0,0,255), 2)
            
            #plt.figure(figsize=(10, 10))
            #plt.imshow(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
            #plt.title(f'Predictions for {img}')
            #plt.axis('off')
            #plt.show()
            
            # Collect all unique IDs seen in this frame
            new_ids = [track.track_id for track in tracker.tracks]
            all_unique_ids.update(new_ids)
            
            tracker.update(detections)
        
        # Now you have all unique IDs across all frames in `all_unique_ids`
        total_unique_ids = len(all_unique_ids)
        #print(f"Total Unique IDs: {total_unique_ids}")
        
        
        # Record the end time
        end_time = time.time()
        
        # Calculate FPS
        elapsed_time = end_time - start_time
        fps = count / elapsed_time
        
        #print(f"Total Unique IDs: {len(all_unique_ids)}")
        print(f"Processing Time: {elapsed_time:.2f} seconds")
        print(f"FPS: {fps:.2f}")
        total_counts = len(max_dims)
        #print(f"length of {variety} {rep} is: {total_counts}")
        print(f"length of {count_} {rep} is: {total_counts}")
        # Remove really small instances (width < 5mm)  
        filtered_max_dims = {k:v for k,v in max_dims.items() if v['w_cont'] >=5}
        
        
        # Extract lists of values for each dimension
        area_cont_values = [info['area_cont'] for info in filtered_max_dims.values()]
        w_cont_values = [info['w_cont'] for info in filtered_max_dims.values()]
        l_cont_values = [info['l_cont'] for info in filtered_max_dims.values()]
        
        
        # Create dataframe
        
        df = pd.DataFrame(filtered_max_dims).T
        X = df[['area_cont','w_cont', 'l_cont']]
        X.to_csv(f'{output_path}/deepsortmask_{rep}.csv', index=False)  
        
        # Compute mean and standard deviation
        statistics_df = X.describe().loc[['mean', 'std']]
        
        # Compute kurtosis for each column
        kurtosis_values = X.apply(kurtosis)
        
        # Append kurtosis to statistics_df
        statistics_df.loc['kurtosis'] = kurtosis_values
        
        # Define output path for statistics.txt
        statistics_file = os.path.join(f'deepsortmask_{output_path}', 'statistics.txt')
        
        
        #with open(statistics_file, 'w') as f:
        #    f.write(statistics_df.to_string())
        
        with open(os.path.join(model_folder, 'trained_model.pkl'), 'rb') as f: 
            clf = pickle.load(f)
        
        y_pred_counts = {}
        
        y_pred_unique = {}
        y_pred = clf.predict(X)
        
        y_pred_unique, y_pred_counts = np.unique(y_pred, return_counts=True)
        
        class_name_mapping = {'blue': 'No1', 'white': 'Fancy', 'red': 'Jumbo'}
        
        y_pred_unique_renamed = np.array([class_name_mapping.get(class_name, class_name) for class_name in y_pred_unique])
        
        # Create a DataFrame with the counts for each class
        
        for class_name in ['No1', 'Fancy', 'Jumbo']:
            class_counts[class_name] = y_pred_counts[np.where(y_pred_unique_renamed == class_name)].sum()
        
        # Convert dictionary to DataFrame
        class_df = pd.DataFrame()
        class_df = pd.DataFrame({'Predicted Classes': list(class_counts.keys()), 'Count': list(class_counts.values())})
        class_df.to_csv(f'{output_path}/grades_deepsortmask_{rep}.csv')
        
        # Plot
        #plt.figure(figsize=(8, 6))
        #sns.barplot(x='Predicted Classes', y='Count', data=class_df)
        #plt.xlabel('Predicted Classes')
        #plt.ylabel('Count')
        #plt.title(f'Predicted Classes. Variety: {variety}. Rep: {rep}')
        #plt.title(f'Predicted Classes. Count: {count_}. Rep: {rep}')
        #plt.tight_layout()
        #plt.savefig(f'{plot_path}/deepsortmask_barplot.png', dpi=400)
        #plt.show()
        #plt.clf()
        #plt.close()
        
        
        
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        
        # Plotting density plots for length, width, and area
        sns.kdeplot(l_cont_values, fill=True, ax=axes[0])
        axes[0].set_title('Density Plot of Length [mm]')
        axes[0].set_xlabel('Length [mm]')
        axes[0].set_ylabel('Density')
        
        sns.kdeplot(w_cont_values, fill=True, ax=axes[1])
        axes[1].set_title('Density Plot of Width [mm]')
        axes[1].set_xlabel('Width [mm]')
        axes[1].set_ylabel('Density')
        
        sns.kdeplot(area_cont_values, fill=True, ax=axes[2])
        axes[2].set_title('Density Plot of Area [mm^2]')
        axes[2].set_xlabel('Area [mm^2]')
        axes[2].set_ylabel('Density')
        
        # Adjust layout and display the plot
        #plt.title(f'Density plots. Variety: {variety}. Rep: {rep}')
        plt.tight_layout()
        plt.savefig(f'{plot_path}/deepsortmask_density.png', dpi=400)
        #plt.show()
        plt.clf()
        plt.close()

