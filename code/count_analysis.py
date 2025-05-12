#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:45:37 2023

@author: enrique
"""

import sys
import datetime
import os
code_dir = os.path.expanduser("~/PodTracker_study/code")
sys.path.append(code_dir)
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
#from raw_to_jpg import raw_to_jpg
#os.chdir('peanuts_code')
#from aruco_finder_measure import findArucoMarkers
import pickle
import csv
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression

#%% Initialize GPU capabilities
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set base folder
base_folder = os.path.expanduser("~/PodTracker_study")

# data folder
data_folder = os.path.join(base_folder, "data/counts_data")
raw_base = os.path.join(data_folder, "raw_data")
jpg_base = os.path.join(data_folder, "jpg_data")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(raw_base, exist_ok=True)
os.makedirs(jpg_base, exist_ok=True)

# models_folder
model_folder = os.path.join(base_folder, "model_checkpoints")

# results folder
results_folder = os.path.join(base_folder, "results")

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
cfg.MODEL.WEIGHTS = os.path.join(base_folder, 'model_checkpoints/pod_models/model_final.pth')
#cfg.MODEL.WEIGHTS = r'./enrique/peanut_model_final2.pth'
# model threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
cfg.TEST.DETECTIONS_PER_IMAGE = 20
predictor = DefaultPredictor(cfg)


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

# Initialize Variables
count = 0
center_points_prv_frame = []
max_dims = {} # Dictionary to store the maximum dims according to max area for each track_id
tracking_objects = {}
track_id = 0
distance_threshold = 300
iou_threshold = 0.5



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

# Switching to agg to avoid Qt dependency when plotting
#plt.switch_backend('agg')

# Iterate over each variety and process its raw to jpg conversion
for count_ in counts:
    for rep in reps:
        filtered_max_dims = {}
        max_dims = {}
        class_counts = {}
        y_pred_counts = {}
        data_path = os.path.join(jpg_base, count_, rep)
        print(data_path)
        output_path = os.path.join(results_folder, "tables", "counts", count_, rep)
        plot_path = os.path.join(results_folder, "plots", "counts", count_, rep)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        image_list = sorted([f for f in  os.listdir(data_path) if f.endswith('.jpg')], key=extract_timestamp)

        compute_cal_factor = False

        for img in image_list:
            
            img_path = os.path.join(data_path, img)
            
            #image_pil = Image.open(img_path)
            image_pil = cv2.imread(img_path)
            image_in = np.asarray(image_pil)
            
            cal_factor = (5.042014999389648 + 5.32524154663086) /2
            cal_factor_sq = cal_factor**2
            compute_cal_factor = True
            
            width = image_in.shape[1]
        
            # Define the region of interest (ROI) as the bottom part of the image
            #roi = image_in[:, width // 4:]
            roi = image_in[width // 4:-width//4, :]
        
        
            #cv2.imwrite(os.path.join(r"data", f'{img}.jpg'), roi)
        
        
            count +=1
            center_points_cur_frame = []     
        
            # Perform inference on image
            outputs = predictor(roi)
            
            predictions = outputs["instances"].to("cpu")
            
            predictions.pred_boxes.tensor[:, 0] += width // 4  # Adjust x_min
            predictions.pred_boxes.tensor[:, 2] += width // 4  # Adjust x_max
            
            masks = predictions.get("pred_masks").numpy().astype(np.uint8) * 255
            for idx, mask in enumerate(masks):
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = [contour + [0, width//4] for contour in contours]
                if len(contours) > 0:
                    contours = [max(contours, key=cv2.contourArea)]
                    
                    # Check if the bounding box is close to the image border
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
        
                        
                            
            # We compare previous and current frame only at the beginning
            if count <= 2:
                for pt in center_points_cur_frame:
                    for pt2 in center_points_prv_frame:
                        try:
                            distance = math.hypot(pt2[0]-pt[0], pt2[1] - pt[1])
                        except TypeError:
                            continue
                        
                        if distance < 200:
                            tracking_objects[track_id] = pt
                            track_id +=1
            
            else:
                tracking_objects_copy = tracking_objects.copy()
                center_points_cur_frame_copy = center_points_cur_frame.copy()
            
                for object_id, pt2 in tracking_objects_copy.items():
                    object_exists = False
                    min_distance = float('inf')
                    closest_pt = None
                    
                    for pt in center_points_cur_frame:
                        try:
                            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        except TypeError:
                            continue
                        
                        # Check for IoU with objects from previous frame
                        max_iou = 0
                        for prev_object_id, prev_pt in tracking_objects_copy.items():
                            try:
                                prev_box = (prev_pt[0] - 5, prev_pt[1] - 5, prev_pt[0] + 5, prev_pt[1] + 5)  # Assuming a bounding box of 10x10
                            except TypeError:
                                continue
                            cur_box = (pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5)
                            iou = calculate_iou(prev_box, cur_box)
                            if iou > max_iou:
                                max_iou = iou
                        
                        if max_iou > iou_threshold:  # Threshold for IoU
                            tracking_objects[object_id] = pt
                            closest_pt = pt
                            object_exists = True
                            
                            #if pt in center_points_cur_frame:
                            #    center_points_cur_frame.remove(pt)
                            continue
        
        
                        # Update IDs position based on distance if there's no significant IoU overlap
                        if distance < distance_threshold and pt[1] > pt2[1]:  # Check if object is lower
                            if distance < min_distance:
                                min_distance = distance
                                closest_pt = pt
                                object_exists = True
                       
        
                    
                    # Update ID position with the closest point
                    if object_exists:
                        tracking_objects[object_id] = closest_pt
                        if closest_pt in center_points_cur_frame:
                            center_points_cur_frame.remove(closest_pt)
                    else:
                        # Remove IDs lost
                        tracking_objects.pop(object_id)
                        
                    
                # Add new found IDs
                for pt in center_points_cur_frame:
                    tracking_objects[track_id] = pt
                    track_id += 1       
            
            
                for object_id, pt in tracking_objects.items():
                    cv2.circle(image_in, pt, 5, (0,0,255), -1)
                    try:
                        cv2.putText(image_in, str(object_id), (pt[0], pt[1]-20), 0, 1, (0,0,255), 2)
                    except TypeError:
                        continue
            
            # Plot the image with predictions
            #plt.figure(figsize=(10, 10))
            #plt.imshow(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
            #plt.title(f'Predictions for {img}')
            #plt.axis('off')
            #plt.show()
        
            
            #print("Tracking Objects")
            #print(tracking_objects)
        
            #print("Cur. Frame Left Pts")
            #print(center_points_cur_frame)
                
            #print("Prev. Frame")
            #print(center_points_prv_frame)
        
            # Make a copy of the center points
            center_points_prv_frame = center_points_cur_frame.copy()
            
            #cv2.imwrite(f'/media/enrique/UBUNTU 22_0/output/{img}.jpg', image_in)
            #cv2.imwrite('test.jpg', image_in)


# Clean up data, compute statistics, and save

        # Remove really small instances (width < 5mm)  
        #filtered_max_dims = {k:v for k,v in max_dims.items() if v['w_cont'] >=5}
    
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #np.save(f'{output_path}.npy', filtered_max_dims)

        
        # Extract lists of values for each dimension
        area_cont_values = [info['area_cont'] for info in max_dims.values()]
        w_cont_values = [info['w_cont'] for info in max_dims.values()]
        l_cont_values = [info['l_cont'] for info in max_dims.values()]
        
        
        # Create dataframe
        
        df = pd.DataFrame(max_dims).T
        X = df[['area_cont','w_cont', 'l_cont']]
        X.to_csv(f'{output_path}/podtracker_{rep}.csv', index=False)  
        
        
        
        # Compute mean and standard deviation
        statistics_df = X.describe().loc[['mean', 'std']]

        # Compute kurtosis for each column
        kurtosis_values = X.apply(kurtosis)

        # Append kurtosis to statistics_df
        statistics_df.loc['kurtosis'] = kurtosis_values

        # Define output path for statistics.txt
        statistics_file = os.path.join(f'{output_path}', 'statistics.txt')


        #with open(statistics_file, 'w') as f:
        #    f.write(statistics_df.to_string())
        
# Visualizations for each rep

        
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
        class_df.to_csv(f'{output_path}/counts_podtracker_{rep}.csv')
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Predicted Classes', y='Count', data=class_df)
        plt.xlabel('Predicted Classes')
        plt.ylabel('Count')
        plt.title(f'Predicted Classes. Count: {count_}. Rep: {rep}') # <- for count analysis

        plt.tight_layout()
        plt.savefig(f'{plot_path}/barplot.png', dpi=400)
        #plt.show()
        #plt.clf()
        plt.close()
        

#%%

# Initialize an empty DataFrame
combined_df = pd.DataFrame()
podtracker_df = pd.DataFrame()
deepsortmask_df = pd.DataFrame()

# Directory where your CSV files are located
output_path = os.path.join(results_folder, 'tables/counts')
# Iterate over directories and files
for root, dirs, files in os.walk(output_path):
    
    # Extract variety and rep from directory path
    rep = os.path.basename(root)
    count = os.path.basename(os.path.dirname(root))
    for file in files:
        if file.startswith('podtracker_rep'):
            
            # Read CSV file
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Add count and rep columns
            df['count'] = count 
            df['rep'] = rep
            df['tracker'] = 'PodTracker'
            
            # Append to podtracker_df
            podtracker_df = pd.concat([podtracker_df, df], ignore_index=True)
        
        
        if file.startswith('deepsortmask_rep'):

            # Read CSV file
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Add count and rep columns
            df['count'] = count # <- for count analysis
            df['rep'] = rep
            df['tracker'] = 'DeepSortMask'
            
            # Append to deepsortmask_df
            deepsortmask_df = pd.concat([deepsortmask_df, df], ignore_index=True)

combined_df = pd.concat([podtracker_df, deepsortmask_df])
# Convert 'rep' column to categorical
combined_df['rep'] = pd.Categorical(combined_df['rep'])         

combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]   


      
# Get unique varieties
counts = combined_df['count'].unique() # <- for count analysis


#%% Counts Line plot

# Define actual counts for count analysis
actual_counts = {
    '20_count': 20,
    '40_count': 40,
    '60_count': 60,
    '80_count': 80,
    '100_count': 100,
    '120_count': 120,
    '140_count': 140,
    '160_count': 160,
    '180_count': 180,
    '200_count': 200,
    '220_count': 220,
    '240_count': 240,
    '260_count': 260,
    '280_count': 280,
    '300_count': 300
}


grouped_df = combined_df.groupby(['count', 'rep', 'tracker']).size().reset_index(name='counts')
tracker_stats = grouped_df.groupby(['tracker'])['counts'].agg(['mean', 'std']).unstack('tracker')

# Convert counts to categorical type with order
categories = list(actual_counts.keys())
cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
grouped_df['count'] = grouped_df['count'].astype(cat_type)

# Group by 'count', 'tracker', and calculate mean and standard deviation across reps
tracker_stats = grouped_df.groupby(['count', 'tracker']).agg({
    'counts': ['mean', 'std']
}).reset_index()

# Flatten MultiIndex columns
tracker_stats.columns = ['count', 'tracker', 'mean', 'std']

# Reformat data for plotting
tracker_means = tracker_stats.pivot(index='count', columns='tracker', values='mean')
tracker_std = tracker_stats.pivot(index='count', columns='tracker', values='std')

# Compute RMSE and MAE
rmse_dict = {}
mae_dict = {}

for tracker in tracker_means.columns:
    # Predicted values for this tracker
    predictions = tracker_means[tracker].reindex(categories)
    
    # Actual values for this tracker
    actuals = pd.Series([actual_counts[c] for c in categories], index=categories)
    
    # Compute errors
    errors = predictions - actuals
    squared_errors = errors ** 2
    absolute_errors = errors.abs()
    
    # Compute RMSE and MAE
    rmse = np.sqrt(squared_errors.mean())
    mae = absolute_errors.mean()
    
    rmse_dict[tracker] = rmse
    mae_dict[tracker] = mae

# Display RMSE and MAE
print("RMSE:")
for tracker, rmse in rmse_dict.items():
    print(f"{tracker}: {rmse:.2f}")

print("\nMAE:")
for tracker, mae in mae_dict.items():
    print(f"{tracker}: {mae:.2f}")
    
# Extract RMSE and MAE for specific trackers
trackers_to_save = ['DeepSortMask', 'PodTracker']

performance_data = {
    'Tracker': [],
    'RMSE': [],
    'MAE': []
}

for tracker in trackers_to_save:
    if tracker in rmse_dict and tracker in mae_dict:
        performance_data['Tracker'].append(tracker)
        performance_data['RMSE'].append(rmse_dict[tracker])
        performance_data['MAE'].append(mae_dict[tracker])

# Create a DataFrame
performance_df = pd.DataFrame(performance_data)

# Save to CSV
csv_path = os.path.join(results_folder, 'tables/count_performance.csv')
performance_df.to_csv(csv_path, index=False)


# Prepare for plotting
plt.figure(figsize=(10, 10))  # Create a square figure

# Positions of the bars on the x-axis
r1 = range(len(categories))

# Prepare data for plotting
actual_counts_values = [actual_counts[c] for c in categories]
x_labels = [c.split('_')[0] for c in categories]  # Extract numeric part from 'count' keys

# Define colors for plotting
tracker_colors = [(128/255, 128/255, 128/255), (186/255, 9/255, 81/255)]

# Plot diagonal line (ideal case where predicted = actual)
plt.plot([0, 300], [0, 300], 'b--', label='Ideal Case', linewidth=2)  # Blue dashed line

# Plot counts for each tracker with error bars
for i, (tracker, color) in enumerate(zip(tracker_means.columns, tracker_colors)):
    means = tracker_means[tracker]
    stds = tracker_std[tracker]
    
    # Line plot for tracker with points connected
    plt.plot(actual_counts_values, means, color=color, linestyle='-', marker='o', linewidth=2, markersize=8, label=f'{tracker}')
    
    # Error bars representing standard deviation
    plt.errorbar(actual_counts_values, means, yerr=stds, fmt='o', color=color, alpha=0.7, capsize=5, capthick=2)

# Adding labels and title
plt.xlabel('True Counts')
plt.ylabel('Predicted Counts')

# Set the x and y axis ticks to be '20', '40', ..., '300'
plt.xticks(np.arange(0, 310, 20), labels=[str(i) for i in np.arange(0, 310, 20)])
plt.yticks(np.arange(0, 610, 20), labels=[str(i) for i in np.arange(0, 610, 20)])

# Ensure the plot is square
plt.gca().set_aspect('equal', adjustable='box')

plt.legend()

# Show plot
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'plots/counts_lineplot.png'), dpi=400)
#plt.savefig(os.path.join(results_folder, 'plots/counts_lineplot.svg'), dpi=400, format='svg')
plt.show()

# Plot with adjusted predictions

plt.figure(figsize=(10, 10))

# Fit a line to the predictions of each tracker and adjust predictions
rmse_dict_adjusted = {}
mae_dict_adjusted = {}

for i, (tracker, color) in enumerate(zip(tracker_means.columns, tracker_colors)):
    # Extract mean and actual counts
    y = tracker_means[tracker].reindex(categories)
    X = np.array([actual_counts[c] for c in categories]).reshape(-1, 1)
    
    # Fit a linear model to the data
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Adjust predictions to remove bias
    adjusted_predictions = (y - intercept) / slope
    
    # Plot adjusted predictions
    plt.plot(actual_counts_values, adjusted_predictions, color=color, linestyle='-', marker='o', linewidth=2, markersize=8, label=f'{tracker} (Adjusted)')
    
    # Optionally, plot error bars for adjusted predictions
    stds = tracker_std[tracker]
    plt.errorbar(actual_counts_values, adjusted_predictions, color = color, yerr=stds / slope, fmt='o', alpha=0.7, capsize=5, capthick=1)
    
    # Compute errors for adjusted predictions
    errors = adjusted_predictions - np.array([actual_counts[c] for c in categories])
    squared_errors = errors ** 2
    absolute_errors = errors.abs()
    
    # Compute RMSE and MAE for adjusted predictions
    rmse_adjusted = np.sqrt(squared_errors.mean())
    mae_adjusted = absolute_errors.mean()
    
    rmse_dict_adjusted[tracker] = rmse_adjusted
    mae_dict_adjusted[tracker] = mae_adjusted

# Display RMSE and MAE for adjusted predictions
print("\nRMSE Adjusted:")
for tracker, rmse in rmse_dict_adjusted.items():
    print(f"{tracker}: {rmse:.2f}")

print("\nMAE Adjusted:")
for tracker, mae in mae_dict_adjusted.items():
    print(f"{tracker}: {mae:.2f}")

# Plot diagonal line (ideal case where adjusted predicted = actual)
plt.plot([0, 300], [0, 300], 'b--', label='Ideal Case', linewidth=2)

# Adding labels and title
plt.xlabel('True Counts')
plt.ylabel('Adjusted Predictions')

# Set the x and y axis ticks to be '20', '40', ..., '300'
plt.xticks(np.arange(0, 310, 20), labels=[str(i) for i in np.arange(0, 310, 20)])
plt.yticks(np.arange(0, 310, 20), labels=[str(i) for i in np.arange(0, 310, 20)])

# Ensure the plot is square
plt.gca().set_aspect('equal', adjustable='box')

plt.legend()

# Show plot
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'plots/adjusted_predictions_plot.png'), dpi=400)
#plt.savefig('/media/enrique/scanner_data/analysis/plots/adjusted_predictions_plot.svg', dpi=400, format='svg')
plt.show()
