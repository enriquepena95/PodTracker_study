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
import os
from sklearn.linear_model import LinearRegression

#%% Initialize GPU capabilities
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Set base folder
base_folder = os.path.expanduser("~/PodTracker_study")

# data folder
data_folder = os.path.join(base_folder, "data/grade_data")
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



# Define specific varieties and their corresponding folders
varieties = [
    "Bailey_1007",
    "BaileyII_1003",
    "Col16_cross",
    "Col73_cross",
    "Col135_cross",
    "Col160_cross",
    "Col_Actual_202-Colossus",
    "Emery_1002",
    "NC21_1001",
    "Sullivan_1005"
]

# Define reps within each variety
reps = ["rep1", "rep2", "rep3"]


# Iterate over each variety and process its raw to jpg conversion
for variety in varieties: 
    for rep in reps:
        filtered_max_dims = {}
        max_dims = {}
        class_counts = {}
        y_pred_counts = {}
        data_path = os.path.join(base_folder, jpg_base, "varieties", variety, rep)
        print(data_path)
        output_path = os.path.join(results_folder, "tables", "varieties", variety, rep)
        plot_path = os.path.join(results_folder, "plots", "varieties", variety, rep)
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
        #X.to_csv(f'{output_path}/{rep}.csv', index=False)  
        
        
        
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
        
#%% Visualizations

        
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
        class_df.to_csv(f'{output_path}/grades_podtracker_{rep}.csv')
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Predicted Classes', y='Count', data=class_df)
        plt.xlabel('Predicted Classes')
        plt.ylabel('Count')
        plt.title(f'Predicted Classes. Variety: {variety}. Rep: {rep}')

        plt.tight_layout()
        plt.savefig(f'{plot_path}/barplot.png', dpi=400)
        #plt.show()
        #plt.clf()
        plt.close()
        
        
        
        
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
        plt.title(f'Density plots. Variety: {variety}. Rep: {rep}') # <- for grade analysis
        plt.tight_layout()
        plt.savefig(f'{plot_path}/density.png', dpi=400)
        #plt.show()
        #plt.clf()
        plt.close()
        
#%%

# Initialize an empty DataFrame
combined_df = pd.DataFrame()
podtracker_df = pd.DataFrame()
deepsortmask_df = pd.DataFrame()

# Directory where your CSV files are located
output_path = os.path.join(results_folder, 'tables/varieties/')
# Iterate over directories and files
for root, dirs, files in os.walk(output_path):
    
    # Extract variety and rep from directory path
    rep = os.path.basename(root)
    variety = os.path.basename(os.path.dirname(root))
    for file in files:
        if file.startswith('grades_podtracker'):
            
            # Read CSV file
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Add variety and rep columns
            df['variety'] = variety
            df['rep'] = rep
            df['tracker'] = 'PodTracker'
            
            # Append to podtracker_df
            podtracker_df = pd.concat([podtracker_df, df], ignore_index=True)
        
        if file.startswith('grades_deepsortmask'):

            # Read CSV file
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            
            # Add variety and rep columns
            df['variety'] = variety # <- for grade analysis
            df['rep'] = rep
            df['tracker'] = 'DeepSortMask'
            
            # Append to deepsortmask_df
            deepsortmask_df = pd.concat([deepsortmask_df, df], ignore_index=True)

combined_df = pd.concat([podtracker_df, deepsortmask_df])
# Convert 'rep' column to categorical
combined_df['rep'] = pd.Categorical(combined_df['rep'])         

combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]   


      
# Get unique varieties
varieties = combined_df['variety'].unique()


# Grades Barplot

# Path to your manual counts CSV file
sizer_counts_path = os.path.join(base_folder, 'data/counts_data/presizer_manual_grade_counts.csv')

# Read the manual counts CSV file
sizer_counts = pd.read_csv(sizer_counts_path)

# Combine dataframes
new_combined_df = pd.concat([combined_df, sizer_counts])

# Remove DeepSortMask since its overcounting
new_combined_df = new_combined_df[new_combined_df['tracker']!='DeepSortMask']

# Calculate the standard deviation
std_dev_df = new_combined_df.groupby(['Predicted Classes', 'variety', 'tracker'])['Count'].std().reset_index()
std_dev_df.rename(columns={'Count': 'std_dev'}, inplace=True)

# Calculate the mean count
mean_counts_df = new_combined_df.groupby(['Predicted Classes', 'variety', 'tracker'])['Count'].mean().reset_index()

# Merge mean and standard deviation dataframes
plot_df = pd.merge(mean_counts_df, std_dev_df, on=['Predicted Classes', 'variety', 'tracker'])

# Pivot the DataFrame to get the desired format
pivot_df = plot_df.pivot_table(
    index=['variety', 'tracker'], 
    columns='Predicted Classes', 
    values=['Count', 'std_dev'],
    aggfunc={'Count': 'mean', 'std_dev': 'mean'}
).reset_index()

# Flatten the MultiIndex columns
pivot_df.columns = [' '.join(col).strip() for col in pivot_df.columns.values]

# Create columns with mean and standard deviation in the desired format
for cls in ['Jumbo', 'Fancy', 'No1']:
    pivot_df[f'{cls} mean (std)'] = (
        pivot_df[f'Count {cls}'].astype(str) + ' (' + 
        pivot_df[f'std_dev {cls}'].astype(str) + ')'
    )
    
# Select and order the final columns
final_columns = ['variety', 'tracker'] + [f'{cls} mean (std)' for cls in ['Jumbo', 'Fancy', 'No1']]
final_df = pivot_df[final_columns]

# Save the final DataFrame to a CSV file
table_csv_path = os.path.join(results_folder, 'tables/mean_std_by_class.csv')
#os.makedirs(os.path.dirname(table_csv_path), exist_ok=True)  # Create directory if it doesn't exist
final_df.to_csv(table_csv_path, index=False)

# Display the table for verification
print(final_df)



# Determine the global y-axis range
global_min = plot_df['Count'].min() - 1  # Padding for better visualization
global_max = plot_df['Count'].max() + 50  # Padding for better visualization


tracker_colors = [(128/255, 128/255, 128/255), (186/255, 9/255, 81/255), (6/255, 150/255, 104/255) ]

# Loop over each Predicted Class to generate individual plots
classes = new_combined_df['Predicted Classes'].unique()
for cls in classes:
    plt.figure(figsize=(12, 8))
    
    # Filter data for the current Predicted Class
    class_data = plot_df[plot_df['Predicted Classes'] == cls]
    
    # Create the barplot
    sns.barplot(
        data=class_data,
        x='variety',
        y='Count',
        hue='tracker',
        palette={'PodTracker': tracker_colors[1], 'DeepSortMask': tracker_colors[0], 'Sizer': tracker_colors[2]},
        ci=None,  # Disable confidence intervals (for error bars, use error bars directly)
        dodge=True  # Ensure bars for different trackers are next to each other
    )
    
    # Add error bars manually
    for i, variety in enumerate(class_data['variety'].unique()):
        subset = class_data[class_data['variety'] == variety]
        x = i - 0.21  # Adjust this offset if needed for multiple trackers
        for j, tracker in enumerate(subset['tracker'].unique()):
            mean_count = subset[subset['tracker'] == tracker]['Count'].values[0]
            std_dev = subset[subset['tracker'] == tracker]['std_dev'].values[0]
            plt.errorbar(
                x + j * 0.39,  # Adjust position for each tracker
                mean_count,
                yerr=std_dev,
                fmt='o',
                color='black',
                capsize=5
            )

    #plt.title(f'Counts by Variety and Tracker for {cls}')
    plt.xlabel('Variety')
    plt.ylabel('Count')
    plt.legend(title='Tracker')
    plt.xticks(rotation=45)
    #Set the y-axis limits to the global range
    plt.ylim(global_min, global_max)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(results_folder, f'plots/{cls}_barplot.png'), dpi=400)
    plt.savefig(os.path.join(results_folder, f'plots/{cls}_barplot.svg'), dpi=400, format='svg')
    plt.show()
    plt.clf()
    plt.close()
    
import matplotlib.image as mpimg

# Paths to your PNG image files
image_paths = [os.path.join(results_folder, 'plots/No1_barplot.png'),
               os.path.join(results_folder, 'plots/Fancy_barplot.png'), 
               os.path.join(results_folder, 'plots/Jumbo_barplot.png')]

# Read images
images = [mpimg.imread(path) for path in image_paths]

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Display each image in its respective subplot
for ax, img, path in zip(axs, images, image_paths):
    ax.imshow(img)
    #ax.set_title(path.split('/')[-1])  # Set title to the image file name
    ax.axis('off')  # Hide axes

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot
plt.savefig(os.path.join(results_folder, 'plots/predicted_grades_by_tracker.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(results_folder, 'plots/predicted_grades_by_tracker.svg'), dpi=400, bbox_inches='tight', format='svg')
# Show the plot
plt.show()

# Compute grade standard deviation

def compute_tracker_std_dev(df):
    # Ensure we have the relevant columns
    assert 'tracker' in df.columns
    assert 'Count' in df.columns
    assert 'Predicted Classes' in df.columns
    
    # Group by Tracker and Predicted Class, then calculate the standard deviation
    std_dev_df = df.groupby(['tracker', 'Predicted Classes'])['Count'].std().reset_index()
    std_dev_df.rename(columns={'Count': 'Standard_Deviation'}, inplace=True)
    
    return std_dev_df

# Compute standard deviation
std_dev_df = compute_tracker_std_dev(new_combined_df)

print(std_dev_df)
std_dev_df.to_csv(os.path.join(results_folder, 'tables/standard_deviation.csv'), index=False)