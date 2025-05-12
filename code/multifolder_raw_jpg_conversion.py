import os
import cv2
import numpy as np

def is_folder_already_converted(raw_folder, jpg_folder):
    """
    Check if all RAW files in the source folder already have corresponding JPG files.
    Returns True if all RAW files have been converted, False otherwise.
    """
    # Ensure jpg folder exists
    if not os.path.exists(jpg_folder):
        return False
    
    # Get list of raw files
    raw_files = [f for f in os.listdir(raw_folder) if f.lower().endswith('.raw')]
    
    if not raw_files:
        print(f"No RAW files found in {raw_folder}")
        return True  # Consider it done if there are no raw files
    
    # Check if each raw file has a corresponding jpg
    for raw_file in raw_files:
        jpg_file = raw_file[:-3] + "jpg"
        jpg_path = os.path.join(jpg_folder, jpg_file)
        
        if not os.path.exists(jpg_path):
            return False  # Found at least one RAW file without a JPG version
    
    return True  # All RAW files have corresponding JPG files


def raw_to_jpg(rawdatafolder, jpgdatafolder):
    rawdir = os.listdir(rawdatafolder)
    
    for filename in rawdir:
        # Skip files that are not raw image files based on extension
        if not filename.lower().endswith('.raw'):
            continue
        
        # Check if JPG already exists (for individual file skipping)
        jpg_path = os.path.join(jpgdatafolder, filename[:-3] + "jpg")
        if os.path.exists(jpg_path):
            continue  # Skip this file if JPG already exists
            
        raw_file_path = os.path.join(rawdatafolder, filename)
    
        # Reshape bayer image
        with open(raw_file_path, 'rb') as fileID:
            out = np.fromfile(fileID, dtype=np.uint8)
            Xsize = 1080//2
            Ysize = 1440//2
            try:
                bayer_im = np.reshape(out, (2*Xsize, 2*Ysize))
            except ValueError:
                continue
            
            # Define the channels
            c1 = bayer_im[::2, ::2]
            c2 = bayer_im[::2, 1::2]
            c3 = bayer_im[1::2, ::2]
            c4 = bayer_im[1::2, 1::2]
            
            # Interpolate channels
            c1i = cv2.resize(c1, (1440, 1080), interpolation=cv2.INTER_LINEAR)
            c2i = cv2.resize(c2, (1440, 1080), interpolation=cv2.INTER_LINEAR)
            c3i = cv2.resize(c3, (1440, 1080), interpolation=cv2.INTER_LINEAR)
            c4i = cv2.resize(c4, (1440, 1080), interpolation=cv2.INTER_LINEAR)
            
            # Merge channels 
            RGB = cv2.merge([c4i, c3i, c1i])
            RGB = RGB.astype(np.uint8)
            
            cv2.imwrite(jpg_path, RGB)


# Define base directories
base_folder = os.path.expanduser("~/PodTracker_study/data")
raw_base = "raw_data"
jpg_base = "jpg_data"
grade_base = "grade_data"
count_base = "counts_data"


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
          "300_count"]

# Define reps within each variety
reps = ["rep1", "rep2", "rep3"]


# Process both varieties and counts
for dataset_type, folders, base_type in [
    ("varieties", varieties, grade_base),  # For variety processing
    ("counts", counts, count_base)         # For count processing
]:
    for folder in folders:
        for rep in reps:
            # Set up paths based on which type we're processing
            if dataset_type == "varieties":
                raw_folder = os.path.join(base_folder, base_type, raw_base, "varieties", folder, rep)
                jpg_folder = os.path.join(base_folder, base_type, jpg_base, "varieties", folder, rep)
                item_name = folder  # variety name
            else:  # counts
                raw_folder = os.path.join(base_folder, base_type, raw_base, folder, rep)
                jpg_folder = os.path.join(base_folder, base_type, jpg_base, folder, rep)
                item_name = folder  # count name
            
            # Skip if raw folder doesn't exist
            if not os.path.exists(raw_folder):
                print(f"Skipping {dataset_type} {item_name} {rep} - RAW folder not found")
                continue
                
            # Create jpg folder if it doesn't exist
            if not os.path.exists(jpg_folder):
                os.makedirs(jpg_folder)
            
            # Check if folder is already fully converted
            if is_folder_already_converted(raw_folder, jpg_folder):
                print(f"Skipping {dataset_type} {item_name} {rep} - Already converted")
                continue
                
            print(f"Working on {dataset_type} {item_name} {rep}")
            raw_to_jpg(raw_folder, jpg_folder)
            
            # Verify conversion
            if is_folder_already_converted(raw_folder, jpg_folder):
                print(f"Successfully converted all files in {dataset_type} {item_name} {rep}")
            else:
                print(f"Warning: Some files may not have been converted in {dataset_type} {item_name} {rep}")