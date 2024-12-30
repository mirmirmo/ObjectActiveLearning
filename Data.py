from google.colab import drive
drive.mount('/content/drive')

import os
import rarfile
dataset_path = '/content/drive/MyDrive/NWPU VHR-10 dataset.rar'

# Path to extract the dataset
extract_path = '/content/'

# Check if the dataset exists
if os.path.exists(dataset_path):
    # Open the RAR file
    with rarfile.RarFile(dataset_path, 'r') as rar:
        # Extract the contents to the specified path
        rar.extractall(extract_path)
        print("Dataset extracted successfully.")
else:
    print("Dataset not found at the specified path.")
    
    
import os
import shutil

# Directories
source_directory = "/content/NWPU VHR-10 dataset/negative image set"
ground_truth_directory = "/content/NWPU VHR-10 dataset/ground truth"
destination_directory = "/content/NWPU VHR-10 dataset/positive image set"

# Iterate over each file in the source directory
for filename in os.listdir(source_directory):
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(source_directory, filename)):
        # Split the filename and extension
        name, ext = os.path.splitext(filename)
        # Rename the file by adding 'n' to its name
        new_name = f"{name}n{ext}"
        # Rename the file
        os.rename(os.path.join(source_directory, filename), os.path.join(source_directory, new_name))
        # Create an empty text file with the new name in ground truth directory
        with open(os.path.join(ground_truth_directory, f"{new_name[0:4]}.txt"), "w"):
            pass
        # Move the file to the destination directory
        shutil.move(os.path.join(source_directory, new_name), destination_directory)

import os

def count_files(directory):
    # Initialize a counter for files
    file_count = 0

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is a regular file
        if os.path.isfile(os.path.join(directory, filename)):
            # Increment the file count
            file_count += 1

    return file_count

# Example usage:
directory_path = "/content/NWPU VHR-10 dataset/positive image set"
print("Number of files in directory:", count_files(directory_path))
directory_path = "/content/NWPU VHR-10 dataset/ground truth"
print("Number of files in directory:", count_files(directory_path))

import os
import numpy as np
import PIL.Image

def xyxy_to_xywh_normalized(bbox_list, image_width, image_height):
    normalized_bboxes = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        x_center_normalized = x_center / image_width
        y_center_normalized = y_center / image_height
        width_normalized = width / image_width
        height_normalized = height / image_height
        normalized_bboxes.append((x_center_normalized, y_center_normalized, width_normalized, height_normalized))
    return np.array(normalized_bboxes)

label_dir = '/content/NWPU VHR-10 dataset/ground truth'
image_dir = '/content/NWPU VHR-10 dataset/positive image set'

for file_name in os.listdir(label_dir):
    if file_name.endswith('.txt'):
        image_name = os.path.splitext(file_name)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, file_name)
        if os.path.exists(image_path):
            image = PIL.Image.open(image_path)
            image_width, image_height = image.size

            # Read original annotations
            with open(label_path, 'r') as file:
                lines = file.readlines()
                original_annotations = [line.strip() for line in lines]

            # Write normalized annotations back to the same file
            with open(label_path, 'w') as file:
                for line in original_annotations:
                    data = line.strip().replace('(', '').replace(')', '').split(',')
                    if len(data) == 5:  # Skip empty lines
                        data = [int(coord.strip()) for coord in data]
                        x1, y1, x2, y2, class_id = data
                        normalized_bbox = xyxy_to_xywh_normalized([(x1, y1, x2, y2)], image_width, image_height)
                        # file.write(f"{normalized_bbox[0][0]:.3f} {normalized_bbox[0][1]:.3f} {normalized_bbox[0][2]:.3f} {normalized_bbox[0][3]:.3f} {class_id}\n")
                        file.write(f"{class_id} {normalized_bbox[0][0]:.3f} {normalized_bbox[0][1]:.3f} {normalized_bbox[0][2]:.3f} {normalized_bbox[0][3]:.3f}\n")

import os

# Define the directory containing the label files
directory = "/content/NWPU VHR-10 dataset/ground truth/"

# Function to process a single label file
def process_label_file(file_path):
    modified_content = ""
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if len(parts) >= 1:
                    class_id = int(parts[0]) - 1  # Subtract 1 from the class ID
                    parts[0] = str(class_id)
                modified_content += ' '.join(parts) + '\n'

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)
        # print("Updated file:", file_path)

    except FileNotFoundError:
        print("File not found:", file_path)

# Process all label files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        process_label_file(file_path)

import os
import shutil
import random

random.seed(1)

# Define paths
image_source_dir = '/content/NWPU VHR-10 dataset/positive image set'
label_source_dir = '/content/NWPU VHR-10 dataset/ground truth'
train_image_dir = '/content/datasets/VOC/images/train2012'
train_label_dir = '/content/datasets/VOC/labels/train2012'
val_image_dir = '/content/datasets/VOC_l/images/val2012'
val_label_dir = '/content/datasets/VOC_l/labels/val2012'

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get list of image and label files
image_files = sorted(os.listdir(image_source_dir))
label_files = sorted(os.listdir(label_source_dir))

# Shuffle the files
random.shuffle(image_files)

# Move 600 random images to train set
for i in range(600):
    image_name = image_files[i]
    label_name = os.path.splitext(image_name)[0] + '.txt'

    # Move images
    shutil.move(os.path.join(image_source_dir, image_name), os.path.join(train_image_dir, image_name))

    # Move labels
    shutil.move(os.path.join(label_source_dir, label_name), os.path.join(train_label_dir, label_name))

# Move remaining images to validation set
for i in range(600, len(image_files)):
    image_name = image_files[i]
    label_name = os.path.splitext(image_name)[0] + '.txt'

    # Move images
    shutil.move(os.path.join(image_source_dir, image_name), os.path.join(val_image_dir, image_name))

    # Move labels
    shutil.move(os.path.join(label_source_dir, label_name), os.path.join(val_label_dir, label_name))
