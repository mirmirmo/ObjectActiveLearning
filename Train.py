import os
import random
import shutil
import numpy as np
import PIL.Image
from google.colab import files
from ultralytics import YOLOWorld
from ultralytics import YOLO
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import argparse

# from requirements import *
# from Data import *
from Create_yaml import *
from Augments import *
from ActiveLearning import *

create_yaml()
parser = argparse.ArgumentParser(description="Active Learning with YOLO and Image Blackout")
parser.add_argument("--cycles", type=int, default=9, help="Number of active learning cycles")
parser.add_argument("--num_samples", type=int, default=50, help="Number of images to sample first cycle")
parser.add_argument("--num_samples2", type=int, default=10, help="Number of images to sample per cycle")
parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters for KMeans")
parser.add_argument("--model_path", type=str, default="yolov8x-worldv2.pt", help="Path to the YOLO model")
parser.add_argument("--train_epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--output_image_dir", type=str, default="/content/datasets/VOC_l/images/train2012", help="Directory for saving output images")
parser.add_argument("--output_label_dir", type=str, default="/content/datasets/VOC_l/labels/train2012", help="Directory for saving output labels")
parser.add_argument("--weight1", type=float, default=0.95, help="weight for image uniformity")
parser.add_argument("--weight2", type=float, default=0.05, help="weight for pool uniformity")

args = parser.parse_args()

import os
os.environ['WANDB_MODE'] = 'disabled'
# torch.cuda.set_device(0)
import time

# Seed the random number generator with the current time
random.seed(time.time())# torch.manual_seed(2)# torch.manual_seed(3)
# torch.cuda.manual_seed(3)
# torch.cuda.manual_seed_all(3)
# np.random.seed(2)

cycles = args.cycles
model = None  # Initialize the model variable

# Function to process labels and image blackout
def blackout_half_image(image, labels_dir, output_image_dir, output_label_dir, confidences, width, height):
    # Create a copy of the image to modify (blackout one side)
    img_black = image.copy()

    # Compute the average confidences for left, right, top, and bottom quadrants
    left_conf = 0
    right_conf = 0
    top_conf = 0
    bottom_conf = 0

    # Calculate average confidences for left, right, top, and bottom halves
    for row in range(20):
        left_conf += confidences[row * 20: row * 20 + 10].mean()   # Avg confidence for the left half
        right_conf += confidences[row * 20 + 10: row * 20 + 20].mean()  # Avg confidence for the right half
        if row < 10:  # Top half (rows 0-9)
            top_conf += confidences[row * 20: (row + 1) * 20].mean()
        else:  # Bottom half (rows 10-19)
            bottom_conf += confidences[row * 20: (row + 1) * 20].mean()

    # Calculate the overall average confidence for the image (all regions combined)
    overall_avg_conf = confidences.mean()

    # Store the quadrant confidence values
    quadrant_confs = {"left": left_conf, "right": right_conf, "top": top_conf, "bottom": bottom_conf}

    # Find the quadrant with the lowest confidence (below the overall average)
    lowest_conf = float('inf')  # Initialize with a very high value
    blackout_side = None

    # Determine which quadrant has the lowest confidence among those below the overall average
    for side, conf in quadrant_confs.items():
        avg_conf = conf / 20  # Normalize by dividing by number of rows
        if avg_conf < overall_avg_conf and avg_conf < lowest_conf:
            lowest_conf = avg_conf
            blackout_side = side

    # Blackout the selected quadrant if any side has lower confidence
    if blackout_side == "left":
        img_black.paste((0, 0, 0), [0, 0, width // 2, height])  # Blackout left half
    elif blackout_side == "right":
        img_black.paste((0, 0, 0), [width // 2, 0, width, height])  # Blackout right half
    elif blackout_side == "top":
        img_black.paste((0, 0, 0), [0, 0, width, height // 2])  # Blackout top half
    elif blackout_side == "bottom":
        img_black.paste((0, 0, 0), [0, height // 2, width, height])  # Blackout bottom half

    # Save the modified image
    img_name = os.path.basename(image.filename)
    img_black.save(os.path.join(output_image_dir, img_name))

    # Modify the labels accordingly
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_name)
    if not os.path.exists(label_path):
        return

    with open(label_path, 'r') as file:
        lines = file.readlines()

    new_labels = []
    for line in lines:
        cls, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

        # Retain labels based on blackout side
        if blackout_side == "left" and x_center > 0.5:
            new_labels.append(line)
        elif blackout_side == "right" and x_center < 0.5:
            new_labels.append(line)
        elif blackout_side == "top" and y_center > 0.5:
            new_labels.append(line)
        elif blackout_side == "bottom" and y_center < 0.5:
            new_labels.append(line)

    # Write the modified labels to the output label directory
    with open(os.path.join(output_label_dir, label_name), 'w') as label_file:
        label_file.writelines(new_labels)

    return img_black  # Return the modified image for visualization or further processing


# Plotting the original and blacked-out images
def plot_images(original_img, blacked_out_img, index):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original_img)
    ax[0].set_title(f'Original Image {index+1}')
    ax[0].axis('off')

    ax[1].imshow(blacked_out_img)
    ax[1].set_title(f'Blacked-Out Image {index+1}')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# Define the softmax function with temperature
def softmax_with_temperature(logits, temperature):
    exp_logits = np.exp(logits / temperature)
    return exp_logits / np.sum(exp_logits)

def write_file_names_to_text(directory_path, output_file):
    # Get the list of file names in the directory
    file_names = os.listdir(directory_path)

    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Write each file name to the text file
        for file_name in file_names:
            f.write(file_name + '\n')

# Dictionary to store the outputs
layer_outputs = {}
confidences = {}

model = YOLOWorld(args.model_path)  # or select yolov8m/l-world.pt for different sizes
# Hook function to store the output of the layer
def hook_fn(module, input, output):
    layer_outputs['conv2d_output'] = output
def hook_fn2(module, input, output):
    if isinstance(output, tuple):
        confidences['conv2d_output'] = output[0]
    else:
        confidences['conv2d_output'] = output



target_layer = model.model.model[-1].cv3[2][2]  # Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1)) layer
target_layer2 = model.model.model[-1]  # The final layer or block

hook_handle = target_layer.register_forward_hook(hook_fn)
hook_handle2 = target_layer2.register_forward_hook(hook_fn2)

# Transformation for the images
transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)  # Move the model to the appropriate device


def get_class_counts(label_dir, class_names):
    # Initialize a dictionary to store class counts
    class_counts = {class_name: 0 for class_name in class_names}

    # Iterate over each label file in the directory
    for filename in os.listdir(label_dir):
        filepath = os.path.join(label_dir, filename)
        # Open the label file
        with open(filepath, 'r') as file:
            # Read each line in the file
            for line in file:
                # The class ID is the first value in each line
                class_id = int(line.split()[0])
                class_name = class_names[class_id]
                class_counts[class_name] += 1

    return class_counts

# List of class names corresponding to the class IDs
class_names2 = ['airplane','ship','storage tank','baseball diamond','tennis court','basketball court','ground track field','harbor','bridge','vehicle']
for c in range(cycles):
    # Path to the directory containing images and labels for training
    image_dir = "/content/datasets/VOC/images/train2012"
    label_dir = "/content/datasets/VOC/labels/train2012"

    # Path to the directory where you want to move the sampled images and labels
    output_image_dir = args.output_image_dir
    output_label_dir = args.output_label_dir

    # Create directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Number of images to sample
    num_samples = args.num_samples

    # Get a list of all image files in the directory
    image_files = os.listdir(image_dir)
    from scipy.special import softmax  # Import the softmax function
    if c == 0:
        sampled_images = []
        sampled_frequencies = []  # To keep track of summed frequencies of sampled images

        model.set_classes([
            'airplane,ship,storage tank,baseball diamond,tennis court,basketball court,ground track field,harbor,bridge,vehicle'
        ])
        # Extract features and confidence scores
        features_list = []
        confidences_list = []
        for image_file in image_files:
            
            img_path = os.path.join(image_dir, image_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                model.model(img_tensor)

                features = layer_outputs['conv2d_output'].cpu().numpy()
                features_list.append(features)
                confidences_array = confidences['conv2d_output'].cpu().numpy()
                confidences_list.append(confidences_array)

        confidences_array = np.array(confidences_list)
        confidences_array = confidences_array[:, :, 4, :400]

        features_array = np.concatenate(features_list, axis=0)
        features_array = features_array.reshape(features_array.shape[0], features_array.shape[1], -1)
        features_array = features_array.transpose(0, 2, 1).reshape(-1, features_array.shape[1])
        # Perform clustering
        num_clusters = args.num_clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(features_array)
        region_labels = kmeans.labels_.reshape(len(image_files), 400)

        # Calculate cluster frequencies
        cluster_frequencies = []
        for i in range(len(image_files)):
            counts = np.bincount(region_labels[i], minlength=num_clusters)
            cluster_frequencies.append(counts)

        # Calculate standard deviation scores for all images
        std_devs = [np.std(freq) for freq in cluster_frequencies]
        std_devs_scores = softmax(-np.array(std_devs))  # Use negative std_devs to prioritize smaller values

        # Select the first image
        selected_index = np.argmax(std_devs_scores)  # Select the image with the highest score
        sampled_images.append(image_files[selected_index])
        sampled_frequencies.append(cluster_frequencies[selected_index])

        # Select subsequent images
        while len(sampled_images) < 50:
            # Compute combined std_devs for all unselected images
            combined_std_devs = []
            for i, freq in enumerate(cluster_frequencies):
                if image_files[i] in sampled_images:
                    combined_std_devs.append(float('inf'))  # Exclude already selected images
                    std_devs_scores[i]=0
                else:
                    combined_frequency = np.sum([*sampled_frequencies, freq], axis=0)
                    combined_std_dev = np.std(combined_frequency)
                    combined_std_devs.append(combined_std_dev)

            # Apply softmax on the negative of combined standard deviations
            combined_scores = softmax(-np.array(combined_std_devs))

            # Combine `std_devs_scores` and `combined_scores` to form a new score
            total_scores = args.weight1 * std_devs_scores + args.weight2 * combined_scores

            # Select the image with the highest combined score
            selected_index = np.argmax(total_scores)
            sampled_images.append(image_files[selected_index])

            # print("scores: " , std_devs_scores[selected_index] , combined_scores[selected_index] , 0.5 * std_devs_scores[selected_index] + 0.5 * combined_scores[selected_index])

            sampled_frequencies.append(cluster_frequencies[selected_index])
        print(len(sampled_images))
        # Apply blackout on sampled images
        for i, sampled_image in enumerate(sampled_images):
            image_path = os.path.join(image_dir, sampled_image)
            img = Image.open(image_path)
            width, height = img.size
            region_confs = confidences_array[selected_index, 0]
            blacked_out_img = blackout_half_image(img, label_dir, output_image_dir, output_label_dir, region_confs, width, height)
            if i < 5:
                plot_images(img, blacked_out_img, i)



    else:
            num_samples = args.num_samples2
            sampled_images = []
            sampled_image_files = set()  # Set to track previously sampled images

            # Dictionary to store the uncertainty scores for each image
            uncertainty_scores = {}

            # Calculate uncertainty scores for each image
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_dir, image_file)
                image = Image.open(image_path)

                uncertainty_score = get_uncertainty(model, image)

                uncertainty_scores[image_file] = uncertainty_score

                # Print progress every 500 files
                if i % 500 == 0:
                    print(f"Processed {i} files out of {len(image_files)}")

            # Sort the images based on their uncertainty scores
            sorted_images = sorted(uncertainty_scores.items(), key=lambda x: x[1])

            # Sample images with the least uncertainty that have not been selected before
            for image_file, _ in sorted_images:
                dest_image_path = os.path.join(output_image_dir, image_file)
                dest_label_path = os.path.join(output_label_dir, image_file.replace(".jpg", ".txt"))

                # Check if the image or label already exists; if so, skip to the next
                if os.path.exists(dest_image_path) or os.path.exists(dest_label_path):
                    print(f"Skipping {image_file}, as it already exists in the destination.")
                    continue

                # Add the image to the sampled list
                sampled_images.append(image_file)
                sampled_image_files.add(image_file)  # Mark this image as selected

                if len(sampled_images) >= num_samples:
                    break  # Stop once we have sampled the required number of images

            # `sampled_images` now c

    # Move sampled images and their corresponding labels to the output directory
    print("hi")
    print("bye")
    for image_file in sampled_images:
        # Move image
        image_path = os.path.join(image_dir, image_file)
        dest_image_path = os.path.join(output_image_dir, image_file)

        # Check if the file already exists in the destination folder
        if not os.path.exists(dest_image_path):
            shutil.move(image_path, dest_image_path)
        else:
            print(f"Image {image_file} already exists in {output_image_dir}")

        # Extract corresponding label file name
        label_file = image_file.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_file)
        dest_label_path = os.path.join(output_label_dir, label_file)

        # Check if the label file already exists in the destination folder
        if not os.path.exists(dest_label_path):
            shutil.move(label_path, dest_label_path)

    print("Number of images: ", len([file for file in os.listdir(output_image_dir) if file.endswith(".jpg")]))
    print("Number of label : ", len([file for file in os.listdir(output_label_dir) if file.endswith(".txt")]))

    total_objects = 0
    # Iterate over each file in the directory
    for filename in os.listdir(output_label_dir):
        filepath = os.path.join(output_label_dir, filename)
        # Open the label file
        with open(filepath, 'r') as file:
            # Count the number of lines in the file
            object_count = sum(1 for line in file)
            # Add the count to the total
            total_objects += object_count
    print("Total number of objects:", total_objects)

    class_counts = get_class_counts(output_label_dir, class_names2)
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    print("*********************************************************")
    # Train the model
    model = YOLO("yolov8n.pt")  # build a new model from YAML
    results = model.train(data='VOC_2012.yaml', epochs=args.train_epochs ,plots=True)
