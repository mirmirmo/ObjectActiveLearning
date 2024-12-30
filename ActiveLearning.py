import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.stats import entropy

def calculate_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # If the intersection is negative (no intersection), return 0
    if x2 < x1 or y2 < y1:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate the area of both input rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Union area by adding areas of both boxes and subtracting the intersection area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the Intersection over Union (IoU) by dividing the intersection area by the union area
    iou = intersection_area / union_area

    return iou


# Precompute the augmented images and boxes once outside the loop
def precompute_augmented_images(image_path, original_image_boxes):
    flip_image, flip_boxes = HorizontalFlip(image_path, original_image_boxes)
    cutout_image = cutout(image_path, original_image_boxes, 2)
    resize_image, resize_boxes = resize(image_path, original_image_boxes, 0.8)
    rot_image, rot_boxes = rotate(image_path, original_image_boxes, 5)

    return [flip_image, cutout_image, resize_image, rot_image], [flip_boxes, original_image_boxes, resize_boxes, rot_boxes]

def get_uncertainty(model, image_path):
    consistency1 = 0
    consistency2 = 0

    original_image_results = model(image_path, verbose=False)
    if len(original_image_results[0].boxes) == 0:
        return 2

    original_image_confs = original_image_results[0].boxes.conf
    original_image_boxes = original_image_results[0].boxes.xyxy
    original_image_labels = original_image_results[0].boxes.cls

    augs = ['flip', 'cut_out', 'smaller_resize', 'rotation']

    # Precompute augmented images and boxes
    augmented_images, augmented_boxes = precompute_augmented_images(image_path, original_image_boxes)

    for aug_image, aug_boxes, aug_name in zip(augmented_images, augmented_boxes, augs):
        aug_image_results = model(aug_image, verbose=False)
        aug_image_confs = aug_image_results[0].boxes.conf
        aug_image_boxes = aug_image_results[0].boxes.xyxy

        iou_max = 0
        for orig_box in aug_boxes:
            max_iou = 0
            for aug_box in aug_image_boxes:
                iou = calculate_iou(orig_box, aug_box)
                if iou > max_iou:
                    max_iou = iou
            iou_max += max_iou
        avg_iou_max = iou_max / len(aug_boxes)
        consistency1 += avg_iou_max

    # Classification uncertainty using Jensen-Shannon Divergence
    max_len = max(len(original_image_confs), len(aug_image_confs))
    original_image_confs_padded = np.pad(original_image_confs.cpu().numpy(), (0, max_len - len(original_image_confs)), mode='constant')
    aug_image_confs_padded = np.pad(aug_image_confs.cpu().numpy(), (0, max_len - len(aug_image_confs)), mode='constant')

    # Compute Jensen-Shannon Divergence
    p = (original_image_confs_padded + aug_image_confs_padded) / 2.0
    js_divergence = (entropy(original_image_confs_padded, p) + entropy(aug_image_confs_padded, p)) / 2.0
    consistency2 = 1 - js_divergence

    consistency1 /= len(augs)

    return consistency1 + consistency2

