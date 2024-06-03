import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import object_segmenter

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ObjectDetector:
    def __init__(self):
        # Load the pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        # move model to cuda
        self.model.eval()

        self.object_segmenter = object_segmenter.ObjectSegmenter()

    def detect(self, image):
        # # Load an image
        # image = Image.open(image_path).convert("RGB")

        # Transform the image to tensor
        image_tensor = F.to_tensor(image)

        # Add a batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Get the prediction results
        pred_boxes = predictions[0]['boxes']
        pred_labels = predictions[0]['labels']
        pred_scores = predictions[0]['scores']

        # Get where the score is above the threshold
        # above_threshold = pred_scores > self.threshold
        # pred_boxes = pred_boxes[above_threshold]
        # pred_labels = pred_labels[above_threshold]
        # pred_scores = pred_scores[above_threshold]

        objects = []

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            
            mask_in_box_coordinates, mask_in_image_coordinates = self.object_segmenter.get_object_segmentation_mask(image, box)
            label_str = COCO_INSTANCE_CATEGORY_NAMES[label.item()]

            objects.append({
                "box": box,
                "label": label_str,
                "score": score,
                "mask_in_box_coordinates": mask_in_box_coordinates,
                "mask_in_image_coordinates": mask_in_image_coordinates
            })

        return objects
    
    def filter_objects_by_threshold(self, objects, threshold):
        return [object for object in objects if object['score'] >= threshold]

    def visualize(self, objects, image):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        visualize_on_ax(ax, objects)

        plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def visualize_on_ax(ax, objects):
    for object in objects:
        xmin, ymin, xmax, ymax = object['box']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{object["label"]} ({object["score"]:.2f})', bbox=dict(facecolor='yellow', alpha=0.5))
        show_mask(object['mask_in_image_coordinates'], ax, random_color=True)
