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
    def __init__(self, threshold=0.75):
        # Load the pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.threshold = threshold

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
        above_threshold = pred_scores > self.threshold
        pred_boxes = pred_boxes[above_threshold]
        pred_labels = pred_labels[above_threshold]
        pred_scores = pred_scores[above_threshold]

        objects = []

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            
            mask_in_box_coordinates, mask_in_image_coordinates = self.object_segmenter.get_object_segmentation_mask(image, box)

            objects.append({
                "box": box,
                "label": label,
                "score": score,
                "mask_in_box_coordinates": mask_in_box_coordinates,
                "mask_in_image_coordinates": mask_in_image_coordinates
            })

        return objects

    def visualize(self, objects, image):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        visualize_on_ax(ax, objects)

        plt.show()

def visualize_on_ax(ax, objects):
    for object in objects:
        label_str = COCO_INSTANCE_CATEGORY_NAMES[object['label'].item()]
        xmin, ymin, xmax, ymax = object['box']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{label_str} ({object["score"]:.2f})', bbox=dict(facecolor='yellow', alpha=0.5))
