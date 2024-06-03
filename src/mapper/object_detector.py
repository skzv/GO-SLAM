from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import object_segmenter

LOCAL_INSTANCE_CATEGORY_NAMES = [
    'door', 'cabinet', 'refrigerator', 'window', 'chair', 'table', 
    'couch', 'bed', 'oven', 'tv'
]

class ObjectDetector:
    def __init__(self, model_path='./pretrained/model_final.pth', threshold=0.75):
        # Configuration setup
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LOCAL_INSTANCE_CATEGORY_NAMES)  # Number of classes
        self.cfg.MODEL.WEIGHTS = model_path  # Path to the trained model weights
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Set the testing threshold for this model
        self.predictor = DefaultPredictor(self.cfg)

        self.object_segmenter = object_segmenter.ObjectSegmenter()

    def detect(self, image, plot=False):
        # Perform inference
        outputs = self.predictor(image)
        print("Outputs:", outputs)

        # Get the prediction results
        pred_boxes = outputs["instances"].pred_boxes
        pred_labels = outputs["instances"].pred_classes
        pred_scores = outputs["instances"].scores
        # print("Output   ",pred_boxes,pred_labels,pred_scores)

        objects = []

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            mask_in_box_coordinates, mask_in_image_coordinates = self.object_segmenter.get_object_segmentation_mask(image, box)

            objects.append({
                "box": box.cpu().numpy(),
                "label": LOCAL_INSTANCE_CATEGORY_NAMES[label.item()],
                "score": score.item(),
                "mask_in_box_coordinates": mask_in_box_coordinates,
                "mask_in_image_coordinates": mask_in_image_coordinates
            })

            print("Output//   ",LOCAL_INSTANCE_CATEGORY_NAMES[label.item()], score)

        if plot:
            fig, ax = plt.subplots(1, figsize=(12, 9))
            temp_visualize_on_ax(ax, objects, image)
            plt.show()


        return objects

    def filter_objects_by_threshold(self, objects, threshold):
        return [object for object in objects if object['score'] >= threshold]

    def visualize(self, objects, image):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image[:, :, ::-1])  # Convert BGR to RGB for displaying with matplotlib

        visualize_on_ax(ax, objects)

        plt.show()

def temp_visualize_on_ax(ax, objects, image_shape):
    for obj in objects:
        xmin, ymin, xmax, ymax = obj['box']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{obj["label"]} ({obj["score"]:.2f})', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.imshow(cv2.cvtColor(image_shape, cv2.COLOR_BGR2RGB))

def visualize_on_ax(ax, objects):
    for object in objects:
        xmin, ymin, xmax, ymax = object['box']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{object["label"]} ({object["score"]:.2f})', bbox=dict(facecolor='yellow', alpha=0.5))
