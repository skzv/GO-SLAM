import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

LOCAL_INSTANCE_CATEGORY_NAMES = [
    'door', 'cabinet', 'refrigerator', 'window', 'chair', 'table', 
    'couch', 'bed', 'oven', 'tv'
]

def visualize_on_ax(ax, objects, image_shape):
    for obj in objects:
        xmin, ymin, xmax, ymax = obj['box']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{obj["label"]} ({obj["score"]:.2f})', bbox=dict(facecolor='yellow', alpha=0.5))
    ax.imshow(cv2.cvtColor(image_shape, cv2.COLOR_BGR2RGB))

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        return

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # Your custom number of classes
    cfg.MODEL.WEIGHTS = './pretrained/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

    try:
        predictor = DefaultPredictor(cfg)
        print("Predictor created successfully.")
    except Exception as e:
        print("Error creating predictor:", e)
        return

    #door fail
    # image_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/room0/results/frame000280.jpg'
    #couch
    image_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/room0/results/frame001673.jpg'
    # Door
    # image_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/room0/results/frame000540.jpg'
    # image_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/office0/results/frame001307.jpg'
    # image_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/room2/results/frame001913.jpg'
    # image_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/room1/results/frame001735.jpg'
    # image_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/office0/results/frame001342.jpg'

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    print("Image loaded successfully")

    try:
        outputs = predictor(image)
        print("Inference successful")
        print("Outputs:", outputs)
        
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes
        pred_classes = instances.pred_classes
        pred_scores = instances.scores

        objects = []
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            objects.append({
                "box": box.cpu().numpy(),
                "label": LOCAL_INSTANCE_CATEGORY_NAMES[cls],
                "score": score.item()
            })

        fig, ax = plt.subplots(1, figsize=(12, 9))
        visualize_on_ax(ax, objects, image)
        plt.show()
        
    except Exception as e:
        print("Error during inference:", e)

if __name__ == "__main__":
    main()