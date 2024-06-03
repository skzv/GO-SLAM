import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from segment_anything_fast import SamAutomaticMaskGenerator, sam_model_registry, build_sam, SamPredictor

class ObjectSegmenter:
    def __init__(self):
        # Load the SAM model
        # path = "./pretrained/sam_vit_h_4b8939.pth"
        path = "./pretrained/sam_vit_b_01ec64.pth"
        self.model = sam_model_registry["vit_b"](checkpoint=path)
        # self.model = build_sam(model_name)
        self.device = "cuda"
        self.model.to(device=self.device)
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=self.model,
        #     points_per_side=8,  # Further reduce to speed up computation
        #     points_per_batch=8,
        #     min_mask_region_area=10000,  # Set higher to filter out smaller regions
        # )
        self.predictor = SamPredictor(self.model)


    def get_object_segmentation_mask(self, image, box):
        image_np = np.array(image)
        # crop to box
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        width, height = xmax - xmin, ymax - ymin
        image_crop = image_np[ymin:ymax, xmin:xmax]

        # assume largest mask is the object of interest
        # print("extracting masks...")
        # masks = self.mask_generator.generate(image)
        self.predictor.set_image(image_crop)
        input_point = np.array([[int(width/2), int(height/2)]])
        input_label = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # print(masks)

        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(image_crop)
        #     show_mask(mask, plt.gca())
        #     show_points(input_point, input_label, plt.gca())
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        #     plt.axis('off')
        #     plt.show()  

        # Find the largest mask
        if masks is not None and len(masks) > 0:
            # get mask with highest score
            best_mask = masks[np.argmax(scores)]
            # Translate the mask coordinates to the original image coordinates
            translated_mask = self.translate_mask(image, best_mask, box)
        else:
            translated_mask = None

        # Plot the original image and the bounding box of the largest mask
        # plt.figure(figsize=(10,10))
        # plt.imshow(image_np)
        # show_mask(translated_mask, plt.gca())
        # input_point_tr = np.array([[int(width/2) + xmin, int(height/2) + ymin]])
        # show_points(input_point_tr, input_label, plt.gca())
        # rect_box = plt.Rectangle((xmin, ymin), width, height, edgecolor='b', facecolor='none')
        # plt.gca().add_patch(rect_box)
        # plt.show()

        return best_mask, translated_mask

        # # TODO: get correct segementation mask using SAM model
        # # for now, return the whole box as the segmentation mask
        # # convert box to COCO mask
        # # Round to integer
        # box_width, box_height = xmax - xmin, ymax - ymin

        # # Convert box to COCO RLE format
        # mask_in_image_coordinates = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # mask_in_image_coordinates[ymin:ymax, xmin:xmax] = 1

        # mask_in_box_coordinates = np.ones((box_height, box_width))
        # return mask_in_box_coordinates, mask_in_image_coordinates
    

    # Define a function to translate mask coordinates
    def translate_mask(self, original_image, mask, box):
        """Translate the mask coordinates to the original image coordinates"""
        x_min, y_min, _, _ = box
        # create an empty segmentation mask with original dims
        h, w, c = original_image.shape
        # create empty mask of all False
        empty_mask = np.zeros((h, w), dtype=bool)
        # copy the mask to the empty mask
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        mask_height, mask_width = mask.shape
        empty_mask[ymin:ymin+mask_height, xmin:xmin+mask_width] = mask
        # translated_mask = mask.copy()
        # translated_mask = np.roll(mask, shift=(y_min, x_min), axis=(0, 1))
        return empty_mask
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    