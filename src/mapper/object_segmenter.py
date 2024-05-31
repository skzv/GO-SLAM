import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, build_sam

class ObjectSegmenter:
    def __init__(self):
        # Load the SAM model
        path = "./pretrained/sam_vit_h_4b8939.pth"
        #self.model = sam_model_registry["default"](checkpoint=path)
        # self.model = build_sam(model_name)
        #self.mask_generator = SamAutomaticMaskGenerator(self.model)


    def get_object_segmentation_mask(self, image, box):
        # image = Image.open(image_path).convert("RGB")

        # assume largest mask is the object of interest

        # TODO: get correct segementation mask using SAM model
        # for now, return the whole box as the segmentation mask
        # convert box to COCO mask
        xmin, ymin, xmax, ymax = box
        # Round to integer
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        box_width, box_height = xmax - xmin, ymax - ymin

        # Convert box to COCO RLE format
        mask_in_image_coordinates = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask_in_image_coordinates[ymin:ymax, xmin:xmax] = 1

        mask_in_box_coordinates = np.ones((box_height, box_width))
        return mask_in_box_coordinates, mask_in_image_coordinates