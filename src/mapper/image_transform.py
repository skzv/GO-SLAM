import cv2
import numpy as np


class ImageTransform:
    def __init__(self, H_out, W_out, H_edge, W_edge):
        self.H_out = H_out
        self.W_out = W_out
        self.H_edge = H_edge
        self.W_edge = W_edge

    def transform_image(self, image):

        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        image = cv2.resize(image, (W_out_with_edge, H_out_with_edge))

        # crop image edge
        if self.W_edge > 0:
            edge = self.W_edge
            image = image[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            image = image[:, :, edge:-edge, :]

        return image
    
    def transform_object_box(self, object_box, original_image_size):
        x1, y1, x2, y2 = object_box
        H_orig, W_orig, _ = original_image_size
        
        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2

        # Calculate scale factors
        scale_x = W_out_with_edge / W_orig
        scale_y = H_out_with_edge / H_orig

        # Scale the bounding box
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y

        # Account for edge cropping
        x1_cropped = x1_scaled - self.W_edge
        y1_cropped = y1_scaled - self.H_edge
        x2_cropped = x2_scaled - self.W_edge
        y2_cropped = y2_scaled - self.H_edge

        # Ensure the coordinates remain within the new image bounds
        x1_final = max(0, x1_cropped)
        y1_final = max(0, y1_cropped)
        x2_final = min(self.W_out, x2_cropped)
        y2_final = min(self.H_out, y2_cropped)

        return (x1_final, y1_final, x2_final, y2_final)

    def transform_object_boxes(self, object_boxes, original_image_size):
        transformed_objects = []
        for object in object_boxes:
            transformed_object = self.transform_object_box(object, original_image_size)
            transformed_objects.append(transformed_object)
        return transformed_objects