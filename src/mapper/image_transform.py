import cv2
import numpy as np


class ImageTransform:
    def __init__(self, H_out, W_out, H_edge, W_edge):
        self.H_out = H_out
        self.W_out = W_out
        self.H_edge = H_edge
        self.W_edge = W_edge

    def transform_image(self, image_path):
        image = cv2.imread(image_path)

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
    
    def transform_object(self, object):
        # TODO: implement
        return object

    def transform_objects(self, objects):
        transformed_objects = []
        for object in objects:
            transformed_object = self.transform_object(object)
            transformed_objects.append(transformed_object)
        return transformed_objects