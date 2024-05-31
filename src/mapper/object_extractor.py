import numpy as np
import geometry_utils


class ObjectExtractor:
    def __init__(self, c2ws, depths, camera):
        self.c2ws = c2ws
        self.depths = depths
        self.camera = camera
        self.H = camera.H
        self.W = camera.W

    def add_depth_to_objects(self, objects, index):
        image = np.zeros((self.H, self.W, 3))

        depth_map = self.depths[index]
        c2w = self.c2ws[index]

        for object in objects:
            mask = object["mask_in_image_coordinates"]

            world_coordinates = geometry_utils.pixel_to_world_coordinates(image, depth_map, self.camera.intrinsic_matrix, c2w)

            world_coordinates = geometry_utils.apply_mask_to_world_coordinates(world_coordinates, mask)
            object['world_coordinates'] = world_coordinates
            object['3d_box_world_coordinates'] = self.min_max_3d_box(world_coordinates)

        return objects

    def min_max_3d_box(self, pcd):
        # pcd has format (N, 3) => (x, y, z)
        min_x = np.min(pcd[:, 0])
        max_x = np.max(pcd[:, 0])
        min_y = np.min(pcd[:, 1])
        max_y = np.max(pcd[:, 1])
        min_z = np.min(pcd[:, 2])
        max_z = np.max(pcd[:, 2])

        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'min_z': min_z,
            'max_z': max_z
        }

    def merge_objects(self, objects_per_frame):
        """
            Merge objects with overlapping boxes and count from how many frames they were extracted. 
            Discard objects that were extracted from less than 3 frames.
            """
        # TODO: implement
        return objects_per_frame