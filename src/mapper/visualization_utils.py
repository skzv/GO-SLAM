
from matplotlib import pyplot as plt
import numpy as np
import geometry_utils
import o3d_utils
import open3d as o3d
import camera_utils
import object_detector

class Visualizer:
    def __init__(self, data_loader, depths, mesh, c2ws, camera, image_transform):
        self.H = camera.H
        self.W = camera.W
        self.data_loader = data_loader
        self.depths = depths
        self.mesh = mesh
        self.c2ws = c2ws
        self.intrinsic_matrix = camera.intrinsic_matrix
        self.camera_poses = camera.visualize_all_camera_poses()
        self.image_transform = image_transform

    def draw_frame_and_estimated_depth_map(self, index, objects=None):

        depth_map = self.depths[index]
        # image = self.get_frame(index)
        image = self.data_loader.get_frame_path(index)
        resized_image = self.image_transform.transform_image(image)

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        axs[0].imshow(resized_image)
        axs[0].set_title('Frame')
        axs[0].axis('off')

        cax = axs[1].imshow(depth_map, cmap='viridis')
        fig.colorbar(cax, ax=axs[1])
        axs[1].set_title('Estimated Depth Map')
        axs[1].axis('off')

        if objects is not None:
            object_detector.visualize_on_ax(axs[0], objects)
            object_detector.visualize_on_ax(axs[1], objects)

        plt.show()

    def get_frame(self, index):
        path = self.data_loader.get_frame_path(index)
        return plt.imread(path)
    

    def visualize_merged_objects_with_mesh(self, merged_objects):
        # TODO: implement
        pass


    def visualize_objects_with_mesh(self, index, objects):
        # object keys:
        # "box"
        # "label"
        # "score"
        # "mask_in_box_coordinates"
        # "mask_in_image_coordinates"

        image = np.zeros((self.H, self.W, 3))

        depth_map = self.depths[index]
        c2w = self.c2ws[index]
        camera_pose = self.camera_poses[index]

        object_pcds = []

        for object in objects:
            mask = object["mask_in_image_coordinates"]

            world_coordinates = geometry_utils.pixel_to_world_coordinates(image, depth_map, self.intrinsic_matrix, c2w)

            pcd = geometry_utils.world_coordinates_to_point_cloud(world_coordinates, mask)
            # give each object a different color
            pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
            # TODO: label object with text
            object_pcds.append(pcd)

        origin = o3d_utils.draw_origin()
        o3d.visualization.draw_geometries([origin, self.mesh, camera_pose, *object_pcds])

    def visualize_frame_depth_map_with_mesh(self, index):
        
        image = np.zeros((self.H, self.W, 3))

        depth_map = self.depths[index]
        c2w = self.c2ws[index]
        camera_pose = self.camera_poses[index]

        world_coordinates = geometry_utils.pixel_to_world_coordinates(image, depth_map, self.intrinsic_matrix, c2w)

        pcd = geometry_utils.world_coordinates_to_point_cloud(world_coordinates)
        origin = o3d_utils.draw_origin()

        o3d.visualization.draw_geometries([origin, self.mesh, camera_pose, pcd])