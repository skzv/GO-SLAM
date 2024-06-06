
import cv2
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
        image_path = self.data_loader.get_frame_path(index)
        image = cv2.imread(image_path)
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

    
    # Function to create a text mesh using trimesh
    def render_text(self, text, position, scale=0.01):
        color = [1.0, 0.0, 0.0]

        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=2).to_legacy()

        # Flip the mesh vertically (invert the y-coordinates)
        flip_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        text_mesh.transform(flip_transform)

        text_mesh.scale(scale, center=text_mesh.get_center())
        text_mesh.translate(position, relative=False)
        text_mesh.paint_uniform_color(color)


        return text_mesh

    # def render_text(self, text, position, font_size=50):
    #     # Define text and its properties
    #     text = "Hello, Open3D!"
    #     position = [0, 0, 0]
    #     font_size = 50  # Adjust the font size as needed

    #     # Create Text3D object
    #     text_3d = o3d.geometry.Text3D(text, pos=position, font_size=font_size)
    #     return text_3d
    

    def visualize_objects_with_mesh_for_frame(self, index, objects):
        camera_pose = self.camera_poses[index]
        self.visualize_objects_with_mesh(index, objects, camera_pose)

    def visualize_objects_with_mesh(self, objects, camera_pose=None, remove_walls=False):
        # object keys:
        # "box"
        # "label"
        # "score"
        # "mask_in_box_coordinates"
        # "mask_in_image_coordinates"

        # remove walls for vis
        room_mesh = o3d.geometry.PointCloud(self.mesh)

        if remove_walls:
            num_planes = 3
            inliers_list = []
            for _ in range(num_planes):  # We need to find the two largest planes
                plane_model, inliers = room_mesh.segment_plane(distance_threshold=0.1,
                                                        ransac_n=3,
                                                        num_iterations=5000)
                
                inliers_list.append(inliers)
                
                # Remove the inliers to find the next plane
                room_mesh = room_mesh.select_by_index(inliers, invert=True)

            # add back floor
            room_mesh = o3d.geometry.PointCloud(self.mesh)
            room_mesh = room_mesh.select_by_index([*inliers_list[0], *inliers_list[2]], invert=True)

        object_pcds = []
        text_labels = []

        for object in objects:
            world_coordinates = object['world_coordinates_cm']/100

            pcd = geometry_utils.flattened_coordinates_to_point_cloud(world_coordinates)
            # give each object a different color
            pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
            text_pos = pcd.get_center()
            text_pos[0] -= 0.25
            text_pos[1] -= 0.25
            text_pos[2] -= 0.25
            text_labels.append(self.render_text(object["label"], text_pos))
            object_pcds.append(pcd)

        origin = o3d_utils.draw_origin()

        # Create a visualizer and add the text mesh
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # vis.add_geometry(text_labels[0])
        if camera_pose is None:
            camera_pose = o3d.geometry.PointCloud()
        for geo in [origin, room_mesh, camera_pose, *object_pcds, *text_labels]:
            vis.add_geometry(geo)
        # vis.add_geometry([origin, self.mesh, camera_pose, *object_pcds, *text_labels])

        # Set up lighting to enhance the color visibility
        opt = vis.get_render_option()
        # opt.light_on = True
        # opt.background_color = np.asarray([0.0, 0.0, 0.0])  # Set background to black
        opt.mesh_show_back_face = True  # Ensure back faces are rendered

        # Render the scene
        vis.run()
        vis.destroy_window()

        # o3d.visualization.draw_geometries([origin, self.mesh, camera_pose, *object_pcds, *text_labels])

    def visualize_frame_depth_map_with_mesh(self, index):
        
        image = np.zeros((self.H, self.W, 3))

        depth_map = self.depths[index]
        c2w = self.c2ws[index]
        camera_pose = self.camera_poses[index]

        world_coordinates = geometry_utils.pixel_to_world_coordinates(image, depth_map, self.intrinsic_matrix, c2w)

        pcd = geometry_utils.world_coordinates_to_point_cloud(world_coordinates)
        origin = o3d_utils.draw_origin()

        o3d.visualization.draw_geometries([origin, self.mesh, camera_pose, pcd])