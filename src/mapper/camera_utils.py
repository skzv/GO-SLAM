import open3d as o3d
import numpy as np


class Camera:
    def __init__(self, config, c2ws):
        self.config = config
        self.H = config['H']
        self.W = config['W']
        self.fx = config['fx']
        self.fy = config['fy']
        self.cx = config['cx']
        self.cy = config['cy']
        self.c2ws = c2ws
        self.intrinsic_matrix = self.create_intrinsic_matrix(self.fx, self.fy, self.cx, self.cy)

    def create_intrinsic_matrix(self, fx, fy, cx, cy):
        return np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0,   1]])

    def visualize_camera_pose(self, index, scale=1.5):
        """
        Visualize a camera pose with Open3D.
        
        :param intrinsic_matrix: Numpy array (3x3) representing the camera's intrinsic parameters.
        :param c2w_matrix: Numpy array (4x4) representing the camera to world transformation.
        :param scale: float representing the scale of the visualization objects.
        """
        # Create an Open3D camera intrinsic object
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(self.W, self.W, self.fx, self.fy, self.cx, self.cy)
        
        # Extrinsic parameters: Identity matrix for simplicity
        extrinsic_matrix = self.c2ws[index]
        # extrinsic_matrix[:3, 1] *= -1
        # extrinsic_matrix[:3, 2] *= -1

        # Create visualization
        camera_lineset = o3d.geometry.LineSet.create_camera_visualization(intrinsic_o3d, extrinsic_matrix, scale=scale)

        # Set all lines to red
        num_lines = np.asarray(camera_lineset.lines).shape[0]
        red_color = [1.0, 0, 0]  # RGB for red
        camera_lineset.colors = o3d.utility.Vector3dVector([red_color] * num_lines)
        
        return camera_lineset


    def visualize_all_camera_poses(self, scale=1.5):
        return [self.visualize_camera_pose(i, scale) for i in range(len(self.c2ws))]