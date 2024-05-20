from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

# ply_file_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/mesh/final_raw_mesh.ply'
ply_file_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/office0/first-try/mesh/final_raw_mesh_forecast.ply'
room_mesh = o3d.io.read_point_cloud(ply_file_path)

# c2w_out = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/office0/first-try/checkpoints/est_poses.npy')
cfg = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/office0/first-try/mesh/cfg.npy')
c2w_est = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/office0/first-try/mesh/estimate_c2w_list.npy')
depths = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/mesh/depth_list.npy', allow_pickle=True)
depths_np = np.array([tensor.numpy() for tensor in depths])

# self.H, self.W, self.fx, self.fy, self.cx, self.cy
H, W, fx, fy, cx, cy = cfg
print(cfg)

# compare c2w_out and c2w_est
# print(c2w_out.shape)
# print(c2w_est.shape)
# print(c2w_out[500])
# print(c2w_est[500])

# TODO: load intrisinic matrix
# Taken from replica.yaml config
# fx = 600.0  # Focal length in x-direction
# fy = 600.0  # Focal length in y-direction
# cx = 599.5  # x-coordinate of the optical center
# cy = 339.5  # y-coordinate of the optical center

# Creating the intrinsic matrix
intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

print("c2w_est shape: ", c2w_est.shape)
print("depths shape: ", depths_np.shape)

# c2w: (N, 4, 4)
# depths_np: (N, H, W)

#print(np.min(depths0))
#print(np.max(depths0))

def draw_origin():
    # Create a coordinate frame (size can be adjusted)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
    return coordinate_frame

def visualize_camera_pose(intrinsic_matrix, c2w_matrix, scale=1.5):
    """
    Visualize a camera pose with Open3D.
    
    :param intrinsic_matrix: Numpy array (3x3) representing the camera's intrinsic parameters.
    :param c2w_matrix: Numpy array (4x4) representing the camera to world transformation.
    :param scale: float representing the scale of the visualization objects.
    """
    # Create an Open3D camera intrinsic object
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    width, height = 640, 320  # Adjust according to your actual camera resolution
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Extrinsic parameters: Identity matrix for simplicity
    extrinsic_matrix = c2w_matrix
    # extrinsic_matrix[:3, 1] *= -1
    # extrinsic_matrix[:3, 2] *= -1

    # Create visualization
    camera_lineset = o3d.geometry.LineSet.create_camera_visualization(intrinsic_o3d, extrinsic_matrix, scale=scale)

    # Set all lines to red
    num_lines = np.asarray(camera_lineset.lines).shape[0]
    red_color = [1.0, 0, 0]  # RGB for red
    camera_lineset.colors = o3d.utility.Vector3dVector([red_color] * num_lines)
    
    return camera_lineset
    
    # Visualize
    # o3d.visualization.draw_geometries([mesh_frame, frustum], window_name="Camera Pose Visualization")

def pixel_to_world_coordinates(image, depth_map, intrinsic_matrix, c2w_matrix):
    """
    Convert image pixels to world coordinates using depth map, camera intrinsic matrix, and camera to world matrix.

    :param image: 2D array or image matrix
    :param depth_map: 2D array where each value is the depth for the corresponding pixel in the image
    :param intrinsic_matrix: 3x3 camera intrinsic matrix
    :param c2w_matrix: 4x4 camera to world transformation matrix
    :return: World coordinates array of shape (H, W, 3)
    """
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create a grid of pixel coordinates
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the indices and depth values
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    depth_values = depth_map.flatten()

    # Create homogeneous coordinates for the pixels
    homogeneous_pixel_coordinates = np.stack([x_indices, y_indices, np.ones_like(x_indices)])

    # Compute the normalized camera coordinates
    normalized_camera_coordinates = np.linalg.inv(intrinsic_matrix) @ homogeneous_pixel_coordinates

    # Multiply each normalized coordinate by the corresponding depth
    x_cam = normalized_camera_coordinates[0, :] * depth_values
    y_cam = normalized_camera_coordinates[1, :] * depth_values
    z_cam = depth_values

    # Create camera space coordinates as homogeneous coordinates
    cam_coords_homogeneous = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)])

    # Transform camera space coordinates to world space coordinates
    world_coords_homogeneous = c2w_matrix @ cam_coords_homogeneous

    # Convert homogeneous world coordinates back to 3D
    world_coords = world_coords_homogeneous[:3, :] / world_coords_homogeneous[3, :]
    world_coords = world_coords.reshape(3, height, width).transpose(1, 2, 0)

    return world_coords

# Example usage
# intrinsic_matrix = np.array([[500, 0, 320],   # fx, 0, cx
#                              [0, 500, 240],   # 0, fy, cy
#                              [0, 0, 1]])      # Normalization factor

# c2w_matrix = np.array([[1, 0, 0, 1],         # Example transformation matrix
#                        [0, 1, 0, 2],
#                        [0, 0, 1, 3],
#                        [0, 0, 0, 1]])

camera_poses = [visualize_camera_pose(intrinsic_matrix, c2w_i) for c2w_i in c2w_est]

idx = 800
c2w_matrix = c2w_est[idx]
depth_map = 2.8 * depths_np[idx]
camera_pose = camera_poses[idx]

print(c2w_matrix)

# plot depthmap as image
plt.imshow(depth_map)
plt.colorbar()
plt.show()

# camera_pose = visualize_camera_pose(intrinsic_matrix, c2w_matrix)

# print(depths0.shape)

# # plot depth map as point cloud
# # Generate grid of pixel indices
# height, width = depth_map.shape
# x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

# # Normalize the x and y coordinates to map them into a space centered around (0,0)
# x_coords = (x_coords - width / 2) / width
# y_coords = (y_coords - height / 2) / height

# # Flatten all arrays to create a list of coordinates
# x_coords_flat = x_coords.flatten()
# y_coords_flat = y_coords.flatten()
# depths_flat = depth_map.flatten()

# # Combine into a single array of (X, Y, Z) coordinates
# coordinates_3D = np.vstack((x_coords_flat, y_coords_flat, depths_flat)).T
# print(coordinates_3D.shape) 

# depth_map_pcd = o3d.geometry.PointCloud()
# depth_map_pcd.points = o3d.utility.Vector3dVector(coordinates_3D)
# o3d.visualization.draw_geometries([depth_map_pcd])

image = np.zeros((320, 640, 3))  # Placeholder for an actual image

# Convert pixels to world coordinates
world_coordinates = pixel_to_world_coordinates(image, depth_map, intrinsic_matrix, c2w_matrix)
print(world_coordinates.shape)  # Output: (320, 640, 3)

flattened_coordinates = world_coordinates.reshape(-1, 3)

origin = draw_origin()
dummy_camera_pose = visualize_camera_pose(intrinsic_matrix, np.eye(4))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(flattened_coordinates)
o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([room_mesh, dummy_camera_pose, origin])
# o3d.visualization.draw_geometries([room_mesh, camera_pose, origin])
o3d.visualization.draw_geometries([room_mesh, camera_pose, pcd])
# o3d.visualization.draw_geometries([room_mesh, *camera_poses[0:50]])
# o3d.visualization.draw_geometries([pcd, room_mesh, camera_pose])
# o3d.visualization.draw_geometries([pcd, room_mesh])
