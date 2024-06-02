import cv2
import numpy as np
import open3d as o3d


def apply_mask_to_world_coordinates(world_coordinates, mask):
    # Combine (H, W, 3) world coordinates into a (H*W, 3) array
    flattened_coordinates = world_coordinates.reshape(-1, 3)

    if mask is not None:
        mask = mask.flatten()
        # Apply mask to the flattened coordinates
        flattened_coordinates = flattened_coordinates[mask > 0]

    return flattened_coordinates

def flattened_coordinates_to_point_cloud(flattened_coordinates):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flattened_coordinates)
    return pcd

def world_coordinates_to_point_cloud(world_coordinates, mask=None):
    flattened_coordinates = apply_mask_to_world_coordinates(world_coordinates, mask)
    return flattened_coordinates_to_point_cloud(flattened_coordinates)

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