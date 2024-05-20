from src.mesher import extract_depth_from_mesh
import open3d as o3d
import numpy as np
import trimesh
import torch

room_mesh = ply_file_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/mesh/final_raw_mesh.ply'
room_mesh = o3d.io.read_triangle_mesh(ply_file_path)

# Convert Open3D mesh to trimesh format
vertices = np.asarray(room_mesh.vertices)
faces = np.asarray(room_mesh.triangles)

# Create trimesh object
trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

c2w = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/checkpoints/est_poses.npy')
# depths = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/mesh/depth_list.npy', allow_pickle=True)
# TODO: load intrisinic matrix
# Taken from replica.yaml config
fx = 600.0  # Focal length in x-direction
fy = 600.0  # Focal length in y-direction
cx = 599.5  # x-coordinate of the optical center
cy = 339.5  # y-coordinate of the optical center
H = 320
W = 640

# extract_depth_from_mesh(mesh,
                            # c2w_list,
                            # H, W, fx, fy, cx, cy,
                            # far=20.0, )
depths = extract_depth_from_mesh(trimesh_mesh, torch.from_numpy(c2w), H, W, fx, fy, cx, fy)

def create_depth_pcd(depth_map):
    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize the x and y coordinates to map them into a space centered around (0,0)
    x_coords = (x_coords - width / 2) / width
    y_coords = (y_coords - height / 2) / height

    # Flatten all arrays to create a list of coordinates
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    depths_flat = depth_map.flatten()

    # Combine into a single array of (X, Y, Z) coordinates
    coordinates_3D = np.vstack((x_coords_flat, y_coords_flat, depths_flat)).T
    print(coordinates_3D.shape) 

    depth_map_pcd = o3d.geometry.PointCloud()
    depth_map_pcd.points = o3d.utility.Vector3dVector(coordinates_3D)
    return depth_map_pcd

o3d.visualization.draw_geometries([create_depth_pcd(depths[800])])