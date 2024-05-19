import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load the point cloud
pcd = o3d.io.read_point_cloud('predicted/predicted_scannet_meshes/0000_rgbd.ply')

# Downsample the point cloud
downpcd = pcd.voxel_down_sample(voxel_size=0.05)

# Remove statistical outliers
cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
filtered_pcd = downpcd.select_by_index(ind)

# Segment the floor plane using RANSAC
plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.02,
                                                  ransac_n=3,
                                                  num_iterations=1000)

# Extract the floor points
floor_pcd = filtered_pcd.select_by_index(inliers)

# Project the floor points onto the XY plane
floor_points = np.asarray(floor_pcd.points)
floor_points_2d = floor_points[:, :2]  # Discard Z-coordinates

# Create a 2D histogram (floor map)
resolution = 0.05  # 5 cm grid resolution
min_x, min_y = floor_points_2d.min(axis=0)
max_x, max_y = floor_points_2d.max(axis=0)

x_bins = np.arange(min_x, max_x, resolution)
y_bins = np.arange(min_y, max_y, resolution)

hist, x_edges, y_edges = np.histogram2d(floor_points_2d[:, 0], floor_points_2d[:, 1], bins=(x_bins, y_bins))

# Visualize the floor map
plt.imshow(hist.T, origin='lower', extent=[min_x, max_x, min_y, max_y], cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Floor Map')
plt.show()
