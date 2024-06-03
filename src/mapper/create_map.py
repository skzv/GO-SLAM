from matplotlib import pyplot as plt
import numpy as np
import map_utils
import data_loader
import visualization_utils
import camera_utils
import image_transform

base_out_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/room0/first-try/mesh/'
base_data_set_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/room0/results/'
mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'


base_out_path = '/home/emailskpal/cs231nfinalproject/base/replica-rgbd/rgbd/room0/first-try/mesh/'
base_data_set_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/room0/results/'
mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'


# base_out_path = '/home/emailskpal/cs231nfinalproject/base/replica-rgbd/rgbd/office0/first-try/mesh/'
# base_data_set_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/office0/results/'
# mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'

# base_out_path = '/home/emailskpal/cs231nfinalproject/base/replica-rgbd/rgbd/office4/first-try/mesh/'
# base_data_set_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/office4/results/'
# mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'


# base_out_path = '/home/emailskpal/cs231nfinalproject/base/replica-rgbd/rgbd/room1/first-try/mesh/'
# base_data_set_path = '/home/emailskpal/cs231nfinalproject/GO-SLAM/datasets/Replica/room1/results/'
# mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'

data_loader = data_loader.DataLoader(base_out_path , base_data_set_path, mesh_path)
data_loader.load()

room_mesh = data_loader.mesh
data = data_loader.data
all_objects = data_loader.load_all_objects()

cfg = data['config']
c2w_est = data['c2ws']
depths_np = data['depths']
N = len(c2w_est)

camera = camera_utils.Camera(cfg, c2w_est)
image_transformer = image_transform.ImageTransform(cfg['H'], cfg['W'], cfg['H_edge'], cfg['W_edge'])

room_mesh, walls_removed, rotation = map_utils.align_pcd_using_ransac(room_mesh)
all_objects = map_utils.apply_rotation_to_all_objects(all_objects, rotation)
object_boxes = map_utils.get_object_boxes(all_objects)

#room_mesh = map_utils.align_pcd_with_pca(room_mesh)
# map_utils.visualize_pcd(room_mesh)

# visualizer2 = visualization_utils.Visualizer(data_loader, depths_np, walls_removed, c2w_est, camera, image_transformer)
# visualizer2.visualize_objects_with_mesh(all_objects)

# visualizer = visualization_utils.Visualizer(data_loader, depths_np, room_mesh, c2w_est, camera, image_transformer)
# visualizer.visualize_objects_with_mesh(all_objects)

# Compute the density of the point cloud
hist, xedges, yedges = map_utils.compute_density(room_mesh, 100)

# Clamp the histogram values to a specified range
min_val = 0
max_val = np.percentile(hist, 99)  # Clamp to the 99th percentile to remove extreme values
hist = np.clip(hist, min_val, max_val)

# Smooth the density
hist = map_utils.smooth_density(hist, sigma=1)

# Apply a median filter to further smooth out spikes
# hist = median_filter(hist, size=3)

# plot_hist_as_surface(hist, xedges, yedges)

# Plot the density
fig, ax = map_utils.plot_pcd_density(hist, xedges, yedges)
map_utils.draw_boxes_with_labels(fig, ax, object_boxes)
plt.show()

# line_segements = probabilistic_hough_line(hist, threshold=10000, line_length=10, line_gap=2)
# plot_line_segments(line_segements, hist)

#line_segements = probabilistic_hough_line(hist)
#plot_line_segments(line_segements, hist)

# Compute the log density
# hist = compute_log_density(hist)
# plot_pcd_density(hist, xedges, yedges)

# Apply Laplace to get edges
# hist = - sharpen_density(hist)
# plot_pcd_density(hist, xedges, yedges)

# TODO: try extracting walls and objects seperately

# Extract walls (high threshold)
# Apply a threshold to create a binary image
edges = map_utils.threshold_otsu_density(hist, threshold_factor=0.4)
fig, ax = map_utils.plot_binary_image(1 - edges.T, xedges, yedges, 'Thresholded Edges (AKA Walls)')
map_utils.draw_boxes_with_labels(fig, ax, object_boxes)
plt.show()

# # Define the structuring element size for closing
# selem_size = 15  # Adjust this size based on your image
# selem = morphology.square(selem_size)
# edges = morphology.diameter_opening(edges)

# # Apply morphological closing
# # edges = morphology.closing(edges, selem)

# # Plot the closed image
# map_utils.plot_binary_image(1 - edges.T, 'Closed Image')
