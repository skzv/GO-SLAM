from matplotlib import patches
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, laplace, sobel, label, median_filter, binary_opening
from skimage import filters, morphology, measure
from skimage.transform import hough_line, hough_line_peaks, rotate, probabilistic_hough_line
import trimesh
from PIL import ImageFont
import o3d_utils
import object_extractor
from scipy.spatial import ConvexHull


def compute_density(pcd, bins_per_unit):
    # Get the points from the point cloud
    points = np.asarray(pcd.points)

    # Project the points onto the XY plane (ignore Z-coordinate)
    xy_points = points[:, :2]
    x_range = np.max(xy_points[:, 0]) - np.min(xy_points[:, 0])
    y_range = np.max(xy_points[:, 1]) - np.min(xy_points[:, 1])

    # Create a 2D histogram to represent density
    x_bins = int(bins_per_unit * x_range)  # Number of bins in x direction
    y_bins = int(bins_per_unit * y_range) # Number of bins in y direction
    hist, xedges, yedges = np.histogram2d(xy_points[:, 0], xy_points[:, 1], bins=[x_bins, y_bins])

    return hist, xedges, yedges


def get_object_boxes(all_objects):
    boxes = []
    for object in all_objects:
        # project object into 2d
        points = object['world_coordinates_cm'] / 100 # convert to meters

        # remove top and bottom 10% of points
        points = points[points[:, 2] < np.percentile(points[:, 2], 90)]
        points = points[points[:, 2] > np.percentile(points[:, 2], 10)]

        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        # create a bounding box
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        width = max(0.1, max_x - min_x)
        height = max(0.1, max_y - min_y)

        box = {
            'x0': min_x,
            'y0': min_y,
            'width': width,
            'height': height,
            'center_x': center_x,
            'center_y': center_y
        }

        # convert points into (x,y) tuples
        points_tuples = np.array([(point[0], point[1]) for point in points])

        # add label and bounding polygon
        boxes.append({
            'label': object['label'],
            'box': box,
            'points_tuples': points_tuples
        })

    return boxes

def draw_boxes_with_labels(fig, ax, boxes):
    """
    Draws boxes with labels on the given figure and axes.

    Parameters:
    - fig: matplotlib figure
    - ax: matplotlib axes
    - boxes: list of dictionaries containing 'label' and 'box' information

    Example of 'boxes' parameter:
    boxes = [
        {
            'label': 'Object 1',
            'box': {
                'x0': 50,
                'y0': 50,
                'width': 100,
                'height': 200
            }
        },
        {
            'label': 'Object 2',
            'box': {
                'x0': 200,
                'y0': 100,
                'width': 150,
                'height': 100
            }
        }
    ]
    """
    for item in boxes:
        box = item['box']
        label = item['label']
        
        # Create a Rectangle patch
        # rect = patches.Rectangle((box['x0'], box['y0']), box['width'], box['height'],
        #                          linewidth=2, edgecolor='b', facecolor='none')
        points_tuples = item['points_tuples']
        polygon = create_patch_from_points(points_tuples)
        # Add the rectangle to the plot
        ax.add_patch(polygon)
        
        # Add label
        ax.text(box['center_x'], box['center_y'] , label, color='white', backgroundcolor='black') 


def create_patch_from_points(points, color='blue', alpha=0.5):
    """
    Create a polygon patch from a list of points and add it to the given axes.

    Parameters:
    - ax: The Matplotlib axes to add the patch to.
    - points: A list of (x, y) tuples representing the points within the polygon.
    - color: The color of the patch.
    - alpha: The transparency level of the patch.
    """
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Get the hull vertices
    hull_points = points[hull.vertices]
    
    # Create a Polygon patch
    polygon = patches.Polygon(hull_points, closed=True, edgecolor='black', facecolor=color, alpha=alpha)
    
    return polygon


def compute_log_density(hist):
    return np.log(hist + 1e-12)


def smooth_density(hist, sigma):
    return gaussian_filter(hist, sigma=sigma)


def sharpen_density(hist):
    return laplace(hist, mode='reflect')


def compute_edges(hist):
    # Apply the Sobel filter to extract edges
    sobel_x = sobel(hist, axis=0, mode='reflect')
    sobel_y = sobel(hist, axis=1, mode='reflect')
    edges_sobel = np.hypot(sobel_x, sobel_y)

    return edges_sobel


def plot_pcd_density(hist, xedges, yedges):
    # Plot the image
    fig, ax = plt.subplots(1)

    # Plot the density
    cax = ax.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Density')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Point Cloud Density')
    return fig, ax


def plot_hist_as_surface(hist, xedges, yedges):
    # Prepare data for surface plot
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    Z = hist.T

    # Plot the density as a surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    ax.set_title('Point Cloud Density Surface Plot')
    plt.show()


def visualize_pcd(pcd):
    # Visualize the point cloud
    origin = o3d_utils.draw_origin()
    o3d.visualization.draw_geometries([origin, pcd])


def project_pcd_onto_xy_plane(pcd):
    pcd_xy = o3d.geometry.PointCloud(pcd)
    points = np.asarray(pcd_xy.points)
    points[:, 2] = 0
    pcd_xy.points = o3d.utility.Vector3dVector(points)
    return pcd_xy


def remove_ceiling(pcd_orig):
    pcd = o3d.geometry.PointCloud(pcd_orig)

    # Get the points from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    max_z = np.max(points[:, 2])
    print(max_z)

    # Filter out points with z-coordinate above 95% of the maximum z-coordinate
    mask = points[:, 2] <= 0.80 * max_z
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Update the point cloud with the filtered points and colors
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return pcd


def remove_floor(pcd_orig):
    pcd = o3d.geometry.PointCloud(pcd_orig)

    # Get the points from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    min_z = np.min(points[:, 2])
    print(min_z)

    # Filter out points with z-coordinate below 5% of the minimum z-coordinate
    mask = points[:, 2] >= 0.05 * min_z
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Update the point cloud with the filtered points and colors
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return pcd


def extract_middle_band(pcd_orig):
    pcd = o3d.geometry.PointCloud(pcd_orig)

    # Get the points from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    z_range = max_z - min_z

    theshold = 0.45
    # Filter out points with z-coordinate below 30% of the z-range
    mask = (points[:, 2] >= min_z + theshold * z_range) & (points[:, 2] <= max_z - theshold * z_range)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Update the point cloud with the filtered points and colors
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return pcd


def align_pcd_to_eigenvectors(pcd_orig, eigenvectors):
    pcd = o3d.geometry.PointCloud(pcd_orig)

    points = np.asarray(pcd.points)

    # Center the point cloud by subtracting the mean
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    # Align the point cloud to the principal axes
    aligned_points = np.dot(centered_points, eigenvectors)

    # Update the point cloud with the aligned points
    pcd.points = o3d.utility.Vector3dVector(aligned_points)

    return pcd


def compute_principle_sorted_eigenvectors(pcd):
    points = np.asarray(pcd.points)

    # Center the point cloud by subtracting the mean
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    # Perform PCA
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors


def align_pcd_with_pca(pcd_orig):
    pcd = o3d.geometry.PointCloud(pcd_orig)

    # Get the points from the point cloud
    points = np.asarray(pcd.points)

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvectors = pca.components_

    # Align the point cloud to the principal axes
    aligned_points = np.dot(points, eigenvectors.T)

    # Update the point cloud with the aligned points
    pcd.points = o3d.utility.Vector3dVector(aligned_points)

    return pcd


def align_plane_to_axis(normal, target_axis):
    rotation_axis = np.cross(normal, target_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(normal, target_axis) / (np.linalg.norm(normal) * np.linalg.norm(target_axis)))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
    return rotation_matrix


def align_pcd_using_ransac(pcd_orig):
    # Step 1: Segment the largest plane using RANSAC
    plane_models = []
    inliers_list = []

    pcd = o3d.geometry.PointCloud(pcd_orig)

    for _ in range(2):  # We need to find the two largest planes
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=1000)
        plane_models.append(plane_model)
        inliers_list.append(inliers)
        
        # Remove the inliers to find the next plane
        pcd = pcd.select_by_index(inliers, invert=True)

    # Retrieve the largest and second largest plane models
    largest_plane_model = plane_models[0]
    second_largest_plane_model = plane_models[1]

    # Normal vectors of the planes
    largest_plane_normal = np.array(largest_plane_model[:3])
    second_largest_plane_normal = np.array(second_largest_plane_model[:3])

    # Step 2: Align the largest plane normal with the z-axis
    rotation_matrix_1 = align_plane_to_axis(largest_plane_normal, np.array([0, 0, 1]))

    # flip the normal if it is pointing downwards
    # if rotation_matrix_1[2, 2] < 0:
    rotation_matrix_1 = -rotation_matrix_1

    # Apply the first rotation to the point cloud
    pcd = o3d.geometry.PointCloud(pcd_orig)
    # pcd.rotate(rotation_matrix_1)
    # rotate second largest plane normal
    second_largest_plane_normal_rotated = np.dot(rotation_matrix_1, second_largest_plane_normal)

    # Step 3: Align the second largest plane normal with the y-axis
    rotation_matrix_2 = align_plane_to_axis(second_largest_plane_normal_rotated, np.array([0, 1, 0]))

    # Apply the second rotation to the point cloud
    # pcd.rotate(rotation_matrix_2)
    rotation =  rotation_matrix_2 @ rotation_matrix_1
    # pcd.rotate(rotation)
    pcd.points = o3d.utility.Vector3dVector(np.dot(rotation, np.asarray(pcd.points).T).T)
    return pcd, rotation

def apply_rotation_to_all_objects(objects, rotation):
    for object in objects:
        world_coordinates = object['world_coordinates_cm'] / 100
        rotated_coordinates = np.dot(rotation, world_coordinates.T).T
        object['world_coordinates_cm'] = np.round(100 * rotated_coordinates).astype(int)
        object['3d_box_world_coordinates'] = object_extractor.ObjectExtractor.min_max_3d_box(rotated_coordinates)
    return objects

def threshold_otsu_density(hist, threshold_factor):
    return hist > threshold_factor * filters.threshold_otsu(hist)


def morph(edges):
    # Apply morphological operations to remove small dots and noise
    # Strengthen binary opening by increasing the size of the structuring element
    structuring_element = morphology.square(3)  # Increase size for stronger effect
    edges = binary_opening(edges, structure=structuring_element)

    return edges


def plot_binary_image(image,  xedges, yedges, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return fig, ax


def plot_image(image, xedges, yedges, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return fig, ax


def detect_line_segments(edges):
    # Use the Probabilistic Hough Transform to detect line segments
    line_segments = probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=10)

    return line_segments


def plot_line_segments(line_segments, edges):
    # Prepare to plot the floor map
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(edges, cmap='gray', origin='lower')

    # Plot the detected line segments
    for line in line_segments:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), 'r', origin='lower')

    ax.set_title('Detected Line Segments for Floor Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def detect_principle_rotation_angle(edges):
    # Use Hough Transform to detect lines
    h, theta, d = hough_line(edges)
    accum, angles, dists = hough_line_peaks(h, theta, d, threshold=0.3*np.max(h))

    # Compute the rotation angle to allign with the walls
    angles_deg = angles * (180 / np.pi)
    print(f'Angles: {angles_deg}')
    rotation_angle = 90 - angles_deg[0]
    print(f'Rotation Angle: {rotation_angle:.2f} degrees')

    return rotation_angle
    # Rotate the image to straighten the lines
    # edges = rotate(edges, angle=-rotation_angle, resize=True)


def rotate_edges(edges, angle):
    return rotate(edges, angle=angle, resize=True)


# Function to create a text mesh using trimesh
def create_text_mesh(text):
    return o3d.t.geometry.TriangleMesh.create_text(text, depth=0.1).to_legacy()


# # Load the PLY file
# ply_file_path = 'predicted/predicted_rgbd_meshes/office1.ply'
# # ply_file_path = 'predicted/predicted_scannet_meshes/0000_rgbd.ply'
# pcd = o3d.io.read_point_cloud(ply_file_path)

# # Print some basic information about the point cloud
# print(pcd)

# # pcd = align_pcd_with_pca(pcd)

# # visualize_pcd(pcd)
# print("Open3D version:", o3d.__version__)

# # Initialize the GUI Application
# # app = o3d.visualization.gui.Application.instance
# # app.initialize()

# # Setup O3DVisualizer
# # vis = o3d.visualization.O3DVisualizer("Open3D - O3DVisualizer", 1024, 768)
# # vis.add_geometry("PointCloud", pcd)

# # Show the visualizer window
# # o3d.visualization.draw([vis])

# # Create a simple point cloud
# #pcd = o3d.geometry.PointCloud()
# #pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# # Create text mesh
# # text = "Hello, Open3D!"
# # text_mesh = o3d.t.geometry.TriangleMesh.create_text('Open3D', depth=1).to_legacy()
# # text_mesh.paint_uniform_color((0.4, 0.1, 0.9))

# # Position the text mesh
# # text_mesh.translate([0.5, 0.5, 0.5])

# # print(type(pcd))
# # print(type(text_mesh))

# mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

# # Visualize the point cloud with text mesh
# o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)

# # Compute the density of the point cloud
# hist, xedges, yedges = compute_density(pcd, 100)

# # Clamp the histogram values to a specified range
# min_val = 0
# max_val = np.percentile(hist, 99)  # Clamp to the 99th percentile to remove extreme values
# hist = np.clip(hist, min_val, max_val)

# # Smooth the density
# hist = smooth_density(hist, sigma=1)

# # Apply a median filter to further smooth out spikes
# # hist = median_filter(hist, size=3)

# # plot_hist_as_surface(hist, xedges, yedges)

# # Plot the density
# plot_pcd_density(hist, xedges, yedges)

# # line_segements = probabilistic_hough_line(hist, threshold=10000, line_length=10, line_gap=2)
# # plot_line_segments(line_segements, hist)

# #line_segements = probabilistic_hough_line(hist)
# #plot_line_segments(line_segements, hist)

# # Compute the log density
# # hist = compute_log_density(hist)
# # plot_pcd_density(hist, xedges, yedges)

# # Apply Laplace to get edges
# # hist = - sharpen_density(hist)
# # plot_pcd_density(hist, xedges, yedges)

# # TODO: try extracting walls and objects seperately

# # Extract walls (high threshold)
# # Apply a threshold to create a binary image
# edges = threshold_otsu_density(hist, threshold_factor=1)

# plot_binary_image(1 - edges.T, 'Thresholded Edges (AKA Walls)')

# # Define the structuring element size for closing
# selem_size = 15  # Adjust this size based on your image
# selem = morphology.square(selem_size)
# edges = morphology.diameter_opening(edges)

# # Apply morphological closing
# # edges = morphology.closing(edges, selem)

# # Plot the closed image
# plot_binary_image(1 - edges.T, 'Closed Image')

# # # Remove walls from histogram
# # hist = hist * (1 - edges)
# # plot_pcd_density(hist, xedges, yedges)

# # edges = threshold_otsu_density(hist, threshold_factor=1)
# # plot_binary_image(1 - edges, 'Thresholded Edges (AKA objects)')


# # edges = morph(edges)
# # plot_binary_image(1 - edges, 'Morphed Edges')

# # line_segements = detect_line_segments(edges)
# # line_segements = detect_line_segments(edges)
# # line_segements = probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=3)
# # plot_line_segments(line_segements, 1 - edges)