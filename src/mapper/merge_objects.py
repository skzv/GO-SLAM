from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import geometry_utils
import camera_utils
import o3d_utils
import data_loader
import visualization_utils
import object_detector
import object_segmenter
import image_transform
import object_extractor
import tqdm

base_out_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/room0/first-try/mesh/'
base_data_set_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/room0/results/'
mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'

use_resnet = False

data_loader = data_loader.DataLoader(base_out_path , base_data_set_path, mesh_path)
data_loader.load()

room_mesh = data_loader.mesh
data = data_loader.data
objects_per_frame = data_loader.load_objects_per_frame_from_base_path()

cfg = data['config']
c2w_est = data['c2ws']
depths_np = data['depths']
N = len(c2w_est)

camera = camera_utils.Camera(cfg, c2w_est)
image_transformer = image_transform.ImageTransform(cfg['H'], cfg['W'], cfg['H_edge'], cfg['W_edge'])

frame_sampling_rate = 1
indices = range(0, N, frame_sampling_rate)

all_objects = None

object_detector = object_detector.ObjectDetector()
object_extractor = object_extractor.ObjectExtractor(c2w_est, depths_np, camera)
visualizer = visualization_utils.Visualizer(data_loader, depths_np, room_mesh, c2w_est, camera, image_transformer)


if use_resnet:
    scrore_threshold = 0.9
    frame_count_threshold = 2
else:
    scrore_threshold = 0.5
    frame_count_threshold = 1

# Extract objects
for idx in tqdm.tqdm(indices, desc="Extracting objects", unit="frame"):
    new_objects = objects_per_frame[idx]
    if len(new_objects) == 0:
        continue
    new_objects = object_detector.filter_objects_by_threshold(new_objects, scrore_threshold)
    new_objects = object_extractor.add_depth_to_objects(new_objects, idx)
    all_objects = object_extractor.merge_objects(all_objects, new_objects)
    
    # visualizer.draw_frame_and_estimated_depth_map(idx, new_objects)
    # visualizer.visualize_objects_with_mesh(idx, new_objects)

all_objects = object_extractor.finish_merging(all_objects, frame_count_threshold)
objects = object_extractor.flatten_all_objects(all_objects)
visualizer.visualize_objects_with_mesh(objects)
# visualizer.visualize_merged_objects_with_mesh(objects)

# Save objects
np.save(base_out_path + 'merged_objects.npy', objects)