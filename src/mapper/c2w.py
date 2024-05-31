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

base_out_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/office0/first-try/'
base_data_set_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/office0/results/'
mesh_path = base_out_path + 'mesh/final_raw_mesh.ply'

data_loader = data_loader.DataLoader(base_out_path + 'mesh/' , base_data_set_path, mesh_path)
data_loader.load()

room_mesh = data_loader.mesh
data = data_loader.data

cfg = data['config']
c2w_est = data['c2ws']
depths_np = data['depths']
N = len(c2w_est)

camera = camera_utils.Camera(cfg, c2w_est)
image_transformer = image_transform.ImageTransform(cfg['H'], cfg['W'], cfg['H_edge'], cfg['W_edge'])

indices = [750]

objects_per_frame = [None] * N

object_detector = object_detector.ObjectDetector(threshold=0.75)
object_extractor = object_extractor.ObjectExtractor(c2w_est, depths_np, camera)
visualizer = visualization_utils.Visualizer(data_loader, depths_np, room_mesh, c2w_est, camera, image_transformer)

# Extract objects
for idx in indices:
    frame_path = data_loader.get_frame_path(idx)
    image = image_transformer.transform_image(frame_path)
    objects = object_detector.detect(image)
    objects = object_extractor.add_depth_to_objects(objects, idx)
    visualizer.draw_frame_and_estimated_depth_map(idx, objects)
    visualizer.visualize_objects_with_mesh(idx, objects)

    objects_per_frame[idx] = objects

# Merge objects
merged_objects = object_extractor.merge_objects(objects_per_frame)
visualizer.visualize_merged_objects_with_mesh(merged_objects)
