from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import geometry_utils
import camera_utils
import o3d_utils
import data_loader
import visualization_utils
import object_detector
import object_detector_detectron
import object_segmenter
import image_transform
import object_extractor
import tqdm


scenes = ["office0", "office4", "room0", "room1", "room2"]
scenes = ["room0"]
for scene in scenes:
    # base_out_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/office0/first-try/mesh/'
    # base_data_set_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/office0/results/'
    # mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'

    use_resnet = False
    draw_objects = False

    base_out_path = '/home/skz/cs231n/GO-SLAM-skz/out/replica/rgbd/' + scene + '/first-try/mesh/'
    base_data_set_path = '/home/skz/cs231n/GO-SLAM-skz/datasets/Replica/' + scene + '/results/'
    mesh_path = base_out_path + 'final_raw_mesh_forecast.ply'

    data_loader_obj = data_loader.DataLoader(base_out_path , base_data_set_path, mesh_path)
    data_loader_obj.load()

    room_mesh = data_loader_obj.mesh
    data = data_loader_obj.data

    cfg = data['config']
    c2w_est = data['c2ws']
    depths_np = data['depths']
    N = len(c2w_est)

    camera = camera_utils.Camera(cfg, c2w_est)
    image_transformer = image_transform.ImageTransform(cfg['H'], cfg['W'], cfg['H_edge'], cfg['W_edge'])

    if use_resnet:
        object_detector = object_detector.ObjectDetector()
        frame_sampling_rate = 20
    else:
        frame_sampling_rate = 5
        object_detector = object_detector_detectron.ObjectDetectorDetectron()

    indices = [750]
    # frame_sampling_rate = 20
    indices = range(0, N, frame_sampling_rate)

    objects_per_frame = np.empty(N, dtype=object)
    objects_per_frame[:] = [[] for _ in range(N)]

    object_extractor = object_extractor.ObjectExtractor(c2w_est, depths_np, camera)
    visualizer = visualization_utils.Visualizer(data_loader_obj, depths_np, room_mesh, c2w_est, camera, image_transformer)

    # Extract objects
    for idx in tqdm.tqdm(indices, desc="Extracting objects", unit="frame"):
        frame_path = data_loader_obj.get_frame_path(idx)
        image = image_transformer.transform_image(frame_path)
        new_objects = object_detector.detect(image)
        new_objects = object_detector.filter_objects_by_threshold(new_objects, 0.5)
        objects_per_frame[idx] = np.array(new_objects)  # Convert new_objects to numpy array

        # new_objects = object_detector.filter_objects_by_threshold(new_objects, 0.75)
        new_objects = object_extractor.add_depth_to_objects(new_objects, idx)

        if draw_objects and len(new_objects) > 0:
            print('Draw frame')
            visualizer.draw_frame_and_estimated_depth_map(idx, new_objects)

        # visualizer.visualize_objects_with_mesh(idx, objects)

    # objects = object_extractor.flatten_all_objects(all_objects)
    # visualizer.visualize_objects_with_mesh(all_objects)

    # Save objects
    np.save(base_out_path + 'objects_per_frame.npy', objects_per_frame, allow_pickle=True)
