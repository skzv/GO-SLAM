## GAS Framework Code

# Mapper code

The GAS framework is built on-top of the GO-SLAM visual SLAM project, available here: https://github.com/youmi-zym/GO-SLAM

The first step in the pipeline is to run scripts/run_scene.sh, which executes GO-SLAM on a scene from a dataset (Replica, or ScanNet, or ETH3D).

GO-SLAM outputs an environment mesh as a ply file. We also modified src/mesher.py to save some configuration information (modified camera intrinsics and modified frame size), the estimated camera poses through the video, and the estimated depth map for each frome, computed by passing the mesh and estimated camera pose to pyrender.

From there, our code that calculates the 3D labels and computes a 2D map resides in src/mapper. 

The next step in the pipeline is to run src/mapper/extract_objects.py, which extracts objects from each frame by running either our fine-tuned Detectron2 model or an out-of-the-box faster rcnn resnet model from pytorch. The output of this is also saved locally. 

The next step is to run src/mapper/object_extractor.py, which computes the 3D coordinates of each object identified in the previous steps, and then merges the same instances of objects seen across frames. These objects are also saved locally.

The final step in the pipeline is src/mapper/create_map.py. This consumes the estimated mesh and 3D object point clouds to label the 3D environment mesh. The 3D mesh, with labeled object point clouds, is then alighed with the xyz frame and projected into 2D to create a 2D floor map.

# Detectron 2 Training Code