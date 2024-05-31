import open3d as o3d


def draw_origin():
    # Create a coordinate frame (size can be adjusted)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
    return coordinate_frame