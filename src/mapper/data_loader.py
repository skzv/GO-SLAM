import cv2
import open3d as o3d
import numpy as np
from PIL import Image

class DataLoader:
    def __init__(self, base_out_path, base_data_set_path, base_mesh_path):
        self.base_out_path = base_out_path
        self.base_data_set_path = base_data_set_path
        self.base_mesh_path = base_mesh_path

    def load(self):
        self.mesh = self.load_mesh_from_base_path()
        self.data = self.load_data_from_base_path()

    def load_mesh_from_base_path(self):
        ply_file_path = self.base_mesh_path
        return self.load_mesh(ply_file_path)


    def load_mesh(self, ply_file_path):
        room_mesh = o3d.io.read_point_cloud(ply_file_path)
        return room_mesh


    def load_data_from_base_path(self):
        cfg_path = self.base_out_path + 'cfg.npy'
        c2w_est_path = self.base_out_path + 'estimate_c2w_list.npy'
        depths_path = self.base_out_path + 'depth_list.npy'
        return self.load_data(cfg_path, c2w_est_path, depths_path)
    

    def load_objects_per_frame_from_base_path(self):
        return np.load(self.base_out_path + 'objects_per_frame.npy', allow_pickle=True)
    

    def load_all_objects(self):
        return np.load(self.base_out_path + 'merged_objects.npy', allow_pickle=True)


    def load_data(self, cfg_path, c2w_est_path, depths_path):
        print('Loading data...')

        cfg = np.load(cfg_path)
        config = {
            'H': int(cfg[0]),
            'W': int(cfg[1]),
            'fx': float(cfg[2]),
            'fy': float(cfg[3]),
            'cx': float(cfg[4]),
            'cy': float(cfg[5]),
            # 'H_edge': 0,
            # 'W_edge': 0
            'H_edge': int(cfg[6]),
            'W_edge': int(cfg[7])
        }
        self.config = config
        
        c2w_est = np.load(c2w_est_path)
        depths = np.load(depths_path, allow_pickle=True)
        depths_np = np.array([tensor.numpy() for tensor in depths])

        print("Load c2ws => shape: ", c2w_est.shape)
        print("Loaded estimated depths => shape: ", depths_np.shape)

        return {'config': config, 'c2ws': c2w_est, 'depths': depths_np}
    
    def get_frame_path(self, index):
        return self.base_data_set_path + f'frame{index:06d}.jpg'
    
    def get_frame_image(self, index):
        return Image.open(self.get_frame_path(index))