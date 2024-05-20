import numpy as np

c2w = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/checkpoints/est_poses.npy')
depths = np.load('/home/skz/cs231n/GO-SLAM-skz/out/replica/office0/first-try/mesh/depth_list.npy', allow_pickle=True)

depths_np = np.array([tensor.numpy() for tensor in depths])

print(c2w.shape)
print(depths_np.shape)

# c2w: (N, 4, 4)
# depths_np: (N, H, W)

