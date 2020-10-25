import matplotlib as mpl
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import cv2
import os

crop_size = 150
# 1, 720, 1280, 1 
def read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance, scale_images=False,
                  scaled_img_width=None,
                  scaled_img_height=None, raw_values_only=False):
    
    if data_type == 'gated':
        depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, "depth_hdl64_gated_compressed", img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size:(depth_lidar1.shape[0] - crop_size),
                    crop_size: (depth_lidar1.shape[1] - crop_size)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / max_distance)

        return np.expand_dims(np.expand_dims(depth_lidar1, axis=2), axis=0), \
               np.expand_dims(np.expand_dims(gt_mask, axis=2), axis=0)
    else:
        depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, "depth_hdl64_rgb_left_compressed", img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size:(depth_lidar1.shape[0] - crop_size),
                    crop_size: (depth_lidar1.shape[1] - crop_size)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / max_distance)

        return np.expand_dims(np.expand_dims(depth_lidar1, axis=2), axis=0), \
               np.expand_dims(np.expand_dims(gt_mask, axis=2), axis=0)


def read_rgb_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance, scale_images=False,
                  scaled_img_width=None,
                  scaled_img_height=None, raw_values_only=False):
    
    if data_type == 'gated':
        depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, "depth_hdl64_gated_compressed", img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size:(depth_lidar1.shape[0] - crop_size),
                    crop_size: (depth_lidar1.shape[1] - crop_size)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / max_distance)

        return np.expand_dims(np.expand_dims(depth_lidar1, axis=2), axis=0), \
               np.expand_dims(np.expand_dims(gt_mask, axis=2), axis=0)
    else:
        depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, "depth_hdl64_rgb_left_compressed", img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size:(depth_lidar1.shape[0] - crop_size),
                    crop_size: (depth_lidar1.shape[1] - crop_size)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / max_distance)

        return np.expand_dims(np.expand_dims(depth_lidar1, axis=2), axis=0), \
               np.expand_dims(np.expand_dims(gt_mask, axis=2), axis=0)


def colorize_pointcloud(depth, min_distance=3, max_distance=80, radius=3):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(depth > 0)

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)

    return pointcloud_color



base_dir = "/home/cv5/data/gated2depth/gated2depth/real"
gta_pass = ""
img_id = "00000"
min_distance=2.
max_distance=200.
min_eval_distance = min_distance
max_eval_distance = 80.



gt_patch, _ = read_gt_image(base_dir, gta_pass, img_id, data_type='gated', raw_values_only=True, min_distance=min_distance, max_distance=max_distance)
depth_lidar1_color = colorize_pointcloud(gt_patch, min_distance=min_eval_distance, max_distance=max_eval_distance, radius=3)
cv2.imwrite("/home/cv5/moon/my_ex/patch_matching/lidar_gated.jpg", depth_lidar1_color)
# 420, 980, 3

gt_patch, _ = read_gt_image(base_dir, gta_pass, img_id, data_type='rgb', raw_values_only=True, min_distance=min_distance, max_distance=max_distance)
depth_lidar2_color = colorize_pointcloud(gt_patch, min_distance=min_eval_distance, max_distance=max_eval_distance, radius=3)

cv2.imwrite("/home/cv5/moon/my_ex/patch_matching/lidar_rgb.jpg", depth_lidar2_color)
# 724, 1620, 3
