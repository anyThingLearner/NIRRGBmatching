import os
import cv2
import numpy as np

crop_size1 = 302
crop_size2 = 470

def read_rgb_image(base_dir, gta_pass, img_id, data_type, num_bits=8, scale_images=False,
                     scaled_img_width=None, scaled_img_height=None,
                     normalize_images=False):
    #gated_imgs = []
    normalizer = 2 ** 8 - 1.

    gate_dir = os.path.join(base_dir, gta_pass, 'rgb_left_8bit')

    img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_ANYCOLOR)
    print(img.shape)
    if data_type == 'real':
        img = img[crop_size1:(img.shape[0] - crop_size1), crop_size2:(img.shape[1] - crop_size2)]
        img = img.copy()
        img[img > 2 ** 8 - 1] = normalizer
    #img = np.float32(img / normalizer)

    return img

train_files = ["/home/cv5/moon/Gated2Depth/splits/real_train_night.txt", "/home/cv5/moon/Gated2Depth/splits/real_test_day.txt", 
                 "/home/cv5/moon/Gated2Depth/splits/real_test_night.txt", "/home/cv5/moon/Gated2Depth/splits/real_train_day.txt",
                 "/home/cv5/moon/Gated2Depth/splits/real_val_day.txt", "/home/cv5/moon/Gated2Depth/splits/real_val_night.txt"]
    
base_dir = "/home/cv5/data/gated2depth/gated2depth/real/"

for train_file in train_files:
    with open(train_file, 'r') as f:
        datas = [line.strip() for line in f.readlines()]
    
    for data in datas:
        in_img = read_rgb_image(base_dir, '', data, 'real')

        cv2.imwrite("/home/cv5/moon/my_ex/patch_matching/rgb_crop/{}.png".format(data), in_img.astype(np.uint8))
        print(data, " complete")

print("fin.")