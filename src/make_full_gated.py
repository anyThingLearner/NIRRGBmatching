import os
import cv2
import numpy as np
import torch

import dataset_util as dutil
import corrnet
import concatnet
import domainnet

train_files = "/home/cv5/moon/Gated2Depth/splits/real_train_night.txt"
crop_size = 150
# def training(rgb_path, nir_path, epochs, ):

def read_gated_image2(base_dir, img_id):
    gated_imgs = []
    normalizer = 2*10 -1

    for gate_id in range(3):
        gate_dir = os.path.join(base_dir, 'gated%d_10bit' %gate_id)
        img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_GRAYSCALE)
        gated_imgs.append(img)
    
    for x in range(gated_imgs[0].shape[0]):
        for y in range(gated_imgs[0].shape[1]):
            gated_imgs[0][x][y] = max(gated_imgs[0][x][y], gated_imgs[1][x][y], gated_imgs[2][x][y])
    
    return gated_imgs[0]



def read_gated_image(base_dir, gta_pass, img_id, data_type, num_bits=10, scale_images=False,
                     scaled_img_width=None, scaled_img_height=None,
                     normalize_images=False):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        gate_dir = os.path.join(base_dir, gta_pass, 'gated%d_10bit' % gate_id)
        img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if data_type == 'real':
            img = img[crop_size:(img.shape[0] - crop_size), crop_size:(img.shape[1] - crop_size)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        print(img.shape)
        gated_imgs.append(np.expand_dims(img, axis=2))

    img = np.concatenate(gated_imgs, axis=2)
    if normalize_images:
        mean = np.mean(img, axis=2, keepdims=True)
        std = np.std(img, axis=2, keepdims=True)
        img = (img - mean) / (std + np.finfo(float).eps)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img, axis=0)


base_dir = "/home/cv5/data/gated2depth/gated2depth/real"



from PIL import Image

def read_gated_image(base_dir, gta_pass, img_id, data_type, num_bits=10, scale_images=False,
                     scaled_img_width=None, scaled_img_height=None,
                     normalize_images=False):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        gate_dir = os.path.join(base_dir, gta_pass, 'gated%d_10bit' % gate_id)
        img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if data_type == 'real':
            img = img[crop_size:(img.shape[0] - crop_size), crop_size:(img.shape[1] - crop_size)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        print(img.shape)
        gated_imgs.append(np.expand_dims(img, axis=2))

    img = np.concatenate(gated_imgs, axis=2)
    if normalize_images:
        mean = np.mean(img, axis=2, keepdims=True)
        std = np.std(img, axis=2, keepdims=True)
        img = (img - mean) / (std + np.finfo(float).eps)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)

    return np.expand_dims(img, axis=0)

train_files = ["/home/cv5/moon/Gated2Depth/splits/real_train_night.txt", "/home/cv5/moon/Gated2Depth/splits/real_test_day.txt", 
                 "/home/cv5/moon/Gated2Depth/splits/real_test_night.txt", "/home/cv5/moon/Gated2Depth/splits/real_train_day.txt",
                 "/home/cv5/moon/Gated2Depth/splits/real_val_day.txt", "/home/cv5/moon/Gated2Depth/splits/real_val_night.txt"]
    
for train_file in train_files:
    with open(train_file, 'r') as f:
        datas = [line.strip() for line in f.readlines()]
    
    for data in datas:
        in_img = dutil.read_gated_image(base_dir, '', data, 'real')
        input_patch = in_img
        scaled_input = cv2.resize(input_patch[0, :, :, :],
                                dsize=(int(input_patch.shape[2]), int(input_patch.shape[1])),
                                interpolation=cv2.INTER_AREA) * 255 ###

        input_output = np.zeros((420, 980*3,3))

        for i in range(3):
            input_output[:scaled_input.shape[0], :scaled_input.shape[1], i] = scaled_input[:, :, 0]
            input_output[:scaled_input.shape[0], scaled_input.shape[1]: 2 * scaled_input.shape[1], i] = scaled_input[:,:, 1]
            input_output[:scaled_input.shape[0], scaled_input.shape[1] * 2:scaled_input.shape[1] * 3, i] = scaled_input[:, :, 2]

        input1 = input_output[:,:980,:]
        input2 = input_output[:,980:1960,:]
        input3 = input_output[:,1960:, :]
        
        full_gate2 = np.zeros((420,980,3))
        for x in range(420):
            for y in range(980):
                for z in range(3):
                    full_gate2[x,y,z] = max(input1[x,y,z], input2[x,y,z], input3[x,y,z])

        cv2.imwrite("/home/cv5/moon/my_ex/patch_matching/full_gated/{}.png".format(data), full_gate2.astype(np.uint8))
        print(data, " complete")

print("fin.")