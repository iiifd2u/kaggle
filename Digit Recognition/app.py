import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from hog_plus_linear_nn import test_data, OneLayer, device, hog

block_size = 7
norm_size = 2

vector_dim = 28*5

def random_digit(model):
    x, y = random.choice(test_data)
    x_img = x.squeeze().numpy()
    x = x.to(device)
    x = x.unsqueeze(0)
    out = model(hog(x)).argmax(-1).item()
    return x_img, out

def preprocessed_img(img: np.ndarray):
    img = cv2.resize(img, dsize=(vector_dim, vector_dim))
    img = img/255.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.where(angle>=180, angle-180, angle).astype(np.uint8)
    w, h = img.shape

    step_x = w//(block_size) - 1
    step_y = h // (block_size) - 1

    angles_imgs = np.zeros(shape=(step_x, step_y))
    magnitude_img = np.zeros(shape=(step_x, step_y))

    for x_pos in range(step_x):
        for y_pos in range(step_y):
            block = angle[x_pos*block_size:(x_pos+1)*block_size, y_pos*block_size:(y_pos+1)*block_size]
            angles_imgs[x_pos][y_pos] = np.mean(block)
            block = mag[x_pos*block_size:(x_pos+1)*block_size, y_pos*block_size:(y_pos+1)*block_size]
            magnitude_img[x_pos][y_pos] = np.mean(block)

    magnitude_img = magnitude_img / magnitude_img.max()*5

    x_center = np.arange(1, angles_imgs.shape[1]+1)*block_size
    y_center = np.arange(1, angles_imgs.shape[1] + 1) * block_size

    xx, yy = np.meshgrid(x_center, y_center)
    coord_centers = np.c_[xx.ravel(), yy.ravel()]

    assert angles_imgs.shape[0]*angles_imgs.shape[1] == coord_centers.shape[0]

    dxs_draw = np.abs(magnitude_img*np.cos(np.deg2rad(angles_imgs))).astype(np.uint8).ravel()
    dys_draw = np.abs(magnitude_img * np.sin(np.deg2rad(angles_imgs))).astype(np.uint8).ravel()

    img = np.zeros(shape=(vector_dim, vector_dim, 3), dtype=np.uint8)

    for idx, coord in enumerate(coord_centers):
        start_pt = (coord[0] - dxs_draw[idx]//2, coord[1] - dys_draw[idx]//2)
        end_pt = (coord[0] + dxs_draw[idx] // 2, coord[1] + dys_draw[idx] // 2)
        if start_pt!=end_pt:
            img = cv2.line(img, pt1=start_pt, pt2=end_pt, color=(0, 0, 255), thickness=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    model = OneLayer(10).to(device)
    model.load_state_dict(torch.load(os.path.join("dataset", "MNIST", "state_dict", "model_weights.pth")))

    img, out = random_digit(model=model)

    print(out)
    intermediate = preprocessed_img(img)
    fig, ax =plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(intermediate)

    ax[0].axis(False)
    ax[1].axis(False)

    plt.show()