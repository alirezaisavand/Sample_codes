import cv2
import numpy as np
import os
from scipy.spatial import Delaunay
import shutil
import ffmpeg


def read_points(path):
    return np.loadtxt(path).reshape(-1, 2).astype(np.int)


def show_triangulation(I, p, tri):
    I2 = I.copy()
    p2 = p.copy()
    p2[:, [1, 0]] = p2[:, [0, 1]]
    triangles = p2[tri.simplices]
    mask = np.zeros_like(I2)
    for i, t in enumerate(triangles):
        cv2.drawContours(mask, [t.astype(np.int)], 0, (i + 1, i + 1, i + 1), -1)

    return mask[:, :, 0]


def transform(I, mask, P_src, P_dest, tri, ind):
    I2 = I.copy()
    P2_src = P_src.copy()
    P2_src[:, [1, 0]] = P2_src[:, [0, 1]]

    P2_dest = P_dest.copy()
    P2_dest[:, [1, 0]] = P2_dest[:, [0, 1]]

    src_triangles = P2_src[tri.simplices]
    dest_triangles = P2_dest[tri.simplices]

    matrix = cv2.getAffineTransform(src_triangles[ind].astype(np.float32), dest_triangles[ind].astype(np.float32))

    result_I = cv2.warpAffine(I2, matrix, (I.shape[1], I.shape[0]))

    triangle_mask = (mask == (ind + 1)) * 1

    triangle_mask = np.dstack((triangle_mask, triangle_mask, triangle_mask))
    result = triangle_mask * result_I

    return result


def save_video(dir_name, all_images):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for ind, im in enumerate(all_images):
        pref = '0'
        if ind >= 10:
            pref = ''
        cv2.imwrite(dir_name + '/' + pref + str(ind) + '.jpg', im)

    (
        ffmpeg
            .input(f'{dir_name}/%2d.jpg', framerate=20)
            .output('morph.mp4')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
    )

    shutil.rmtree(dir_name)


def get_middle_image(I1, I2, tri, p1, p2, alpha):
    p = alpha * (p2 - p1) + p1

    tri_mask = show_triangulation(I1, p, tri)

    image = np.zeros_like(I1)
    for j in range(len(tri.simplices)):
        i1 = transform(I1, tri_mask, p1, p, tri, j)
        i2 = transform(I2, tri_mask, p2, p, tri, j)
        image = image + i1 * (1 - alpha) + i2 * alpha
    return image


def morph(path1, path2, steps, dir_name):
    I1 = cv2.imread(path1)[:-1]

    p_extra = np.array([
        [0, 0],
        [0, I1.shape[1] - 1],
        [I1.shape[0] - 1, 0],
        [I1.shape[0] - 1, I1.shape[1] - 1],
    ])

    p1 = read_points('points1.txt')
    p1 = np.concatenate((p1, p_extra), axis=0)

    tri = Delaunay(p1)

    I2 = cv2.imread(path2)[:-1]
    I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    p2 = read_points('points2.txt')
    p2 = np.concatenate((p2, p_extra), axis=0)

    all_images = []

    for i in range(1, steps + 1):
        alpha = i / steps
        image = get_middle_image(I1, I2, tri, p1, p2, alpha)
        all_images.append(image)
        if i == 15:
            cv2.imwrite('res03.jpg', image)
        if i == 30:
            cv2.imwrite('res04.jpg', image)

    all_images = all_images + all_images[::-1]

    save_video(dir_name, all_images)


morph('res01.jpeg', 'res02.jpg', 45, 'Morphing')
