import numpy as np
import cv2
import ffmpeg
import os
from matplotlib import pyplot as plt
import shutil


def normalize(A):
    res = A - np.min(A)
    return (255 * res / np.max(res))


def get_dis(y1, x1, y0, x0):
    return np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)


def index_1D_to_2D(ind):
    k = 3
    r, c = np.unravel_index(int(ind), (k, k))
    return r - k // 2, c - k // 2


def get_line_points(p0, p1):
    dis = get_dis(p1[1], p1[0], p0[1], p0[0])
    num = int(dis / 4)
    return np.array(list(zip(np.linspace(p0[0], p1[0], num), np.linspace(p0[1], p1[1], num))))


def select_points(img):
    def click_event(event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            r, c = y, x
            if len(selected_points) > 0 and get_dis(r, c, selected_points[-1][0], selected_points[-1][1]) < 5:
                cv2.destroyAllWindows()
            else:
                if len(selected_points) == 0:
                    selected_points.append([r, c])
                else:
                    new_points = get_line_points(selected_points[-1], [r, c])

                    selected_points.extend(new_points[1:].tolist())

    selected_points = []
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(selected_points).astype(int)


def get_dis2(p1, p2, d):
    return (np.sum((p1 - p2) ** 2) - d) ** 2


def get_dis2_wo_d(p1, p2):
    return np.sum((p1 - p2) ** 2)


def get_dis3(p1, p2, p3):
    return np.sum((p1 - 2 * p2 + p3) ** 2)


def get_all_dis2_wo_d(p0, p1_s):
    return np.sum((p1_s - p0) ** 2, axis=1)


def get_all_dis2(p0, p1_s, d):
    return (np.sum((p1_s - p0) ** 2, axis=1) - d) ** 2


def get_all_dis3(p1, p2, p3_s):
    return np.sum((p3_s - 2 * p2 + p1) ** 2, axis=1)


def get_point(p, neigh_ind):
    return p + index_1D_to_2D(neigh_ind)


def get_path_simple(points, par, min_config):
    i0, i1 = min_config
    n = len(points)
    new_points = np.zeros_like(points)
    new_points[-1] = get_point(points[n - 1], i1)
    for i in range(n - 2, -1, -1):
        next_ind = par[int(i + 1)][int(i0)][int(i1)]
        new_points[i] = get_point(points[i], next_ind)
        i1 = next_ind
    return new_points


def get_deltas(k):
    deltas = []
    for i in range(k):
        for j in range(k):
            deltas.append([i - k // 2, j - k // 2])
    return np.array(deltas)


def get_inside_points(points):
    shifted_points = np.roll(points, 1, axis=0)
    mid_points = (points + shifted_points) / 2
    vecs = points - shifted_points
    vecs[:, [0, 1]] = vecs[:, [1, 0]]
    vecs[:, 1] = -vecs[:, 1]
    vecs = 10 * vecs / ((np.linalg.norm(vecs, axis=1) + 0.00001)[:, np.newaxis])
    return mid_points + vecs


def move_points_simple(points, gradients, lam, alpha, d, gamma):
    r = 3
    a = r ** 2
    n = len(points)
    dp = np.ones((n, a, a)) * np.inf
    par = np.zeros_like(dp, dtype=int)
    for i in range(a):
        p0 = get_point(points[0], i)
        dp[0][i][i] = -lam * gradients[p0[0], p0[1]]

    deltas = get_deltas(3)
    min_ans = np.inf
    min_config = np.array([-1, -1], dtype=int)

    inside_points = get_inside_points(points)

    inside_dists = np.zeros((len(inside_points), a))
    for i in range(len(inside_points)):
        inside_dists[i] = get_all_dis2_wo_d(inside_points[i], deltas + points[i - 1])

    all_dists = np.zeros((len(points), a, a))
    for i in range(len(points)):
        for j in range(a):
            p_y = get_point(points[i], j)
            all_dists[i][j] = get_all_dis2(p_y, deltas + points[i - 1], d)

    for i in range(1, n):
        for j in range(a):
            p0 = get_point(points[0], j)
            for y in range(a):
                p_y = get_point(points[i], y)

                temp = dp[i - 1, j, :] + alpha * all_dists[i][y] + gamma * inside_dists[i]

                t = np.argmin(temp)
                par[i][j][y] = t
                dp[i][j][y] = temp[t] - lam * gradients[p_y[0], p_y[1]] + gamma * get_dis2_wo_d(inside_points[i], p_y)

                if i == n - 1:
                    dp[i][j][y] = dp[i][j][y] + alpha * get_dis2(p0, p_y, d) + gamma * (
                            get_dis2_wo_d(inside_points[0], p_y) + get_dis2_wo_d(p0, inside_points[0]))
                    if dp[i][j][y] < min_ans:
                        min_ans = dp[i][j][y]
                        min_config = j, y

    new_points = get_path_simple(points, par, min_config)
    return new_points


def get_gradients(B):
    A = B.copy()
    if len(A.shape) > 2:
        A = np.average(A, axis=2)
    x_filter = np.array([[0, 0, 0],
                         [1, 0, -1],
                         [0, 0, 0]])
    y_filter = np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, -1, 0]])

    g1 = cv2.filter2D(A, -1, x_filter)
    g2 = cv2.filter2D(A, -1, y_filter)
    g = (g1 ** 2 + g2 ** 2)
    g = normalize(g)
    g = cv2.GaussianBlur(g, (15, 15), 0)

    g = normalize(g)
    g = cv2.blur(g, (15, 15), 0)
    g = normalize(g)

    g = cv2.medianBlur(g.astype(np.uint8), 15, 0)

    g[g < 0.1 * np.max(g)] = 0
    g[g > 0] = 255
    g = cv2.Canny(g.astype(np.uint8), 0.5, 0.7)

    return g


def get_avg_dists(points):
    shifted_points = np.roll(points, 1, axis=0)
    return np.sum((points - shifted_points) ** 2) / len(points)


def draw_curve(I, points):
    cv_points = points[:, [1, 0]]
    I_curve = I.copy()
    for i in range(len(cv_points)):
        I_curve = cv2.line(I_curve, tuple(cv_points[i]), tuple(cv_points[i - 1]), color=(0, 0, 255), thickness=2)
    return I_curve


def active_contour(path, lam, alpha, gamma_itr, gamma_val):
    dir_name = "temp_contour"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    I = cv2.imread(path)
    gradients = get_gradients(I)
    points = select_points(I)

    d = get_avg_dists(points)

    gamma = 0
    for i in range(401):
        print('iteration:', i)
        if i == gamma_itr:
            gamma = gamma_val
        d = d * 0.95

        points = move_points_simple(points, gradients, lam, alpha, d, gamma)
        if i % 10 == 0:
            cv2.imwrite(dir_name + '/' + (str(i // 10 + 1) if i // 10 + 1 > 9 else "0" + str(i // 10 + 1)) + '.jpg',
                        draw_curve(I, points))

    (
        ffmpeg
            .input(f'{dir_name}/*.jpg', pattern_type='glob', framerate=5)
            .output('contour.mp4')
            .overwrite_output()
            .global_args('-loglevel', 'quiet')
            .run()
    )
    cv2.imwrite('res11.jpg', draw_curve(I, points))

    shutil.rmtree(dir_name)

    return points


active_contour('tasbih.jpg', lam=100000, alpha=1, gamma_itr=200, gamma_val=100)
