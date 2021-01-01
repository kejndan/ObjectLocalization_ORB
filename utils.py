import matplotlib.pyplot as plt
import numpy as np
import cv2

def read_image(path, color=False):
    if color:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=np.float32)
    else:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), dtype=np.float64)
    return image

def show_image(image, title=None, cmap=None, textbox=None):
    plt.figure(figsize=(17, 10))
    if cmap is not None:
        plt.imshow(np.uint8(image),cmap=cmap)
    else:
        plt.imshow(np.uint8(image))
    if title is not None:
        plt.title(title)

    # plt.axis('off')
    plt.show()


def uniq_elems(points):
    points = list(set(points))
    points.sort(key = lambda x: x[2])

    new_points = []
    prev_point = []
    # print(len(points))
    for point in points:
        if point[2] < -90:
            prev_point.append(list(point)[:2])
        else:
            new_points.append(list(point)[:2])
    # print(len(new_points))
    # new_points = np.array(new_points + prev_point)
    return new_points + prev_point

def all_comb(x_center, y_center, x, y):

    points = [(x_center + x, y_center + y, np.rad2deg(np.arctan2(y,x))),
              (x_center - x, y_center + y, np.rad2deg(np.arctan2(y,-x))),
              (x_center + x, y_center - y, np.rad2deg(np.arctan2(-y,x))),
              (x_center - x, y_center - y, np.rad2deg(np.arctan2(-y,-x))),
              (x_center + y, y_center + x, np.rad2deg(np.arctan2(x,y))),
              (x_center - y, y_center + x, np.rad2deg(np.arctan2(x,-y))),
              (x_center + y, y_center - x, np.rad2deg(np.arctan2(-x,y))),
              (x_center - y, y_center - x, np.rad2deg(np.arctan2(-x,-y)))
              ]

    return points




def get_point_circle(x_center, y_center, r=3):
    x, y = 0, r
    d = 3 - 2*r
    points_circle = []
    points_circle.extend(all_comb(x_center,y_center,x,y))
    while y >=x:

        x += 1
        if d > 0:
            y -= 1
            d += 4*(x-y) + 10
        else:
            d += 4*x + 6
        points_circle.extend(all_comb(x_center,y_center,x,y))

    return uniq_elems(points_circle)

def calc_grads_img(img):
    y_filter = np.array(([-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]))
    x_filter = y_filter.T
    grad_y = cv2.filter2D(img, ddepth=-1, kernel=y_filter)
    grad_x = cv2.filter2D(img, ddepth=-1, kernel=x_filter)
    return grad_x, grad_y

def create_help_matrix(matrix, n):
    new_matrix = np.zeros(matrix.shape)
    height, width = matrix.shape
    for i in range(0, height, n + 1):
        for j in range(0, width, n + 1):
            end_j2 = min(j + n + 1, width)
            for j2 in range(j, end_j2):
                upper = i
                upper_max = matrix[upper, j2]
                new_matrix[upper, j2] = upper_max
                upper += 1
                lower = min(i + n, height - 1)
                lower_max = matrix[lower, j2]
                new_matrix[lower, j2] = lower_max
                lower -= 1

                while upper != n // 2 + i and upper != height:
                    if matrix[upper, j2] > upper_max:
                        upper_max = matrix[upper, j2]
                    else:
                        new_matrix[upper, j2] = upper_max

                    if matrix[lower, j2] > lower_max:
                        lower_max = matrix[lower, j2]
                    else:
                        new_matrix[lower, j2] = lower_max
                    upper += 1
                    if lower == n // 2 + 1:
                        lower -= 1
                if upper == height:
                    upper -= 1
                new_matrix[upper, j2] = max(upper_max, lower_max, matrix[upper, j2])
    return new_matrix


def nms_2d(matrix, size_filter=29, tau=80):
    new_matrix = np.zeros(matrix.shape)
    maxes = []
    height, width = matrix.shape
    n = (size_filter - 1) // 2
    accum_matrix = create_help_matrix(matrix, n)

    for i in range(0, height, n + 1):
        for j in range(0, width, n + 1):
            is_not_max = False
            mi, mj = i, j

            for i2 in range(i, min(i + n + 1, height)):
                for j2 in range(j, min(j + n + 1, width)):
                    if matrix[i2, j2] > matrix[mi, mj]:
                        mi, mj = i2, j2
            i2 = max(mi - n, 0)
            max_i2 = min(mi + n, height - 2) + 1
            while i2 < max_i2:
                j2 = max(mj - n, 0)
                max_j2 = min(mj + n, width - 2) + 1
                while j2 < max_j2:
                    if i2 < i:
                        if i2 < i - n // 2 - 1:
                            if matrix[i2, j2] > matrix[mi, mj]:
                                mi, mj = i2, j2
                                is_not_max = True
                                break
                        elif i2 == i - n // 2 - 1:
                            if matrix[i2, j2] == accum_matrix[i2, j2]:
                                if matrix[i2, j2] > matrix[mi, mj]:
                                    mi, mj = i2, j2
                                    is_not_max = True
                                    break
                        else:
                            if matrix[i2, j2] > matrix[mi, mj]:
                                mi, mj = i2, j2
                                is_not_max = True
                                break
                            if j2 + 1 == max_j2:
                                i2 = i - 1
                    elif i2 > i + n:
                        if i2 == i + n + n // 2 or i2 + 1 == max_i2:
                            if matrix[i2, j2] > matrix[mi, mj]:
                                mi, mj = i2, j2
                                is_not_max = True
                                break
                            if j2 + 1 == max_j2:
                                i2 += 1
                        elif i2 == i + n + n // 2 + 1:
                            if matrix[i2, j2] == accum_matrix[i2, j2]:
                                if matrix[i2, j2] > matrix[mi, mj]:
                                    mi, mj = i2, j2
                                    is_not_max = True
                                    break
                        elif i2 > i + n + n // 2 + 1:
                            if matrix[i2, j2] > matrix[mi, mj]:
                                mi, mj = i2, j2
                                is_not_max = True
                                break
                    elif j2 < j or j2 > j + n:
                        if matrix[i2, j2] > matrix[mi, mj]:
                            mi, mj = i2, j2
                            is_not_max = True
                            break
                        if j2 + 1 == max_j2:
                            i2 = i + n
                    j2 += 1
                if is_not_max:
                    break
                i2 += 1
            if not is_not_max and tau <= matrix[mi, mj]:
                maxes.append((mi, mj, matrix[mi, mj]))
    for t in maxes:
        i, j, m = t
        new_matrix[i, j] = m
    return new_matrix

def get_all_points_in_circle(x_center, y_center, img_shape, r=31):
    all_points = []
    # size_square = r*2 + 1
    # half_diag_square = size_square * np.sqrt(2) / 2
    # if not (x_center - r >= 0 and x_center + r < img_shape[1] and \
    #     y_center - r >= 0 and y_center + r < img_shape[0]):
    #     return None
    points = np.array(get_point_circle(x_center, y_center, r))


    all_points.extend(points)
    points = np.array(points)
    y_min = y_center - r if y_center - r >= 0 else 0
    y_max = y_center + r if y_center + r < img_shape[0] else img_shape[0] -1

    for row in range(y_min + 1, y_max):
        points_current_row = points[points[:, 1] == row]

        x_min = points_current_row[1, 0]
        x_max = points_current_row[0, 0]
        for col in range(x_min+1, x_max):
            all_points.append([row,col])
    points = np.asarray(all_points)
    points = np.where(points < 0, 0,points)
    points = np.where(points >= img_shape[0], img_shape[0] - 1, points)
    points = np.where(points >= img_shape[1], img_shape[1] - 1, points)
    return points
    # else: return None


# def pyramid_scale(img, scale=1.2, begin_lvl=0, end_lvl=8):
#     current_shape = img.shape
#     current_lvl = begin_lvl
#     scaled_img = img
#
#     while current_lvl < end_lvl and current_shape[0] != 1 and current_shape[1] != 1:
#         current_shape = (int(img.shape[0] // np.power(scale, current_lvl)),
#                              int(img.shape[1] // np.power(scale, current_lvl)))
#         scaled_img = cv2.resize(scaled_img, current_shape[::-1])
#         if current_lvl != 0:
#             scaled_img = cv2.GaussianBlur(scaled_img, (3, 3), 1.5)
#         current_lvl += 1
#         return scaled_img, np.power(scale, current_lvl-1)

def pyramid_scale(img, scale=1.2, begin_lvl=0, end_lvl=8):
    current_shape = img.shape
    current_lvl = begin_lvl
    scaled_img = img

    while current_lvl < end_lvl and current_shape[0] != 1 and current_shape[1] != 1:
        current_shape = (int(img.shape[0] // np.power(scale, current_lvl)),
                             int(img.shape[1] // np.power(scale, current_lvl)))
        scaled_img = cv2.resize(scaled_img, current_shape[::-1])
        if current_lvl != 0:
            scaled_img = cv2.GaussianBlur(scaled_img, (3, 3), 1.5)
        current_lvl += 1
        return scaled_img, np.power(scale, current_lvl-1)




def draw_matches(matches, query_img, test_img, coord_localization=None,title=None):
    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    img = np.zeros((max(query_img.shape[0], test_img.shape[0]), query_img.shape[1] + test_img.shape[1], 3))
    img[:query_img.shape[0], :query_img.shape[1]] = query_img
    img[:test_img.shape[0], query_img.shape[1]:query_img.shape[1] + test_img.shape[1]] = test_img
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(np.uint8(img))
    cmap = get_cmap(len(matches[0]))
    for type_img in ['query', 'test']:
        keypoints_data = matches[0] if type_img == 'query' else matches[1]
        for i, keypoint in enumerate(keypoints_data):
            coords_begin = keypoint.coords * keypoint.scale_factor
            r = keypoint.scale_factor * 3
            y1 = coords_begin[0]
            x1 = coords_begin[1]
            if type_img == 'test':
                x1 += query_img.shape[1]
                ax.plot([matches[0][i].coords[1] * matches[0][i].scale_factor, x1],
                        [matches[0][i].coords[0] * matches[0][i].scale_factor, y1], color=cmap(i), linewidth=0.5)
            c = plt.Circle((x1, y1), r, fill=False, linewidth=1, color=cmap(i))
            ax.add_patch(c)
    if coord_localization is not None:
        coord_localization[1, :] += query_img.shape[1]

        ax.plot([coord_localization[1][0], coord_localization[1][1]],
                [coord_localization[0][0], coord_localization[0][1]], color='y', linewidth=3)
        ax.plot([coord_localization[1][1], coord_localization[1][3]],
                [coord_localization[0][1], coord_localization[0][3]], color='y', linewidth=3)
        ax.plot([coord_localization[1][3], coord_localization[1][2]],
                [coord_localization[0][3], coord_localization[0][2]], color='y', linewidth=3)
        ax.plot([coord_localization[1][0], coord_localization[1][2]],
                [coord_localization[0][0], coord_localization[0][2]], color='y', linewidth=3)
    if title is not None:
        plt.title(title)
    plt.show()