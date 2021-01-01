import numpy as np
import cv2
from hw_6.utils import get_point_circle, calc_grads_img, nms_2d, get_all_points_in_circle


class FeatureDetector:
    def __init__(self,tau=10, radius=3,k=0.04,patch_size=31,harris_tau=512):
        self.tau = tau
        self.radius = radius
        self.k = k
        self.harris_tau = harris_tau
        self.features_fast = []
        self.features = []
        self.response = []
        self.response_matrix = None
        self.features_harris = []
        self.patch_size = patch_size

    def clear_attributes(self):
        self.features_fast = []
        self.features = []
        self.response = []
        self.response_matrix = []
        self.features_harris = []



    def _fast_lite(self, img):
        for x in range(self.radius, img.shape[1] - self.radius):
            for y in range(self.radius, img.shape[0] - self.radius):
                points = get_point_circle(x,y,self.radius)
                size_square = 31
                half_diag_square = int(size_square * np.sqrt(2) / 2)

                if x + half_diag_square >= img.shape[1] or x - half_diag_square < 0 or \
                        y + half_diag_square >= img.shape[0] or y - half_diag_square< 0:
                    continue
                for i in range(len(points)):
                    flag_end = False
                    for j in range(i, i + len(points) // 4 * 3):

                        if i == j:
                            if img[points[j % len(points)][1], points[j % len(points)][0]] - img[y, x] > self.tau:
                                greater = True
                            elif img[points[j % len(points)][1], points[j % len(points)][0]] - img[y, x] < -self.tau:
                                greater = False
                            else:
                                break
                        else:
                            if greater and img[points[j % len(points)][1], points[j % len(points)][0]] - img[y, x]\
                                    <= self.tau:
                                break
                            elif not greater and img[points[j % len(points)][1], points[j % len(points)][0]] - img[y, x]\
                                    >= - self.tau:
                                break
                        if j == i + len(points) // 4 * 3 - 1:
                            self.features_fast.append((y,x))
                            flag_end = True
                    if flag_end:
                        break
        self.features = np.array(list(set(self.features_fast.copy())))
        return np.array(list(set(self.features_fast)))
    
    def fast_detector(self, img):
        for x in range(self.radius, img.shape[1] - self.radius):
            for y in range(self.radius, img.shape[0] - self.radius):


                size_square = 31
                half_diag_square = int(size_square * np.sqrt(2) / 2)
                if x + half_diag_square >= img.shape[1] or x - half_diag_square < 0 or \
                        y + half_diag_square >= img.shape[0] or y - half_diag_square< 0:
                    continue

                calc_diag = 0
                if img[y - self.radius, x] - img[y, x] > self.tau:
                    calc_diag += 1
                if img[y + self.radius, x] - img[y, x] > self.tau:
                    calc_diag += 1
                if img[y, x + self.radius] - img[y, x] > self.tau:
                    calc_diag += 1
                if img[y, x - self.radius] - img[y, x] > self.tau:
                    calc_diag += 1

                if calc_diag < 3:
                    greater = False
                    calc_diag = 0
                    if img[y - self.radius, x] - img[y, x] < -self.tau:
                        calc_diag += 1
                    if img[y + self.radius, x] - img[y, x] < -self.tau:
                        calc_diag += 1
                    if img[y, x + self.radius] - img[y, x] < -self.tau:
                        calc_diag += 1
                    if img[y, x - self.radius] - img[y, x] < -self.tau:
                        calc_diag += 1
                else:
                    greater = True
                if calc_diag < 3:
                    continue

                points = np.array(get_point_circle(x, y, self.radius))
                len_points = len(points)
                intensity = img[points[:,1],points[:,0]]

                if greater:
                    satisfy_values = intensity > img[y,x] + self.tau
                else:
                    satisfy_values = intensity < img[y,x] - self.tau
                count_nonzero = np.count_nonzero(satisfy_values)
                if len_points - count_nonzero < 2:
                    self.features_fast.append((y, x))
                    self.response.append(len_points - count_nonzero)
                elif len_points - count_nonzero <= 4:
                    idxs_unsatisfy = np.where(satisfy_values == False)[0]
                    for i in range(len(idxs_unsatisfy)-1):
                        if i == 0:
                            if len_points - 1 - idxs_unsatisfy[-1] + idxs_unsatisfy[i] >= len_points// 4 * 3:
                                self.features_fast.append((y,x))
                                self.response.append(len_points - 1 - idxs_unsatisfy[-1] + idxs_unsatisfy[i])

                        if idxs_unsatisfy[i+1] - idxs_unsatisfy[i] - 1 >= len_points// 4 * 3:
                            self.features_fast.append((y,x))
                            self.response.append(idxs_unsatisfy[i+1] - idxs_unsatisfy[i] - 1)
        if self.features_fast != []:
            self.__display_response2matrix(img, np.array(self.features_fast), np.array(self.response))
        self.features = self.features_fast.copy()
        return np.array(self.features)

    def __harris_selection_w_sort(self, r, points):
        sorted_r = r.argsort()
        self.features_harris = points[sorted_r[-self.harris_tau:]].tolist()
        self.response_harris = r[sorted_r[-self.harris_tau:]]
        # print(r[sorted_r[-n:]].max(),r[sorted_r[-n:]].min())
        return self.features_harris

    def __display_response2matrix(self, img, points, response):
        clear_img = np.zeros(img.shape)
        clear_img[points[:,0],points[:,1]] = np.array(response)
        self.response_matrix = clear_img
        return clear_img

    def __display_matrix2points(self,matrix):
        return np.concatenate((np.where(matrix != 0)[0][:,np.newaxis], np.where(matrix !=0)[1][:,np.newaxis]), axis=1)

    def non_max_suppression(self, size_filter=5):
        self.features_fast = self.__display_matrix2points(nms_2d(self.response_matrix, size_filter, 0)).tolist()
        self.features = self.features_fast.copy()
        return self.features_fast

    def __harris_selection_w_thr(self, r, points, thr=0):
        pass

    def draw_key_points(self, img, color=[255,0,0]):
        for key in np.array(self.features):
            points = np.array(get_point_circle(key[1], key[0]))
            img[points[:, 1], points[:, 0]] = color
        return img

    def draw_key_points_w_orient(self, img, color=[255,0,0]):
        for i,key in enumerate(np.array(self.features)):
            points = np.array(get_point_circle(key[1], key[0]))
            img[points[:, 1], points[:, 0]] = color
        return img

    def harris_detector(self, img):
        if self.features_fast == []:
            points = []
            for x in range(self.radius, img.shape[1] - self.radius):
                for y in range(self.radius, img.shape[0] - self.radius):
                    points.append([y, x])
            points = np.array(points)
        else:
            points = np.array(self.features_fast.copy())
            
        grad_x, grad_y = calc_grads_img(img)

        d = np.array([[1, 4, 6, 4, 1]]) / 16
        w = np.power(np.dot(d.T, d), 2)

        g_dx2 = cv2.filter2D(grad_x * grad_x, ddepth=-1, kernel=w)
        g_dy2 = cv2.filter2D(grad_y * grad_y, ddepth=-1, kernel=w)
        g_dxdy = cv2.filter2D(grad_x * grad_y, ddepth=-1, kernel=w)

        first_oper = (g_dx2[points[:, 0], points[:, 1]] + g_dy2[points[:, 0], points[:, 1]]) / 2
        second_oper = np.sqrt(4 * g_dxdy[points[:, 0], points[:, 1]] * g_dxdy[points[:, 0], points[:, 1]]
                              + np.power((g_dx2[points[:, 0], points[:, 1]] - g_dy2[points[:, 0], points[:, 1]]), 2)) / 2
        lambda1 = first_oper + second_oper
        lambda2 = first_oper - second_oper

        det_m = lambda1 * lambda2
        trace_m = lambda1 + lambda2

        r = det_m - self.k * trace_m * trace_m
        self.features = self.__harris_selection_w_sort(r,points).copy()

        return np.array(self.features)

    def get_points(self):
        points = []
        coords = np.arange(-self.patch_size, self.patch_size + 1)
        for i in range(len(coords)):
            for j in range(len(coords)):
                if coords[i]*coords[i] + coords[j]*coords[j] <= self.patch_size*self.patch_size:
                    # if 0 <= x_coords[i] + x < shape[0] and 0 <= y_coords[j] + y < shape[1]:
                    points.append([coords[i],coords[j]])
        return np.array(points)

    def points_in_circle(self,points,shape):
        new_points = []
        for point in points:
            # q = np.logical_and(0 <= point, point[0] < shape[0], point[1] < shape[1])
            if np.all(0 <= point) and  point[0] < shape[0] and point[1] < shape[1]:
                new_points.append(point)
        return np.array(new_points)



    def calc_orientation_key_points(self, img):
        thetas = []
        idx_accepted = []
        img_shape = img.shape
        points = self.get_points()
        for i,feature in enumerate(self.features):
            if img.shape[0] - feature[0] <= self.patch_size or img.shape[1] - feature[1] <= self.patch_size or \
                feature[0] < self.patch_size or feature[1] < self.patch_size:
                working_points = self.points_in_circle(points.copy()+ feature,img_shape)
            else:
                working_points = points.copy() + feature
            if points is not None:
                m01 = ((working_points[:,1]-feature[1])*img[working_points[:,0],working_points[:,1]]).sum()

                m10 = ((working_points[:,0]-feature[0])*img[working_points[:,0],working_points[:,1]]).sum()

                thetas.append(np.arctan2(m01,m10))
                idx_accepted.append(i)
        self.thetas = np.asarray(thetas)
        self.idx_accepted = np.array(idx_accepted)
        if len(self.idx_accepted) != 0:
            self.features = np.array(self.features)[self.idx_accepted]
        else:
            self.features = []
        return thetas



            
                