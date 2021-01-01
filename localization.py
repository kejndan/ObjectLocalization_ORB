import numpy as np
import cv2

class Localization:
    def __init__(self, keypoints_query, keypoints_test, type_transform='projective'):
        self.keypoints_query = keypoints_query
        self.keypoints_test = keypoints_test
        self.type_transform = type_transform
        self.transform_matrix = None
        self.inlier_keypoints_query = None
        self.inlier_keypoints_test = None
        self.max_inliers = None

    def LES(self, idxs_matches, LS=False):
        if self.type_transform == 'affine':
            col_in_A = 6
        elif self.type_transform == 'projective':
            col_in_A = 8
        A = np.zeros((len(idxs_matches) * 2, col_in_A))
        b = np.zeros((len(idxs_matches)  * 2, 1))
        row = 0
        for i, idx in enumerate(idxs_matches):
            x, y = self.keypoints_query[idx].coords * self.keypoints_query[idx].scale_factor
            u, v = self.keypoints_test[idx].coords * self.keypoints_test[idx].scale_factor
            if self.type_transform == 'affine':
                A[row] = np.array([x, y, 0, 0, 1, 0])
                A[row + 1] = np.array([0, 0, x, y, 0, 1])
            elif self.type_transform == 'projective':
                A[row] = np.array([x, y, 1, 0, 0, 0, -x * u, -y * u])
                A[row + 1] = np.array([0, 0, 0, x, y, 1, -x * v, -y * v])
            b[row] = u
            b[row + 1] = v
            row += 2

        if LS:
            params = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            params = np.linalg.solve(A, b)

        if self.type_transform == 'affine':
            trans = np.array([[params[0][0], params[1][0], params[4][0]],
                              [params[2][0], params[3][0],params[5][0]]
                              ])
        elif self.type_transform == 'projective':
            trans = np.array([[params[0][0], params[1][0], params[2][0]],
                              [params[3][0], params[4][0], params[5][0]],
                              [params[6][0], params[7][0], 1]
                              ])

        return trans


    def fit(self, iterations=2000, proj_threshold=3,seed=None):
        if seed is not None:
            np.random.seed(seed)

        max_inliers = []
        for i in range(iterations):
            if self.type_transform == 'affine':
                select_pair = 3
            elif self.type_transform == 'projective':
                select_pair = 4
            idxs_matches = np.random.choice(np.arange(len(self.keypoints_query)), select_pair, False)
            try:
                A = self.LES(idxs_matches)
                coord_keypoints_test = np.array(
                    [self.keypoints_test[i].coords * self.keypoints_test[i].scale_factor for i in range(len(self.keypoints_test))]).T
                coord_keypoints_query = np.array(
                    [self.keypoints_query[i].coords * self.keypoints_query[i].scale_factor for i in range(len(self.keypoints_query))]).T
                # if self.type_transform == 'projective':
                ones = np.ones(coord_keypoints_query.shape[1])

                coord_keypoints_query = np.vstack((coord_keypoints_query, ones))
                coord_keypoints_test = np.vstack((coord_keypoints_test, ones))

                calc_coord_keypoints_test = A @ coord_keypoints_query
                if self.type_transform == 'affine':
                    calc_coord_keypoints_test = np.vstack((calc_coord_keypoints_test, ones))
                elif self.type_transform == 'projective':
                    calc_coord_keypoints_test /= calc_coord_keypoints_test[2]

                inliers = []
                for j in range(len(self.keypoints_query)):
                    if np.linalg.norm(calc_coord_keypoints_test[:, j] - coord_keypoints_test[:, j]) < proj_threshold:
                        inliers.append(j)
                if len(inliers) > len(max_inliers):
                    max_inliers = inliers.copy()
            except np.linalg.LinAlgError:
                pass


        self.transform_matrix = self.LES(max_inliers, LS=True).copy()
        self.max_inliers = max_inliers.copy()

    def get_inlier_keypoints(self):
        self.inlier_keypoints_query = [self.keypoints_query[i] for i in self.max_inliers]
        self.inlier_keypoints_test = [self.keypoints_test[i] for i in self.max_inliers]
        return self.inlier_keypoints_query, self.inlier_keypoints_test


    def predict(self,image):
        points_on_query = np.array([[0, 0],
                                    [image.shape[0], 0],
                                    [0, image.shape[1]],
                                    [image.shape[0], image.shape[1]]]).T
        ones = np.ones(4)
        points_on_query = np.vstack((points_on_query, ones))
        points_on_test = self.transform_matrix @ points_on_query
        if self.type_transform == 'projective':
            points_on_test = points_on_test / points_on_test[2]
        return points_on_test



