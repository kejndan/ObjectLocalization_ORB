import numpy as np
import cv2
import matplotlib.pyplot as plt


class Matcher:
    def __init__(self, keypoints_query, keypoints_test, lowe_tau=0.75, test_lowe=True, cross_check=True):
        self.keypoints_query = keypoints_query
        self.keypoints_test = keypoints_test
        self.lowe_tau = lowe_tau
        self.test_lowe = test_lowe
        self.cross_check = cross_check
        self.__search_points()
        self.best_match = None

    @staticmethod
    def dist_hamming(arg1, arg2):
        return np.count_nonzero(arg1 != arg2)

    def __search_points(self):
        self.keypoints_query_select = [tuple(kp.coords) for kp in self.keypoints_query]
        self.keypoints_test_select = [tuple(kp.coords) for kp in self.keypoints_test]

    @staticmethod
    def __get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    def fit(self):
        best_match = {}
        distance_matrix = np.zeros((len(self.keypoints_query), len(self.keypoints_test)))
        for i in range(len(self.keypoints_query)):
            for j in range(len(self.keypoints_test)):
                distance_matrix[i, j] = self.dist_hamming(self.keypoints_query[i].descriptor,
                                                          self.keypoints_test[j].descriptor)
        for i in range(len(self.keypoints_query)):
            idxs_sorted_test = np.argsort(distance_matrix[i])
            if self.test_lowe and distance_matrix[i, idxs_sorted_test[0]] \
                    / distance_matrix[i, idxs_sorted_test[1]] >= self.lowe_tau:
                continue
            if self.cross_check:
                idxs_sorted_query = np.argsort(distance_matrix[:, idxs_sorted_test[0]])
                if idxs_sorted_query[0] != i:
                    continue
            best_match[tuple(self.keypoints_query[i].coords.tolist())] = tuple(
                self.keypoints_test[idxs_sorted_test[0]].coords.tolist())
        self.best_match = best_match
        print('Matches: ', len(self.best_match))

    def get_matched_keypoints(self):
        keypoints_query = []
        keypoints_test = []
        for type_keypoints in ['query', 'test']:
            if type_keypoints == 'query':

                coords = [tuple(kp.coords) for kp in self.keypoints_query]
                for coord in self.best_match.keys():
                    keypoints_query.append(self.keypoints_query[coords.index(coord)])
            else:

                coords = [tuple(kp.coords) for kp in self.keypoints_test]
                for coord in self.best_match.values():
                    keypoints_test.append(self.keypoints_test[coords.index(coord)])
        return keypoints_query, keypoints_test









