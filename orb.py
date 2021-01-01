import numpy as np
import cv2
from hw_6.detector import FeatureDetector
from hw_6.descriptor import DescriptorKeypoints
from hw_6.keypoint import Keypoint
import matplotlib.pyplot as plt
import pickle

class ORB:
    def __init__(self, lvls_pyramid=8, scale_factor=1.2,sectors_angle=30,radius=3,tau=30,harris_tau=512,harris_k=0.04,
                 patch_size=31, len_descriptor=256,create_points_brief='normal'):
        self.lvls_pyramid = lvls_pyramid
        self.scale_factor = scale_factor
        self.sectors_angle = sectors_angle
        self.keypoints_data = []
        self.radius = radius
        self.tau = tau
        self.harris_tau = harris_tau
        self.harris_k = harris_k
        self.patch_size = patch_size
        self.len_descriptor = len_descriptor
        self.create_points_brief = create_points_brief

    def fit(self, img):
        self.keypoints_data = []
        height, width = img.shape
        current_img = img.copy()
        detector = FeatureDetector(self.tau,self.radius,self.harris_k,self.patch_size,self.harris_tau)
        descriptor = DescriptorKeypoints(self.len_descriptor,self.sectors_angle,self.patch_size,self.create_points_brief)
        for t in range(1, self.lvls_pyramid+1):
            print(f'Scale {np.power(1.2, t - 1)} {current_img.shape}')
            keys = detector.fast_detector(current_img.copy())
            if len(keys) != 0:
                print(f'Nums after FAST {len(keys)}')
                # detector.non_max_suppression()
                # print(f'Nums after NMS {len(detector.features)}')
                detector.harris_detector(current_img.copy())
                print(f'Nums after detector Harris {len(detector.features)}')
                detector.calc_orientation_key_points(current_img.copy())
                current_img = cv2.GaussianBlur(current_img, (3, 3), 1.5)
                descriptor.brief(current_img.copy(), detector.features, detector.thetas)
                for i in range(len(detector.features)):
                    self.keypoints_data.append(
                        Keypoint(detector.features[i], detector.thetas[i], np.array(descriptor.descriptors[i]), detector.response_harris[i],
                                 np.power(self.scale_factor, t - 1)))
                current_img = cv2.resize(current_img, (int(width / self.scale_factor), int(height / self.scale_factor)))
                height, width = current_img.shape[0], current_img.shape[1]
                detector.clear_attributes()
                descriptor.clear_attributes()
        return self.keypoints_data

    def draw_keypoints(self, img, keypoints_data=None):
        if keypoints_data is None:
            keypoints_data = self.keypoints_data
        fig, ax = plt.subplots(figsize=(17, 10))
        ax.imshow(np.uint8(img))
        for keypoint in keypoints_data:
            coords_begin = keypoint.coords * keypoint.scale_factor
            r = keypoint.scale_factor * self.radius
            x2 = r * np.cos(keypoint.angle) + coords_begin[1]
            y2 = r * np.sin(keypoint.angle) + coords_begin[0]
            y1 = coords_begin[0]
            x1 = coords_begin[1]
            ax.plot([x1, x2], [y1, y2], color='r', linewidth=0.5)
            c = plt.Circle((x1, y1), r, fill=False, linewidth=0.8,color='g')
            ax.add_patch(c)
        ax.set_title(f'Tau = {self.tau} Harris_k = {self.harris_k} Harris_N = {self.harris_tau} '
                        f'Radius = {self.radius}')
        plt.show()

    def save_descriptors(self, name='descriptors_img'):
        descriptors = []
        for kp in self.keypoints_data:
            descriptors.append(kp.descriptor)
        descriptors = np.array(descriptors)
        with open(name+'.pickle','wb') as f:
            pickle.dump(descriptors,f)

    def print_kp(self):
        for keypoint in self.keypoints_data:
            print(5 * '-')
            print(f'Coords: {keypoint.coords}')
            print(f'Angle: {keypoint.angle}')
            print(f'Descriptor: {np.array(keypoint.descriptor).sum()}')
            print(f'Response: {keypoint.response}')
            print(f'Scale_factor: {keypoint.scale_factor}')





