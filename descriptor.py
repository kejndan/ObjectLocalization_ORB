import numpy as np
import cv2
from hw_6.utils import get_point_circle, calc_grads_img, nms_2d,get_all_points_in_circle


class DescriptorKeypoints:
    def __init__(self, length_descriptors=256, sectors_angle=30,patch_size=31, points_create='normal'):



        self.length_descriptors = length_descriptors
        self.sectors_angle = sectors_angle
        self.descriptors =[]
        self.points = None
        self.orient_points = []
        self.angles = None
        self.patch_size = patch_size
        self.points_create = points_create # если файла, то нужно передать в название переменной путь до файла

        self.calc_points_orientation()

    def clear_attributes(self):
        self.descriptors = []

    def get_points_from_file(self):
        points = []
        with open(self.points_create, 'r') as f:
            for line in f.readlines():
                points.append(list(map(float,line.split())))
        self.points = np.array(points)
        return self.points

    def get_points_from_normal_distribution(self):
        np.random.seed(52)
        # print((1/25*self.patch_size**2)**(1/2))
        points = np.random.normal(0,(1/25*self.patch_size**2)**(1/2), size=(self.length_descriptors,4))
        return points



    def calc_points_orientation(self):
        if self.points_create == 'normal':
            all_pair_points = self.get_points_from_normal_distribution()
        else:
            all_pair_points = self.get_points_from_file()


        restructed_points = []
        for pair_points in all_pair_points:
            restructed_points.append([pair_points[0],pair_points[2]])
            restructed_points.append([pair_points[1],pair_points[3]])
        restructed_points = np.array(restructed_points).T
        angles = np.linspace(-np.pi,np.pi,self.sectors_angle+1)

        for alph in angles:
            rot_mat = np.array([[np.cos(alph), -np.sin(alph)],
                                [np.sin(alph), np.cos(alph)]])
            self.orient_points.append(np.round(rot_mat @ restructed_points, 0).astype(np.int))

        self.angles = angles
        return self.orient_points, angles


    def brief(self, img, keypoints, orientation):
        self.descriptors =[]
        for i, keypoint in enumerate(keypoints):
            descriptor = []
            idx_orient = int(np.argmin(abs(self.angles - orientation[i])))
            if idx_orient == 15:
                print(keypoint,idx_orient)

            shift_x = self.orient_points[idx_orient][:, 1::2] + keypoint[1]
            shift_y = self.orient_points[idx_orient][:, ::2] + keypoint[0]
            if keypoint[0] == 115 and keypoint[1] == 226:

                print(shift_y[0, 0]-keypoint[0], shift_x[0, 0]-keypoint[1])
                print(shift_y[1, 0]-keypoint[0], shift_x[1, 0]-keypoint[1])
                print(shift_y[0, 0] , shift_x[0, 0] )
                print(shift_y[1, 0] , shift_x[1, 0] )
            for j in range(shift_x.shape[1]):

                if img[shift_y[0,j],shift_x[0,j]] < img[shift_y[1,j],shift_x[1,j]]:
                    descriptor.append(1)
                else:
                    descriptor.append(0)
            #     if keypoint[0] == 115 and keypoint[1] == 226:
            #         print(shift_y[0,j],shift_x[0,j],shift_y[1,j],shift_x[1,j], img[shift_y[0,j],shift_x[0,j]],img[shift_y[1,j],shift_x[1,j]],descriptor[-1])
            # # if keypoint[0] == 115 and keypoint[1] == 226:
            #     print(descriptor)
            self.descriptors.append(descriptor)
        return self.descriptors





if __name__ == "__main__":
    c = DescriptorKeypoints()

    print(c.calc_points_orientation())