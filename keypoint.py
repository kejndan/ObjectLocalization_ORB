import numpy as np


class Keypoint:
    def __init__(self, coords, angle, descriptor, response, scale_factor):
        self.coords = coords
        self.angle = angle
        self.descriptor = descriptor
        self.response = response
        self.scale_factor = scale_factor




