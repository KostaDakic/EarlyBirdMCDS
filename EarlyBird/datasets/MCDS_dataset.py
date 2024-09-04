import os
import numpy as np
import cv2
from torchvision.datasets import VisionDataset

class MCDS(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'MCDS'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [300, 300]  # Adjust as needed
        self.num_cam, self.num_frame = 6, 400  # Adjust as needed
        self.worldcoord_from_worldgrid_mat = np.array([[0.025, 0, 0], [0, 0.025, 0], [0, 0, 1]])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            for fname in sorted(os.listdir(os.path.join(self.root, 'image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        MAP_WIDTH = 30
        MAP_EXPAND = 10
        grid_x = pos % (MAP_WIDTH * MAP_EXPAND)
        grid_y = pos // (MAP_WIDTH * MAP_EXPAND)
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i, frame):
        intrinsic_path = os.path.join(self.root, 'calibrations', 'intrinsic', f'intr_Drone{camera_i+1}_{frame:04d}.xml')
        extrinsic_path = os.path.join(self.root, 'calibrations', 'extrinsic', f'extr_Drone{camera_i+1}_{frame:04d}.xml')

        fp_intrinsic = cv2.FileStorage(intrinsic_path, flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_intrinsic.getNode('camera_matrix').mat()
        fp_intrinsic.release()

        fp_extrinsic = cv2.FileStorage(extrinsic_path, flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_extrinsic.getNode('rvec').mat().squeeze(), fp_extrinsic.getNode('tvec').mat().squeeze()
        fp_extrinsic.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix