import numpy as np
import odl
import os
from skimage.measure import block_reduce

FILE_DIR = "/home/lpd/work/Walnuts/data_downsampled_120ang/"

class DataLoader(object):
    def __init__(self):
        self.data_dir =FILE_DIR
        self.orbit_id = 2
        self.n_data = 42
        self.n_proj = 1200 # the first and last projections coincide
        self.proj_rows = 972
        self.proj_cols = 768
        self.det_pixel = 0.1496
        self.n_slices = 501
        self.im_size = 501
        self.roi = 501 / 10
        self.src_rad = 66.001404
        self.det_rad = 199.006195 - self.src_rad
        self.ang_ds = 10
        self.ds = 2
        # angle-data correspondance is a bit of 
        # in the script that produces ground truth images
        # so, we mimic it here 
        self.proj_idx = range(self.ang_ds-1, self.n_proj, self.ang_ds) 
            
    def load_projections(self, walnut_id):
        data_path_full = os.path.join(self.data_dir, 
                                      'Walnut{}_data.npy'.format(walnut_id))
        projs = np.load(data_path_full)

        return projs
    
    def load_images(self, walnut_id):
        image_full_path = os.path.join(self.data_dir, 
                                       'Walnut{}_image.npy'.format(walnut_id))
        image = np.load(image_full_path)
        return image
    
    def odl_geometry(self, walnut_id):
        # Each Walnut has a slightly different geometry.
        # However, the only difference that can't be neglected is the offset along z-axis.
        geom_path_full = os.path.join(self.data_dir, 
                                      'Walnut{}_scan_geom_corrected.geom'.format(walnut_id))
        vecs = np.loadtxt(geom_path_full)
        vec = vecs[0]
        src0 = vec[0:3] # source position
        det0 = vec[3:6] # detector position
        da1 = vec[6:9] # detector axis
        da2 = vec[9:12] # detector axis
        offset = src0[2] + (det0[2] - src0[2]) * self.src_rad / (self.src_rad + self.det_rad)
        n_proj = len(self.proj_idx)
        a0 = np.pi / n_proj
        
        angle_partition = odl.uniform_partition(a0, 2 * np.pi + a0, n_proj)
        detector_partition = odl.uniform_partition([-0.5 * self.proj_cols * self.det_pixel, 
                                                    -0.5 * self.proj_rows * self.det_pixel], 
                                                   [0.5 * self.proj_cols * self.det_pixel, 
                                                    0.5 * self.proj_rows * self.det_pixel], 
                                                   (int(self.proj_cols / self.ds),
                                                    int(self.proj_rows / self.ds)))
        
        geometry = odl.tomo.ConeBeamGeometry(angle_partition, detector_partition, 
                                             src_radius=self.src_rad, det_radius=self.det_rad, 
                                             axis=(1, 0, 0), src_to_det_init=(det0 - src0)[::-1], 
                                             det_axes_init=[da1[::-1], da2[::-1]], 
                                             offset_along_axis=offset
                                            )
        return geometry
    
    def odl_space(self):
        space = odl.uniform_discr([-0.5 * self.roi] * 3, [0.5 * self.roi] * 3, 
                                  [int(self.im_size / self.ds) + 1] * 3, dtype='float32')
        return space
    
    def generate_data(self, mode="train"):
        # mode = {"train", "test", "val"}
        
        if mode == "val":
            ind = 1
        elif mode == "test":
            ind = 2
        else:
            ind = np.random.randint(1, self.n_data-1) + 2
            
        data = self.load_projections(ind)
        image = self.load_images(ind)
        geometry = self.odl_geometry(ind)

        return image, data, geometry