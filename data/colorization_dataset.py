from data.aligned_dataset import AlignedDataset
import torch
from skimage import color  # used for lab2rgb
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
import torchvision.transforms as transforms
from data.image_folder import make_dataset
import os.path


class ColorizationDataset(AlignedDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.
    This dataset is required by pix2pix-based colorization model ('--model colorization')
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        dir_A = dir_B = '_img'  # RGB and gray data will be same source

        ### input A (label maps)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A.size)
        # if self.opt.label_nc == 0:
        transform_A = get_transform(self.opt, params,convert=False)
        A = transform_A(A)
        A = np.array(A)
        lab = color.rgb2lab(A).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        A_tensor = lab_t[[0], ...] / 50.0 - 1.0  ##将L通道(index=0)归一化到-1和1之间

        B_tensor = 0
        inst_tensor = feat_tensor = 0  # for pass train, need tensor
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_tensor = lab_t[[1, 2], ...] / 110.0  ##将A，B通道(index=1,2)归一化到0和1之间

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def name(self):
        return 'ColorizationDataset'

