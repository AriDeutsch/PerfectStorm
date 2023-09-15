import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, dynamic_normalisation
from util.util import mkdirs
from PIL import Image
import random
import json
import numpy as np


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        
        stats_file = os.path.join(opt.checkpoints_dir, opt.name, 'stats.json') # file to save stats
        
        # Only calculate if train phase, if dyno requested, and no stats file exists already
        if opt.isTrain and opt.dyno and not os.path.isfile(stats_file):        
            self.opt.dyno_stats,self.opt.statsA,self.opt.statsB = dynamic_normalisation(self.A_paths,self.B_paths, grayscale = self.opt.Grey3Ch)
            stats_file = os.path.join(opt.checkpoints_dir, opt.name, 'stats.json')
            with open(stats_file, 'w') as f:
                json.dump((self.opt.dyno_stats,self.opt.statsA,self.opt.statsB), f)   # save the stats 
        # If the stats file exists, load the stats, regardless of phase
        elif os.path.isfile(stats_file):
            with open(stats_file, 'r') as f:
                self.opt.dyno_stats,self.opt.statsA,self.opt.statsB = json.load(f)  # load the stats
        else:
            self.opt.dyno_stats=self.opt.statsA=self.opt.statsB = None  # If no stats file exists, and it's not train phase or no dyno requested, use standard 0.5-level stats
            
        
 
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        if self.opt.Grey3Ch:
            assert (input_nc==3 and output_nc==3), "Number of in and out channels should be 3 if you want 3 channel greyscale."
            
        # Calculate a rough measure in the saturation difference between 
        # domain A and B, and pass the ratio to get_transform, 
        # dependent on which domain has higher saturation
        RelativeSaturationA = np.std(self.opt.statsA['mean'])
        RelativeSaturationB = np.std(self.opt.statsB['mean'])
        SatScaleA = (RelativeSaturationB/RelativeSaturationA)
        SatScaleB = 1/SatScaleA
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), Grey3Ch = self.opt.Grey3Ch, stats = self.opt.dyno_stats, col_jit = self.opt.col_jit, SatScale=SatScaleA if SatScaleA>1 else None)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), Grey3Ch = self.opt.Grey3Ch, stats = self.opt.dyno_stats, col_jit = self.opt.col_jit, SatScale=None if SatScaleA>1 else SatScaleB)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
