"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
import time

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def dynamic_normalisation(trainA, trainB, grayscale = False):
    """
    Here we choose K to be an estimate of the mean per channel
    We shift every pixel in a given channel by the associated K. 
    This ensures no cancellation will occur due to the subtraction 
    during the 'var' assignment statement, otherwise a potential 
    cancellation could ruin the natural precision. This does not 
    change the variance, and thus standard deviation, since Var(X) = Var(X-K)
    """
    paths = trainA+trainB
    Sums = []
    SSq = []
    K = (0.412 if grayscale else np.array([0.338, 0.439, 0.463]).reshape(1,1,3))
    totalPix = 0 # counter for total number of pixels in dataset (per channel)
    start_time = time.time()
    
    for path in paths:
        with Image.open(path) as image:
            if grayscale:
                image = image.convert('L') # convert to grayscale
            else:
                image = image.convert('RGB')
            image = np.array(image)/255
            totalPix += image.shape[0]*image.shape[1] # increment by the image height*width, i.e. number of pixels per channel
            shift = image-K
            Sums.append(np.sum(shift,axis=(0,1))) # sum along the height and width (indices 0 and 1) but not along the channels
            SSq.append(np.sum(shift**2,axis=(0,1))) # same as line above
            
    TotalSum = np.sum(Sums,axis=0) # sum the appended items along the axis of appending
    TotalSSq = np.sum(SSq,axis=0) # same as line above
    
    mean = K + TotalSum/totalPix
    var = (TotalSSq-TotalSum**2/totalPix)/(totalPix-1) 
    # use (totalPix) instead of (totalPix-1) if want to compute the exact variance of the given data
    # use (totalPix-1) if data are samples of a larger population
    std = np.sqrt(var)
    
    end_time = time.time()-start_time
    print('Total time for dyno calculation taken: %.6f seconds' % end_time)
    print('Time taken per image for dyno calculation: %.6f seconds' % (end_time/len(paths)))
    del Sums, SSq
    return {'mean':mean.flatten().tolist(),'std':std.flatten().tolist()}


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
