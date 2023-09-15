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


def dynamic_normalisation(trainA, trainB = [], grayscale = False):
    """
    Here we choose K to be an estimate of the mean per channel
    We shift every pixel in a given channel by the associated K. 
    This ensures no cancellation will occur due to the subtraction 
    during the 'var' assignment statement, otherwise a potential 
    cancellation could ruin the natural precision. This does not 
    change the variance, and thus standard deviation, since Var(X) = Var(X-K)
    """
    print('Calculating normalisation stats')
    paths = trainA+trainB
    Sums = []
    SSq = []
    K = (0.412 if grayscale else np.array([0.338, 0.439, 0.463]).reshape(1,1,3))
    totalPix = 0 # counter for total number of pixels in dataset (per channel)
    start_time = time.time()
    
    for i,path in enumerate(paths):
        with Image.open(path) as image:
            if grayscale:
                image = image.convert('L') # convert to grayscale
            else:
                image = image.convert('RGB')
            image = np.array(image)/255
            totalPix += image.shape[0]*image.shape[1] # increment by the image height*width, i.e. number of pixels per channel
            if i==(len(trainA)-1):
                totalPixA = totalPix
            shift = image-K
            Sums.append(np.sum(shift,axis=(0,1))) # sum along the height and width (indices 0 and 1) but not along the channels
            SSq.append(np.sum(shift**2,axis=(0,1))) # same as line above
            
    TotalSum = np.sum(Sums,axis=0) # sum the appended items along the axis of appending
    TotalSSq = np.sum(SSq,axis=0) # same as line above
    TotalSum_A = np.sum(Sums[:len(trainA)],axis=0) # sum the appended items along the axis of appending till element len(trainA)-1
    TotalSSq_A = np.sum(SSq[:len(trainA)],axis=0) # same as line above
    TotalSum_B = np.sum(Sums[len(trainA):],axis=0) # sum the appended items along the axis of appending after element len(trainA)-1
    TotalSSq_B = np.sum(SSq[len(trainA):],axis=0) # same as line above
    
    mean = (K + TotalSum/totalPix).flatten().tolist()
    var = (TotalSSq-TotalSum**2/totalPix)/(totalPix-1) 
    # use (totalPix) instead of (totalPix-1) if want to compute the exact variance of the given data
    # use (totalPix-1) if data are samples of a larger population
    std = np.sqrt(var).flatten().tolist()
    
    mean_A = (K + TotalSum_A/totalPixA).flatten().tolist()
    var_A = (TotalSSq_A-TotalSum_A**2/totalPixA)/(totalPixA-1) 
    # use (totalPix) instead of (totalPix-1) if want to compute the exact variance of the given data
    # use (totalPix-1) if data are samples of a larger population
    std_A = np.sqrt(var_A).flatten().tolist()
    
    totalPixB = totalPix-totalPixA
    mean_B = (K + TotalSum_B/totalPixB).flatten().tolist()
    var_B = (TotalSSq_B-TotalSum_B**2/totalPixB)/(totalPixB-1) 
    # use (totalPix) instead of (totalPix-1) if want to compute the exact variance of the given data
    # use (totalPix-1) if data are samples of a larger population
    std_B = np.sqrt(var_B).flatten().tolist()
    
    end_time = time.time()-start_time
    print('Total time taken for dyno calculation: %.6f seconds' % end_time)
    print('Time taken per image for dyno calculation: %.6f seconds' % (end_time/len(paths)))
    del Sums, SSq
    if grayscale:
        mean.append(mean[0]);mean_A.append(mean_A[0]);mean_B.append(mean_B[0])
        mean.append(mean[0]);mean_A.append(mean_A[0]);mean_B.append(mean_B[0])
        std.append(std[0]);std_A.append(std_A[0]);std_B.append(std_B[0])
        std.append(std[0]);std_A.append(std_A[0]);std_B.append(std_B[0])
    statsTotal = {'mean':mean,'std':std}
    statsA = {'mean':mean_A,'std':std_A}
    statsB = {'mean':mean_B,'std':std_B}
    return statsTotal,statsA,statsB


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
