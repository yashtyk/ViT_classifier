import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torchvision.transforms import ToTensor


# creating true sky model, CASA UV representation dataset

class CASA_dataset(Dataset):

    "return trye sky model, CASA UV representation dataset"

    #def __init__(self, images_path = '/srv/beegfs/scratch/users/s/shtyk1/vit_casa/sky_data/sky_true_wo_processing.npy', vectors_path = '/srv/beegfs/scratch/users/s/shtyk1/vit_casa/sky_data/sky_info.npy' ,  transform_images=None, transform_vectors=None):
    def __init__(self, images_path='/Users/yanashtyk/Documents/sky/sky_true_wo_processing.npy',vectors_path='/Users/yanashtyk/Documents/sky/sky_info.npy',transform_images=None, transform_vectors=None):
        self.transform_images  = ToTensor()
        self.transform_vectors = ToTensor()

        self.true_sky_images = np.load(images_path)
        self.num_sources = np.load(vectors_path)

    def __len__(self):
        return self.true_sky_images.shape[0]

    def __getitem__(self, idx):
        img = self.true_sky_images[idx]
        num_sources = self.num_sources[idx]

        img = self.transform_images(img)

        #num_sources = self.transform_vectors(np.array([num_sources]))

        return img, num_sources











