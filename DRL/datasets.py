import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from skimage.util import view_as_blocks

class FractalDataset(Dataset,config):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
	self.transform_files=sorted(glob.glob(os.path.join(root, mode) + "/*.tft"))
	self.rshape=config.rshape
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
	    self.transform_files.extend(sorted(glob.glob(os.path.join(root, mode) + "/*.tft")))

    def getTransform(self,file)
	data=open(file).readlines()
	tfs=[]
	for d in data:
		tfs=d.split(",")
	return tfs
	
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
	# see astronaut as a matrix of blocks (of shape block_shape)
	    view = view_as_blocks(img, block_shape)

        tfs=self.getTransforms(self.transform_files[index])

        return {"X": view, "TF": tfs}

    def __len__(self):
        return len(self.files)
