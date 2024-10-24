import glob
import os

from torchvision import transforms
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from utils.augment import Augmentation, mirrorPadding2D
# import augment as augment
from PIL import Image
# import random

class WSIDataset(Dataset):
    def __init__(self, root_dir, mode, args=None):

        self.root_dir = root_dir
        self.mode = mode
        # modify    
        # self.args = args
        self.probabilityOfExchangeTemporal = args.probabilityOfExchangeTemporal
        self.probabilityOfSelfPair = args.probabilityOfSelfPair
        self.all_png_dir_1    = []
        self.all_png_dir_2    = []
        self.all_label_change = []
        for k,v in self.root_dir.items():
            self.all_png_dir_1    += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "T1" + os.sep + '*'))#[0: 128]
            self.all_png_dir_2    += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "T2" + os.sep + '*'))#[0: 128]
            self.all_label_change += sorted(glob.glob(self.root_dir[k] + os.sep + self.mode + os.sep + "label" + os.sep + '*'))#[0: 128]
        self.all_png_dir_1_name =  [os.path.splitext(os.path.split(i)[1])[0] for i in self.all_label_change]
        print("T1 patch numbers: ", len(self.all_png_dir_1))
        print("T2 patch numbers: ", len(self.all_png_dir_2))
        print("label patch numbers: ", len(self.all_label_change))
        self.source_size = args.dataset[self.mode]["sourceImageSize"]
        self.randomImgSize = args.dataset[self.mode]["randomImgSize"]
        self.colorJitterParams = args.colorJitterParams
        self.colorJitterProbs = args.colorJitterProbs
        self.imgNormalizeMean = args.imgNormalizeMean
        self.imgNormalizeStd = args.imgNormalizeStd
        self.listOfCategoryNames = args.listOfCategoryNames
        self.isMirrorPadding = args.isMirrorPadding
        
    def __getitem__(self, index):
        dir        = self.all_png_dir_1_name[index]
        img1       = self.all_png_dir_1[index]
        img2       = self.all_png_dir_2[index]
        labelc     = self.all_label_change[index]
        if self.mode == "train":        

            img1       = Image.open(img1).resize(self.randomImgSize)
            img2       = Image.open(img2).resize(self.randomImgSize)
            labelc     = Image.open(labelc).resize(self.randomImgSize, resample=Image.Resampling.NEAREST)
            if self.isMirrorPadding:
                img1       = np.array(img1)
                img2       = np.array(img2)
                labelc     = np.expand_dims(np.array(labelc), axis=2)
                img1       = mirrorPadding2D(img1)
                img2       = mirrorPadding2D(img2)
                labelc     = mirrorPadding2D(labelc)
                img1       = Image.fromarray(img1)
                img2       = Image.fromarray(img2)
                labelc     = Image.fromarray(np.squeeze(labelc))
            #First, the generation of invariant image pairs is determined, followed by a random interchange of the temporal order of the two images.
            
            # With a probability of p, the images are stochastically assembled using their own components to form invariant image pairs. 

            # Additionally, with a probability of p, the temporal order of the two images is randomly exchanged.
            if random.random() < self.probabilityOfSelfPair:
                tempImg = random.choice([img1, img2])
                img1 = tempImg
                img2 = img1.copy(img1)
                labelc = Image.fromarray(np.zeros_like(np.array(labelc)))
            else:
                if random.random() < self.probabilityOfExchangeTemporal:
                    temp = img1
                    img1 = img2
                    img2 = temp


            aug = Augmentation()
            # geometric distortion
            img2_combine, bias_y, bias_x = aug.randomSpaceAugmentWithGlobalMirrorCrop([img1,img2,labelc], source_size=self.randomImgSize, unoverlap=None)
            # photometric distortion
            img1,img2,labelc = img2_combine
            imgPhotometricDistortion1 = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(*self.colorJitterParams)], p=self.colorJitterProbs),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imgNormalizeMean, std=self.imgNormalizeStd),

            ])
            imgPhotometricDistortion2 = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(*self.colorJitterParams)], p=self.colorJitterProbs),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imgNormalizeMean, std=self.imgNormalizeStd),
            ])
            img1 = imgPhotometricDistortion1(img1)
            img2 = imgPhotometricDistortion2(img2)
            # print(np.unique(labelc))
            labelc     = torch.FloatTensor(np.array(labelc))/255
            if len(np.unique(labelc)) != self.listOfCategoryNames:
                labelc[labelc != 0] = 1

        elif self.mode in "validation" or self.mode in "test":
            img1       = Image.open(img1).resize(self.randomImgSize)
            img2       = Image.open(img2).resize(self.randomImgSize)
            imgTransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imgNormalizeMean, std=self.imgNormalizeStd),
            ])
            labelc     = np.expand_dims(np.array(Image.open(labelc).resize(self.randomImgSize, resample=Image.Resampling.NEAREST)), axis=2)
            labelc     = torch.FloatTensor(np.squeeze(labelc))/255
            if len(np.unique(labelc)) != self.listOfCategoryNames:
                labelc[labelc != 0] = 1
            img1       = imgTransforms(img1)
            img2       = imgTransforms(img2)
        label1 = torch.FloatTensor([0])
        label2 = torch.FloatTensor([0])
        return img1, img2, label1, label2, labelc, dir

    def __len__(self):
        return len(self.all_png_dir_1)

if __name__ == "__main__":

    pass