#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 0

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'image', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_RGB.tif', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gt', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'noBoundary' in el]
            names = [el.replace('_label_noBoundary.tif', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))
            
        ## parse deep directory
        self.deeps = {}
        deepnames = []
        deeppth = osp.join(rootpth, 'deep', mode)
        folders = os.listdir(deeppth)
        for fd in folders:
            fdpth = osp.join(deeppth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'normalized' in el]
            names = [el.replace('_normalized_lastools.jpg', '') for el in lbnames]
            deeppths = [osp.join(fdpth, el) for el in lbnames]
            deepnames.extend(names)
            self.deeps.update(dict(zip(names, deeppths)))
            
        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(imgnames) == set(deepnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())
        assert set(self.imnames) == set(self.deeps.keys())
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.tensor = transforms.ToTensor()
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            VeticalFlip(),
            RandomScale((0.5,0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        deepth = self.deeps[fn]
#        print(deepth)
        img = Image.open(impth)
        label = Image.open(lbpth)
        deep = Image.open(deepth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label,dp=deep)
            im_lb = self.trans_train(im_lb)
            img,deep, label = im_lb['im'], im_lb['dp'],im_lb['lb']
        img = self.to_tensor(img)
        deep = self.tensor(deep)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
#        label = self.convert_labels(label)
        return img, deep,label,fn


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

