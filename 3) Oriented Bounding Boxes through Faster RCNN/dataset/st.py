import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import json

class SceneTextDataset(Dataset):
    def __init__(self, split, root_dir):
        self.split = split
        self.root_dir = root_dir
        self.im_dir = os.path.join(root_dir, 'img')
        self.ann_dir = os.path.join(root_dir, 'annots')
        classes = [
            'text'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images = glob.glob(os.path.join(self.im_dir, '*.jpg')) 
        self.annotations = [os.path.join(self.ann_dir, os.path.basename(im) + '.json') for im in self.images]
        
        if(split == 'train'):
            self.images = self.images[:int(0.8*len(self.images))]
            self.annotations = self.annotations[:int(0.8*len(self.annotations))]
        else:
            self.images = self.images[int(0.8*len(self.images)):]
            self.annotations = self.annotations[int(0.8*len(self.annotations)):]
    
    def __len__(self):
        return len(self.images)
    
    def convert_xcycwh_to_xyxy(self, box):
        x, y, w, h = box
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return [x1, y1, x2, y2]
    
    def __getitem__(self, index):
        im_path = self.images[index]
        im = Image.open(im_path)
        
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        ann_path = self.annotations[index]
        with open(ann_path, 'r') as f:
            im_info = json.load(f)
            xc = [detec['obb']['xc'] for detec in im_info['objects']]
            yc = [detec['obb']['yc'] for detec in im_info['objects']]
            w = [detec['obb']['w'] for detec in im_info['objects']]
            h = [detec['obb']['h'] for detec in im_info['objects']]
            
            # read the angles here as well from the json file...
            
            boxes = [self.convert_xcycwh_to_xyxy([xc[i], yc[i], w[i], h[i]]) for i in range(len(xc))]
            
        targets['bboxes'] = torch.as_tensor(boxes).float()
        targets['labels'] = torch.as_tensor(torch.ones(len(im_info['objects'])).long())
        
        return im_tensor, targets, im_path