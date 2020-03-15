# coding: utf-8
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import os

# # 颜色rgb
# # 黑，红，绿，蓝，黄，淡紫，亮蓝，紫，白
# color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),
#     (255,0,255),(0,255,255),(128,0,128),(255,255,255)] 
# color_ox = ["#000000","#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF",
#             "#800080", "#FFFFFF"]
# # 0背景、1皮质灰质、2基底神经节、3白质、4白质病变、5脑脊液、6脑室、7小脑、8脑干、9梗塞、10其他
# color_name = ["Background", "Cortical gray matter", "Basal ganglia", 
#                 "White matter", "White matter lesions", 
#                 "Cerebrospinal fluid in the extracerebral space", 
#                 "Ventricles", "Cerebellum", "Brain stem", "Infarction"、"Other"]
# 数据集说明：https://mrbrains18.isi.uu.nl/data/


class BrainS18Dataset(Dataset):
    def __init__(self, root_dir='data/BrainS18', folders=['1_img', '5_img'], class_num=9, file_names=['_reg_T1.png', '_segm.png']):
        print('Preparing BrainS18Dataset {} ... '.format(folders), end='')

        self.file_names = file_names
        self.mean_std = {}  # mean and std of a volume
        self.img_paths = [] # e.g. './datasets/BrainS18/14/2'
        self._prepare_dataset(root_dir, folders) # 准备数据集

        self.class_num = class_num

        print('Done')

    def _prepare_dataset(self, root_dir, folders):
        """计算数据均值和方差，并计算图片数据路径"""
        # compute mean and std and prepare self.img_paths
        for folder in folders:
            paths = [os.path.join(root_dir, folder, str(i)) for i in range(48)]
            self.img_paths += paths
            self.mean_std[folder] = {}
            for file_name in self.file_names[:-1]:
                volume = np.array([mpimg.imread(path + file_name) for path in paths])
                self.mean_std[folder][file_name] = [volume.mean(), volume.std()]
    
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index):
        folder = self.img_paths[index].split('/')[-2]
        
        series_uid = self.img_paths[index].split('/')[-2:]

        # read imgs
        imgs = [mpimg.imread(self.img_paths[index] + fn).reshape((1, 240, 240)) for fn in self.file_names]

        # normalization
        for i in range(len(self.file_names)-1):
            # e.g. mean_std = {'_FLAIR.png': [0.14819147, 0.22584382], 
            #                  '_reg_IR.png': [0.740661, 0.18219014], 
            #                  '_reg_T1.png': [0.1633398, 0.25954548]}
            mean = self.mean_std[folder][self.file_names[i]][0]
            std = self.mean_std[folder][self.file_names[i]][1]
            imgs[i] = (imgs[i] - mean) / std

        # 标签
        label = imgs[-1].reshape((1,240,240))
        label *= 255 # 元素值变为0-9
        # 将标签为9的变为1
        label[label==9] = 1
        
        label += 1 # 加入背景类，元素值变为1-10

        
        # 选输入图片类型 ～～0:_FLAIR/1:_reg_IR/2:_reg_T1～～
        image = torch.from_numpy(imgs[0])
        label = torch.from_numpy(label)
        
        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, series_uid

if __name__ == "__main__": 
    dataset = BrainS18Dataset()
    # for i in range(len(dataset)):
    #     image, label, series_uid = dataset.__getitem__(i)
    print(len(dataset))
    count = 0
    class_ = 11
    for i in range(len(dataset)):
        _, label, _ = dataset.__getitem__(i)
        isFound = False
        for x in range(240):
            for y in range(240):
                if int(label[0, x, y]) == class_:
                    count += 1
                    isFound = True
                    break
            if isFound:
                break
    print("Class {}: {} slice".format(class_-1, count))
    
