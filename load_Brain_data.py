# coding: utf-8
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import os

# 颜色rgb
# 黑，红，绿，蓝，黄，淡紫，亮蓝，紫，白
color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),
    (255,0,255),(0,255,255),(128,0,128),(255,255,255)] 
color_ox = ["#000000","#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF",
            "#800080", "#FFFFFF"]
# 皮质灰质、基底神经节、白质、白质病变、脑脊液中、心室、小脑、脑干
color_name = ["Cortical gray matter", "Basal ganglia", 
                "White matter", "White matter lesions", 
                "Cerebrospinal fluid in the extracerebral space", 
                "Ventricles", "Cerebellum", "Brain stem"]


class BrainS18Dataset(Dataset):
    def __init__(self, root_dir='data/BrainS18', folders=['1', '5', '7', '4', '148', '070', '14'], class_num=0):
        print('Preparing BrainS18Dataset {} ... '.format(folders), end='')

        self.file_names = ['_FLAIR.png', '_reg_IR.png', '_reg_T1.png', '_segm.png']
        self.mean_std = {}  # mean and std of a volume
        self.img_paths = [] # e.g. './datasets/BrainS18/14/2'
        self._prepare_dataset(root_dir, folders)

        self.class_num = class_num

        print('Done')

    def _prepare_dataset(self, root_dir, folders):
        # compute mean and std and prepare self.img_paths
        for folder in folders:
            paths = [os.path.join(root_dir, folder, str(i)) for i in range(48)]
            self.img_paths += paths
            self.mean_std[folder] = {}
            for file_name in ['_FLAIR.png', '_reg_IR.png', '_reg_T1.png']:
                volume = np.array([mpimg.imread(path + file_name) for path in paths])
                self.mean_std[folder][file_name] = [volume.mean(), volume.std()]
    
    def __len__(self):
        return len(self.img_paths)

    def _getlabel(self, img):
        """"将label从单通道0-8变为九通道0-1"""
        class_num = 9 # 九类（包括背景类）
        img = img.reshape((240,240))
        # img *= 255
        label = np.zeros((class_num, 240, 240))
        for i in range(class_num):
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if int(img[x,y]) == i:
                        label[i+1,x,y] = 1
        return label

    def _getrgb(self, img):
        """将单通道label变为rgb"""

        label = np.zeros((240, 240, 3))
        for c in range(9):
            for x in range(240):
                for y in range(240):
                    if int(img[0,x,y]) == c+1:
                        label[x,y,0] = color[c][0]
                        label[x,y,1] = color[c][1]
                        label[x,y,2] = color[c][2]
        label /= 255
        return label


    def __getitem__(self, index):
        folder = self.img_paths[index].split('/')[-2]
        # read imgs
        imgs = [mpimg.imread(self.img_paths[index] + fn).reshape((1, 240, 240)) for fn in self.file_names]

        # normalization
        for i in range(3):
            # e.g. mean_std = {'_FLAIR.png': [0.14819147, 0.22584382], 
            #                  '_reg_IR.png': [0.740661, 0.18219014], 
            #                  '_reg_T1.png': [0.1633398, 0.25954548]}
            mean = self.mean_std[folder][self.file_names[i]][0]
            std = self.mean_std[folder][self.file_names[i]][1]
            imgs[i] = (imgs[i] - mean) / std

        # 标签
        label = imgs[3].reshape((1,240,240))
        label *= 255
        label += 1 # 加入背景类
        # label = self._getlabel(imgs[3])
        
        # 选输入图片类型 0:_FLAIR/1:_reg_IR/2:_reg_T1
        image = torch.from_numpy(imgs[2])
        label = torch.from_numpy(label)
        
        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        series_uid = 0

        return image, label, series_uid

    def show_imgs(self, index):
        image, label, series_uid = self.__getitem__(index)
        
        # torch->numpy
        image = image.numpy()[0]
        label = label.numpy()

        #label单通道变rgb 
        label = self._getrgb(label)

        # plt显示图片
        fig, axs = plt.subplots(1,2, sharey=True, figsize=(8.5,4))
        axs[0].set_title("origin")
        axs[1].set_title("label")

        ax00 = axs[0].imshow(image, aspect="auto", cmap="gray")
        ax01 = axs[1].imshow(label, aspect="auto")
        
        # 图片标题&保持图像
        fig.suptitle(self.img_paths[index])
        plt.savefig('picture/test_{}.jpg'.format(index))

if __name__ == "__main__": 
    dataset = BrainS18Dataset()
    print(len(dataset))
    dataset.show_imgs(20)
