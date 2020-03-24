import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F
from load_Brain_data import BrainS18Dataset

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib

from probabilistic_unet import ProbabilisticUnet
from utils import label2multichannel, cal_variance, save_8_pred_img, mask2rgb
from save_load_net import load_model
from evaluate import evaluate
import param


def func_1():
    """打印部分mri图像及其标签
    """
    # 参数
    class_num = param.class_num # 选择分割类别数

    # 选择数据集
    dataset = BrainS18Dataset(root_dir='data/BrainS18', 
                            folders=['1_img'], 
                            class_num=class_num, 
                            file_names=['_reg_T1.png', '_segm.png'])
    # dataset = BrainS18Dataset(root_dir='data/BrainS18', folders=['1_Brats17_CBICA_AAB_1_img'],
    #                           class_num=class_num,
    #                           file_names=['_reg_T1.png', '_segm.png'])

    # 数据划分并设置sampler（（固定训练集和测试集））
    dataset_size = len(dataset)  # 数据集大小
    
    imgs = np.zeros((8, 240, 240))
    labels = np.zeros((8, 240, 240))
    count = 0
    for i in range(48):
        if i in (0,6,12,18,24,30,36,42):
            image, label, series_uid = dataset.__getitem__(i)
            image = image.numpy().reshape(240, 240)
            label -= 1
            label = label.numpy().reshape(240, 240)
            imgs[count] = image
            labels[count] = label
            count += 1


    fig, ax = plt.subplots(2, 8, sharey=True, figsize=(20, 5))
    cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors,tab10,Paired

    for i in range(8):
        ax[0][i].imshow(imgs[i], aspect="auto", cmap="gray")
        ax[1][i].imshow(labels[i], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    # # ax[0][0].imshow(orig, aspect="auto", cmap="gray")

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig('picture/a_func_1.png', format='png', transparent=True, dpi=300, pad_inches = 0)


if __name__ == "__main__": 
    func_1()
