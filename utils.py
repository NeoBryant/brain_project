import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")


def label2multichannel(mask, class_num=10):
    """将单通道（元素值1-10）变为多通道（元素值0-1）
    输入为torch
    输出为torch
    mask.shape: batch_size,1,240,240
    """
    mask = mask.numpy()
    batch_size = mask.shape[0]
    h, w = mask.shape[2], mask.shape[3]
    # mask = mask.reshape((240, 240))
    label = np.zeros((batch_size,class_num, 240, 240))

    for b in range(batch_size):
        for i in range(class_num):
            for x in range(h):
                for y in range(w):
                    if int(mask[b,0,x,y]) == i+1:
                        label[b,i,x,y] = 1
    label = torch.from_numpy(label)
    label = label.type(torch.FloatTensor)

    return label

def mask2rgb(mask, class_num=10):
    """单通道变rgb图，以不同颜色显示大脑不同区域
    输入numpy(1,240,240)/(240,240)
    输出numpy(3,240,240)
    mask: numpy
    mask.shape (240,240)
    """
    mask = mask.reshape((240,240))
    # 颜色rgb （黑，红，绿，蓝，黄，淡紫，青，紫，黄绿，白）
    color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),
    (255,0,255),(0,255,255),(128,0,128),(128,128,0),(255,255,255)] 
    img = np.zeros((240, 240, 3))
    for c in range(class_num):
        for x in range(240):
            for y in range(240):
                if int(mask[x,y]) == c:
                    img[x,y,0] = color[c][0]
                    img[x,y,1] = color[c][1]
                    img[x,y,2] = color[c][2]
    img /= 255

    return img


def cal_variance(image_np, label_np, mask_pros, mask_pres, class_num, series_uid):
    """计算多个预测结果的方差
    image_np:(240,240)
    label_np:(240,240),元素值0-8
    mask_pres:(m,class_num,240,240),元素值为预测结果, np
    mask_pros:(m,240,240),m为预测次数,元素值0-8, list(np)
    class_num: 分割任务类别数
    series_uid: 每张图片的序列号，list，[病人，张数]
    """
    m = len(mask_pros) # 预测结果（次）数
    h, w = image_np.shape # 图片高宽
    
    ## 方法二（对预测正确的，对0/1求方差）
    # mean_result = np.zeros((h, w))      # 均值
    # variance_result = np.zeros((h, w))  # 方差
    # mask_pros_temp = [np.zeros((240, 240)) for i in range(m)] # 记录预测正确的像素点设为1
    # for i in range(m):
    #     mask_pros_temp[i] = (mask_pros[i]==label_np)
    # # 计算均值
    # for i in range(m):
    #     mean_result += mask_pros_temp[i]
    # mean_result /= m
    # # 计算方差
    # for i in range(m):
    #     variance_result += np.square(mean_result - mask_pros_temp[i])
    # variance_result /= m
    
    ## 方法三
    mask_pres = np.array(mask_pres)   # (m,class_num,240,240)
    variance_result = np.var(mask_pres, axis=0)
    variance_result = np.sum(variance_result, axis=0)
    
    # 熵
    eps = 0.0001
    sample = np.mean(mask_pres,axis=0)  # (class_num,240,240)
    sample += eps
    entropy = -np.sum(sample * np.log(sample), axis=0) # (240,240)
    
    # 保存原图、标签、和m张预测结果
    save_8_pred_img(image_np, label_np, variance_result,
                    mask_pros, entropy, class_num, series_uid)
    
    # 保存原图、标签、和方差
    # save_variance_img(image_np, mask2rgb(label_np), variance_result,series_uid)

    return 


def save_variance_img(orig, mask, var, series_uid):
    """保存原图、标签、预测结果的方差进行对比"""
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
    
    ax[0].set_title("Original")
    ax[1].set_title("Ground Truth")
    ax[2].set_title("variance")

    ax[0].imshow(orig, aspect="auto", cmap="gray")
    ax[1].imshow(mask, aspect="auto")
    ax[2].imshow(var, aspect="auto")

    # 去掉刻度
    for i in range(3):
        ax[i].axis('off')
    
    fig.suptitle('patient {} - slice {}'.format(series_uid[0][0], series_uid[1][0]))
    plt.savefig('picture/p{}_s{}_var.jpg'.format(series_uid[0][0], series_uid[1][0]))
    plt.close()


def save_8_pred_img(orig, mask, var, pred, entropy, class_num, series_uid):
    """保存原图、标签、8个预测结果进行对比
    orig:(240,240)
    mask:(240,240),元素值0-9
    pred:(k,240,240),k为预测次数,元素值0-9
    """
    m = len(pred)
    fig, ax = plt.subplots(5, 4, sharey=True, figsize=(24, 25))
    cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors,tab10,Paired

    # 设置每张子图的标题
    ax[0][0].set_title("Original")
    ax[0][1].set_title("Ground Truth")
    ax[0][2].set_title("Entropy")
    ax[0][3].set_title("variance")
    for i in range(4):
        ax[1][i].set_title("predict_{}".format(i))
        ax[2][i].set_title("predict_{}".format(i+4))
        ax[3][i].set_title("predict_{}".format(i+8))
        ax[4][i].set_title("predict_{}".format(i+12))
    
    # 设置每张子图显示内容
    ax00 = ax[0][0].imshow(orig, aspect="auto", cmap="gray")
    ax01 = ax[0][1].imshow(mask, cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax02 = ax[0][2].imshow(entropy, aspect="auto", cmap="jet", vmin=0, vmax=2)
    ax03 = ax[0][3].imshow(var, aspect="auto", cmap="jet")

    # for i in range(4):
    ax10 = ax[1][0].imshow(pred[0], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax11 = ax[1][1].imshow(pred[1], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax12 = ax[1][2].imshow(pred[2], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax13 = ax[1][3].imshow(pred[3], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax20 = ax[2][0].imshow(pred[4], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax21 = ax[2][1].imshow(pred[5], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax22 = ax[2][2].imshow(pred[6], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax23 = ax[2][3].imshow(pred[7], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax30 = ax[3][0].imshow(pred[8], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax31 = ax[3][1].imshow(pred[9], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax32 = ax[3][2].imshow(pred[10], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax33 = ax[3][3].imshow(pred[11], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax40 = ax[4][0].imshow(pred[12], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax41 = ax[4][1].imshow(pred[13], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax42 = ax[4][2].imshow(pred[14], cmap=cmap, aspect="auto", vmin=0, vmax=9)
    ax43 = ax[4][3].imshow(pred[15], cmap=cmap, aspect="auto", vmin=0, vmax=9)

    # color bar
    fig.colorbar(ax00, ax=ax[0][0])
    fig.colorbar(ax01, ax=ax[0][1])
    fig.colorbar(ax02, ax=ax[0][2])
    fig.colorbar(ax03, ax=ax[0][3])
    fig.colorbar(ax10, ax=ax[1][0])
    fig.colorbar(ax11, ax=ax[1][1])
    fig.colorbar(ax12, ax=ax[1][2])
    fig.colorbar(ax13, ax=ax[1][3])
    fig.colorbar(ax20, ax=ax[2][0])
    fig.colorbar(ax21, ax=ax[2][1])
    fig.colorbar(ax22, ax=ax[2][2])
    fig.colorbar(ax23, ax=ax[2][3])
    fig.colorbar(ax30, ax=ax[3][0])
    fig.colorbar(ax31, ax=ax[3][1])
    fig.colorbar(ax32, ax=ax[3][2])
    fig.colorbar(ax33, ax=ax[3][3])    
    fig.colorbar(ax40, ax=ax[4][0])
    fig.colorbar(ax41, ax=ax[4][1])
    fig.colorbar(ax42, ax=ax[4][2])
    fig.colorbar(ax43, ax=ax[4][3])

    # 不显示刻度尺、标
    for i in range(5):
        for j in range(4):    
            ax[i][j].axis('off')
    
    fig.suptitle('patient {} - slice {}'.format(series_uid[0][0], series_uid[1][0]))
    plt.savefig('picture/c{}_p{}_s{}_pre{}.jpg'.format(class_num, series_uid[0][0], series_uid[1][0], m))
    plt.close()
