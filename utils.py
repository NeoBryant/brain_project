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


def label2multichannel(mask, class_num=9):
    """将单通道（元素值1-9）变为多通道（元素值0-1）
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

def mask2rgb(mask):
    """单通道变rgb图
    输入numpy，输出numpy
    mask: numpy
    mask.shape (240,240)
    """
    mask = mask.reshape((240,240))
    # 颜色rgb （黑，红，绿，蓝，黄，淡紫，青，紫，白）
    color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),
    (255,0,255),(0,255,255),(128,0,128),(255,255,255)] 
    img = np.zeros((240, 240, 3))
    for c in range(9):
        for x in range(240):
            for y in range(240):
                if int(mask[x,y]) == c:
                    img[x,y,0] = color[c][0]
                    img[x,y,1] = color[c][1]
                    img[x,y,2] = color[c][2]
    img /= 255

    return img


def show_curve(y1s, title='loss'):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    plt.plot(x, y1, label='train')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("picture/{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure: picture/{}.png'.format(title))


# def bayes_uncertain(image_np, label_np, results, count, the_class):
#     """
#     Keyword arguments:
#     image_np -- 原图
#     label_np -- 标签
#     results -- 同一张图片的不同预测结果
#     Return: 预测结果的均值和方差
#     """
#     results = np.array(results) # list->numpy
#     shape = results.shape
#     # mean_result
#     # variance_result

#     mean_result = np.zeros((shape[1], shape[2]))      # 均值
#     variance_result = np.zeros((shape[1], shape[2]))  # 方差

#     # 计算均值
#     for i in range(shape[0]):
#         mean_result += results[i]
#     mean_result /= shape[0]

#     # 计算方差
#     for i in range(shape[0]):
#         variance_result += np.square(mean_result-results[i])
#     variance_result /= shape[0]
    
#     # 显示保存图片
#     fig, ax = plt.subplots(2,2, sharey=True, figsize=(14,12))

#     ax[0][0].set_title("Original data")
#     ax[0][1].set_title("Ground Truth")
#     ax[1][0].set_title("mean predicted result")
#     ax[1][1].set_title("variance")

#     ax00 = ax[0][0].imshow(image_np, aspect="auto", cmap="gray")
#     ax01 = ax[0][1].imshow(label_np, aspect="auto")
#     ax10 = ax[1][0].imshow(mean_result, aspect="auto")
#     ax11 = ax[1][1].imshow(variance_result, aspect="auto")

#     fig.colorbar(ax00, ax=ax[0][0])
#     fig.colorbar(ax01, ax=ax[0][1])
#     fig.colorbar(ax10, ax=ax[1][0])
#     fig.colorbar(ax11, ax=ax[1][1])
    
#     # 保存
#     plt.savefig('picture/class_{}_mean_variance_threshold_{}.jpg'.format(the_class, count))

    
def bayes_uncertain(image_np, label_np, results, count, class_num):
    """
    Keyword arguments:
    image_np -- 原图
    label_np -- 标签
    results -- 同一张图片的不同预测结果, 8通道
    Return: 预测结果的均值和方差
    """
    results = np.array(results)  # list->numpy
    shape = results.shape

    result = results.reshape((9,240,240))
    

    # # mean_result
    # # variance_result
    # mean_result = np.zeros((shape[1], shape[2]))      # 均值
    # variance_result = np.zeros((shape[1], shape[2]))  # 方差

    # # 显示保存图片
    fig, ax = plt.subplots(6, 2, sharey=True, figsize=(14, 36))

    ax[0][0].set_title("Original")
    ax[0][1].set_title("Ground Truth")
    ax[1][0].set_title("class 1")
    ax[1][1].set_title("class 2")
    ax[2][0].set_title("class 3")
    ax[2][1].set_title("class 4")
    ax[3][0].set_title("class 5")
    ax[3][1].set_title("class 6")
    ax[4][0].set_title("class 7")
    ax[4][1].set_title("class 8")
    ax[5][0].set_title("class 9")

    # ax00 = ax[0][0].imshow(image_np, aspect="auto", cmap="gray")
    ax00 = ax[0][0].imshow(image_np, aspect="auto", cmap="gray")
    ax01 = ax[0][1].imshow(label_np, aspect="auto")
    ax10 = ax[1][0].imshow(result[0], aspect="auto")
    ax11 = ax[1][1].imshow(result[1], aspect="auto")
    ax20 = ax[2][0].imshow(result[2], aspect="auto")
    ax21 = ax[2][1].imshow(result[3], aspect="auto")
    ax30 = ax[3][0].imshow(result[4], aspect="auto")
    ax31 = ax[3][1].imshow(result[5], aspect="auto")
    ax40 = ax[4][0].imshow(result[6], aspect="auto")
    ax41 = ax[4][1].imshow(result[7], aspect="auto")
    ax50 = ax[5][0].imshow(result[8], aspect="auto")

    fig.colorbar(ax00, ax=ax[0][0])
    fig.colorbar(ax01, ax=ax[0][1])
    fig.colorbar(ax10, ax=ax[1][0])
    fig.colorbar(ax11, ax=ax[1][1])
    fig.colorbar(ax20, ax=ax[2][0])
    fig.colorbar(ax21, ax=ax[2][1])
    fig.colorbar(ax30, ax=ax[3][0])
    fig.colorbar(ax31, ax=ax[3][1])
    fig.colorbar(ax40, ax=ax[4][0])
    fig.colorbar(ax41, ax=ax[4][1])
    fig.colorbar(ax50, ax=ax[5][0])

    # 保存
    plt.savefig(
        'picture/class_{}_{}.jpg'.format(class_num, count))


def save_result_img(orig, mask, pred,step):
    """保存原图、标签、预测结果进行对比"""
    fig, ax = plt.subplots(2, 2, sharey=True, figsize=(14, 12))
    
    ax[0][0].set_title("Original")
    ax[0][1].set_title("Ground Truth")
    ax[1][0].set_title("predict")

    ax00 = ax[0][0].imshow(orig, aspect="auto", cmap="gray")
    ax01 = ax[0][1].imshow(mask, aspect="auto")
    ax10 = ax[1][0].imshow(pred, aspect="auto")

    fig.colorbar(ax00, ax=ax[0][0])
    fig.colorbar(ax01, ax=ax[0][1])
    fig.colorbar(ax10, ax=ax[1][0])
    
    plt.savefig('picture/compare_{}.jpg'.format(step))

    return