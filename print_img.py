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
    # dataset = BrainS18Dataset(root_dir='data/BrainS18', 
    #                         folders=['1_img'], 
    #                         class_num=class_num, 
    #                         file_names=['_reg_T1.png', '_segm.png'])
    dataset = BrainS18Dataset(root_dir='data/BrainS18', folders=['1_Brats17_CBICA_AAB_1_img'],
                              class_num=class_num,
                              file_names=['_reg_T1.png', '_segm.png'])

    # 数据划分并设置sampler（（固定训练集和测试集））
    dataset_size = len(dataset)  # 数据集大小
    
    imgs = np.zeros((8, 240, 240))
    labels = np.zeros((8, 240, 240))
    count = 0
    for i in range(48):
        # if i in (0,6,12,18,24,30,36,42):
        if i in (14,15,16,17,18,19,20,21):
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
    fig.savefig('picture/a_func_1_v2.png', format='png', transparent=True, dpi=300, pad_inches = 0)
    plt.close()

def func_2():
    """打印原图、标签、和预测结果"""
    # 参数
    class_num = param.class_num  # 选择分割类别数
    predict_time = 16  # 每张图预测次数,(1,4,8,16)
    latent_dim = 6  # 隐空间维度
    train_batch_size = 1  # 预测
    test_batch_size = 1  # 预测
    model_name = 'unet_e100_p6_c9_ld6.pt'  # 加载模型名称
    device = param.device  # 选gpu

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
    test_indices = list(range(dataset_size))
    test_sampler = SequentialSampler(test_indices)
    # 数据加载器
    test_loader = DataLoader(
        dataset, batch_size=test_batch_size, sampler=test_sampler)
    print("Number of test patches: {}".format(len(test_indices)))
    # 加载已经训练好的网络进行预测
    model = ProbabilisticUnet(input_channels=1,
                            num_classes=class_num,
                            num_filters=[32, 64, 128, 192],
                            latent_dim=latent_dim,
                            no_convs_fcomb=4,
                            beta=10.0)
    net = load_model(model=model,
                    path='model/{}'.format(model_name),
                    device=device)
    # 预测
    with torch.no_grad():
        for step, (patch, mask, series_uid) in enumerate(test_loader):
            if step == 14:
                for i in range(20):
                    print("Picture {} (patient {} - slice {})...".format(step,
                                                                        series_uid[0][0], series_uid[1][0]))
                    # 记录numpy
                    # (batch_size,1,240,240)->(1,240,240)
                    image_np = patch.cpu().numpy().reshape(240, 240)
                    label_np = mask.cpu().numpy().reshape(240, 240)  # (batch_size,1,240,240) 元素值1-10
                    label_np -= 1  # (batch_size,1,240,240) 元素值从1-10变为0-9
                    # 预测
                    patch = patch.to(device)
                    net.forward(patch, None, training=False)
                    # 预测结果, (batch_size,class_num,240,240)
                    mask_pre = net.sample(testing=True)
                    # torch变numpy(batch_size,class_num,240,240)
                    mask_pre_np = mask_pre.cpu().detach().numpy()
                    mask_pre_np = mask_pre_np.reshape((class_num, 240, 240))  # 降维
                    ## 统计每个像素的对应通道最大值所在通道即为对应类
                    # 计算每个batch的预测结果最大值，单通道,元素值0-9
                    mask_pro = mask_pre_np.argmax(axis=0)
                    # print(label_np.shape, image_np.shape, mask_pro.shape)
                    # 原图
                    # plt.figure(figsize=(1, 1))
                    # plt.imshow(image_np, aspect="auto", cmap="gray")
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    # plt.margins(0,0)
                    # plt.savefig('picture/a_func2_orgin.png', format='png', transparent=True, dpi=300, pad_inches = 0)
                    # plt.close()
                    # ground truth
                    # plt.figure(figsize=(1, 1))
                    # # 10 discrete colors,tab10,Paired
                    # cmap = plt.cm.get_cmap('tab10', 10)
                    # plt.imshow(label_np, cmap=cmap, aspect="auto", vmin=0, vmax=9)
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    # plt.margins(0,0)
                    # plt.savefig('picture/a_func2_gt.png', format='png', transparent=True, dpi=300, pad_inches = 0)
                    # plt.close()
                    # 预测结果
                    plt.figure(figsize=(1, 1))
                    # 10 discrete colors,tab10,Paired
                    cmap = plt.cm.get_cmap('tab10', 10)
                    plt.imshow(mask_pro, cmap=cmap, aspect="auto", vmin=0, vmax=9)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    plt.margins(0,0)
                    plt.savefig('picture/a_func2_pre{}.png'.format(i), format='png', transparent=True, dpi=300, pad_inches = 0)
                    plt.close()


def func_3():
    """打印部分mri图像，标签，单个预测结果
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
    model_name = 'punet_e128_c9_ld6_f1.pt' # 加载模型名称
    device = param.device  # 选gpu
    dataset_size = len(dataset)  # 数据集大小
    test_indices = list(range(dataset_size))
    test_sampler = SequentialSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
    model = ProbabilisticUnet(input_channels=1, 
                            num_classes=class_num, 
                            num_filters=[32,64,128,192], 
                            latent_dim=6,
                            no_convs_fcomb=4, 
                            beta=10.0)
    net = load_model(model=model, 
                    path='model/{}'.format(model_name), 
                    device=device)


    imgs = np.zeros((8, 240, 240))
    labels = np.zeros((8, 240, 240))
    predicts = np.zeros((8, 240, 240))

    with torch.no_grad():
        count = 0
        for step, (patch, mask, series_uid) in enumerate(test_loader):
            if step in (0,6,12,18,24,30,36,42):
                print("Picture {} (patient {} - slice {})...".format(step,
                                                    series_uid[0][0], series_uid[1][0]))
                # 记录numpy
                # (batch_size,1,240,240)->(1,240,240)
                image_np = patch.cpu().numpy().reshape(240, 240)
                label_np = mask.cpu().numpy().reshape(240, 240)  # (batch_size,1,240,240) 元素值1-10
                label_np -= 1  # (batch_size,1,240,240) 元素值从1-10变为0-9
                # 预测
                patch = patch.to(device)
                net.forward(patch, None, training=False)
                # 预测结果, (batch_size,class_num,240,240)
                mask_pre = net.sample(testing=True)
                # torch变numpy(batch_size,class_num,240,240)
                mask_pre_np = mask_pre.cpu().detach().numpy()
                mask_pre_np = mask_pre_np.reshape(
                    (class_num, 240, 240))  # 降维
                ## 统计每个像素的对应通道最大值所在通道即为对应类
                # 计算每个batch的预测结果最大值，单通道,元素值0-9
                mask_pro = mask_pre_np.argmax(axis=0)
                predicts[count] = mask_pro
                count += 1

    count = 0
    for i in range(48):
        if i in (0,6,12,18,24,30,36,42):
        # if i in (14,15,16,17,18,19,20,21):
            image, label, series_uid = dataset.__getitem__(i)
            image = image.numpy().reshape(240, 240)
            label -= 1
            label = label.numpy().reshape(240, 240)
            imgs[count] = image
            labels[count] = label
            count += 1

    fig, ax = plt.subplots(3, 8, sharey=True, figsize=(20, 7.5))
    cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors,tab10,Paired

    for i in range(8):
        ax[0][i].imshow(imgs[i], aspect="auto", cmap="gray")
        ax[1][i].imshow(labels[i], cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax[2][i].imshow(predicts[i], cmap=cmap, aspect="auto", vmin=0, vmax=9)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig('picture/a_func_3.png', format='png', transparent=True, dpi=300, pad_inches = 0)
    plt.close()


if __name__ == "__main__": 
    # func_1()
    # func_2()
    func_3()
    pass
    
