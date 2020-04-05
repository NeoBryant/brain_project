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

onlyEva = True # 是否只是评估

# 参数
class_num = param.class_num # 选择分割类别数
predict_time = 16 # 每张图预测次数,(1,4,8,16)
latent_dim = 6 # 隐空间维度

train_batch_size = 1 # 预测
test_batch_size = 1 # 预测

model_name = 'punet_e128_c9_ld6_f070.pt' # 加载模型名称
device = param.device # 选gpu

# 选择数据集
# dataset = BrainS18Dataset(root_dir='data/BrainS18', 
#                           folders=['070_img'], 
#                           class_num=class_num, 
#                           file_names=['_reg_T1.png', '_segm.png'])
dataset = BrainS18Dataset(root_dir='data/BrainS18', 
                          folders=['1_img','4_img','5_img','7_img','14_img','148_img',], 
                          class_num=class_num, 
                          file_names=['_reg_T1.png', '_segm.png'])


# dataset = BrainS18Dataset(root_dir='data/BrainS18', folders=['4_Brats17_CBICA_AAB_1_img'],
#                           class_num=class_num,
#                           file_names=['_reg_T1.png', '_segm.png'])

# 数据划分并设置sampler（（固定训练集和测试集））
dataset_size = len(dataset)  # 数据集大小
# split = param.split # 划分
# indices = param.indices # 数据选择
# train_indices, test_indices = indices[split:], indices[:split]
# train_indices, test_indices = indices[:], indices[:]
test_indices = list(range(dataset_size))

# train_sampler = SequentialSampler(train_indices)
test_sampler = SequentialSampler(test_indices)

# 数据加载器
# train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=test_sampler)
print("Number of test patches: {}".format(len(test_indices)))

# 加载已经训练好的网络进行预测
model = ProbabilisticUnet(input_channels=1, 
                        num_classes=class_num, 
                        num_filters=[32,64,128,192], 
                        latent_dim=latent_dim,
                        no_convs_fcomb=4, 
                        beta=10.0)
net = load_model(model=model, 
                path='model/{}'.format(model_name), 
                device=device)

# 预测
with torch.no_grad():
    if not onlyEva:
        for step, (patch, mask, series_uid) in enumerate(test_loader): 
            print("Picture {} (patient {} - slice {})...".format(step, series_uid[0][0], series_uid[1][0]))
            mask_pros = [] # 记录每次预测结果（选择最大值后的）
            mask_pres = [] # 记录每次预测结果
            # 记录numpy
            image_np = patch.numpy().reshape(240,240) # (batch_size,1,240,240)->(1,240,240)
            label_np = mask.numpy().reshape(240,240) # (batch_size,1,240,240) 元素值1-10
            label_np -= 1 # (batch_size,1,240,240) 元素值从1-10变为0-9
            
            # 预测predict_time次计算方差
            for i in range(predict_time):
                patch = patch.to(device)
                net.forward(patch, None, training=False) 
                mask_pre = net.sample(testing=True) # 预测结果, (batch_size,class_num,240,240)
                
                # 记录softmax后的值
                p_value = F.softmax(mask_pre, dim=1)
                p_value = p_value.cpu().numpy().reshape((class_num,240,240)) # 降维
                mask_pres.append(p_value)

                # torch变numpy(batch_size,class_num,240,240)
                mask_pre_np = mask_pre.cpu().detach().numpy()
                mask_pre_np = mask_pre_np.reshape((class_num,240,240)) # 降维

                ## 统计每个像素的对应通道最大值所在通道即为对应类
                mask_pro = mask_pre_np.argmax(axis=0) # 计算每个batch的预测结果最大值，单通道,元素值0-9
                mask_pros.append(mask_pro)

            # 计算均值和方差,并保存相应图片
            cal_variance(image_np, label_np, mask_pros, mask_pres, class_num, series_uid)  
    
    # 评估
    print("Evaluating ...")
    # evaluate(net, train_loader, device, class_num, test=False)     
    evaluate(net, test_loader, device, class_num, test=True)   
