import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from load_Brain_data import BrainS18Dataset
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation, label2multichannel
from save_load_net import save_model, load_model

from evaluate import evaluate
import param

# 参数
class_num = param.class_num # 选择分割类别数
epochs = 100  # 训练周期
learning_rate = 1e-4 # 学习率
latent_dim = 6 # 隐空间维度

train_batch_size = 16 # 训练
test_batch_size = 1 # 预测

model_name = 'unet_0.pt' # 待保存的模型名
device = param.device # 选择cpu

# 打印记录训练超参数


# 数据集
dataset = BrainS18Dataset(root_dir='data/BrainS18', 
                          folders=['1_img', '5_img', '7_img', '4_img', '148_img', '070_img', '14_img'],
                          class_num=class_num,
                          file_names=['_FLAIR.png', '_reg_IR.png', '_reg_T1.png', '_segm.png'])

# 数据划分并设置sampler（（固定训练集和测试集））
dataset_size = len(dataset)  # 数据集大小
split = param.split
indices = param.indices
train_indices, test_indices = indices[split:], indices[:split] # 用上述所有数据训练

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# 数据加载器
train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler) # 训练
train_eval_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=train_sampler) # 评估
test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=test_sampler) # 评估
# print("Number of training/test patches:", (len(train_indices),len(test_indices)))
print("Number of training/test patches: {}/{}".format(len(train_indices),len(test_indices)))

# 网络模型
net = ProbabilisticUnet(input_channels=1, 
                        num_classes=class_num, 
                        num_filters=[32,64,128,192], 
                        latent_dim=latent_dim, 
                        no_convs_fcomb=4, 
                        beta=10.0)
net.to(device)

# 优化器
optimizer = torch.optim.Adam(net.parameters(), 
                            lr=learning_rate, 
                            weight_decay=0)
# 训练模型并保存
try:
    # 训练
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        # 训练
        net.train()
        losses = 0 # 计算平均loss值
        for step, (patch, mask, _) in enumerate(train_loader): 
            patch = patch.to(device)
            mask = mask.to(device)
            # mask = torch.unsqueeze(mask,1) (batch_size,240,240)->(batch_size,1,240,240)
            net.forward(patch, mask, training=True)
            # label通道数1->9，单通道（1-9）变多通道（0/1）
            mask = label2multichannel(mask.cpu(), class_num)
            mask = mask.to(device)
            elbo = net.elbo(mask)
            ###
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            losses += loss
            if step%10 == 0:
                print("-- [step {}] loss: {}".format(step, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("-- [step {}] loss: {}".format(step, loss))
        # 评估
        losses /= (step+1)
        print("Loss (Train): {}".format(losses))
        evaluate(net, train_eval_loader, device, class_num, test=False)     
        evaluate(net, test_loader, device, class_num, test=True)        
except KeyboardInterrupt as e:
    print('KeyboardInterrupt: {}'.format(e))
except Exception as e:
    print('Exception: {}'.format(e))
finally:
    # 保存模型
    print("saving the trained net model -- {}".format(model_name))
    save_model(net, path='model/{}'.format(model_name))
