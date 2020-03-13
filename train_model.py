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
class_num = 9 # 选择分割类别数
epochs = 50  # 训练周期
learning_rate = 1e-4 # 学习率

train_batch_size = 16 # 训练
test_batch_size = 1 # 预测

model_name = 'unet_1.pt' # 待保存的模型名
device = param.device # 选择cpu

# 数据集
dataset = BrainS18Dataset(root_dir='data/BrainS18', class_num=class_num)

# 数据划分并设置sampler（（固定训练集和测试集））
dataset_size = len(dataset)  # 数据集大小
# indices = list(range(dataset_size))
# split = int(np.floor(0.1 * dataset_size))
# np.random.shuffle(indices)
# indices = [258, 176, 46, 317, 114, 128, 163, 239, 150, 249, 191, 166, 64, 284, 298, 70, 109, 57, 332, 325, 76, 253, 209, 274, 184, 186, 185, 222, 161, 61, 233, 2, 195, 292, 5, 106, 309, 6, 293, 205, 194, 269, 137, 135, 256, 211, 75, 130, 322, 289, 232, 40, 77, 16, 241, 192, 245, 121, 39, 314, 238, 242, 24, 96, 259, 7, 326, 275, 225, 42, 62, 88, 122, 105, 12, 219, 278, 175, 297, 305, 126, 312, 138, 290, 65, 134, 162, 0, 1, 318, 103, 220, 67, 334, 146, 268, 189, 260, 236, 90, 63, 287, 49, 193, 264, 296, 212, 237, 140, 85, 144, 198, 306, 23, 27, 117, 164, 78, 270, 262, 243, 30, 111, 38, 190, 148, 149, 320, 95, 34, 54, 329, 308, 22, 180, 266, 196, 141, 118, 82, 131, 36, 120, 110, 143, 174, 18, 324, 20, 136, 247, 261, 79, 331, 215, 302, 178, 124, 168, 273, 107, 97, 55, 291, 58, 216, 4,
#            17, 251, 11, 47, 43, 37, 283, 74, 101, 154, 229, 230, 244, 327, 172, 3, 73, 48, 104, 280, 323, 127, 33, 86, 321, 335, 231, 100, 265, 92, 129, 147, 201, 311, 281, 315, 279, 29, 142, 159, 252, 301, 300, 44, 246, 282, 81, 139, 158, 32, 204, 119, 227, 179, 165, 303, 68, 208, 263, 15, 8, 25, 156, 60, 113, 125, 71, 94, 255, 307, 285, 210, 203, 133, 35, 157, 83, 328, 98, 267, 226, 84, 234, 218, 224, 213, 313, 99, 14, 10, 152, 155, 19, 276, 199, 254, 206, 183, 288, 56, 182, 173, 250, 188, 223, 72, 248, 108, 200, 51, 28, 272, 50, 116, 217, 310, 87, 235, 89, 316, 145, 187, 294, 53, 59, 52, 170, 26, 112, 221, 299, 295, 31, 167, 319, 153, 160, 277, 304, 207, 9, 169, 197, 151, 80, 132, 330, 171, 333, 21, 123, 41, 271, 66, 257, 214, 93, 240, 181, 45, 228, 177, 115, 13, 202, 102, 91, 69, 286]
split = param.split
indices = param.indices
train_indices, test_indices = indices[split:], indices[:split]

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
                        latent_dim=2, 
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
