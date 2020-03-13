import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
import numpy as np

from utils import  label2multichannel, mask2rgb

class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.dot(input.contiguous().view(-1),
                               target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # print(self.inter, torch.sum(input), torch.sum(target))
        dice = (2 * self.inter.float() + eps) / self.union.float()
        return dice

# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).cuda().zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()

#     for i, c in enumerate(zip(input, target)):
#         s = s + DiceCoeff().forward(c[0], c[1])
#     result = s/(i+1)
#     result = float(result[0])
#     return result

def dice_coeff(input, target):
    """Dice coeff
    input：预测结果，np，（240，240）
    target：标签，np，（240，240）
    """
    eps = 0.0001
    same = (input==target) # 相同的为1,不同的为0
    inter = float(same.sum()) # 交集
    union  = float(input.shape[0]*input.shape[1]+target.shape[0]*target.shape[1]) # 并集
    dice = (2 * inter + eps) / union
    return dice

def IoU_coeff(input, target):
    """IoU coeff
    input：预测结果，np，（240，240）
    target：标签，np，（240，240）
    """
    eps = 0.0001
    same = (input==target) # 相同的为1,不同的为0
    inter = float(same.sum()) # 交集
    union  = float(input.shape[0]*input.shape[1]) # 并集
    IoU = (inter + eps) / union
    return IoU

def evaluate(model, val_loader, device, class_num, test=True):
    """评估模型，dice值等"""
    # 评估模型
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        dices = 0 # 记录dice值
        IoUs = 0 # 记录IoU值
        for step, (patch, mask, _) in enumerate(val_loader):
            # 标签
            # mask = label2multichannel(mask.cpu()) # 单通道(0-8)变多通道
            mask = mask.cpu().numpy().reshape((240,240)) # 单通道1-9,(1,1,240,240)->(240,240)

            # 预测
            patch = patch.to(device)
            model.forward(patch, None, training=False)
            mask_pre = model.sample(testing=True).cpu().numpy() # 预测结果,(batch_size,9,240,240)，元素值为连续变量
            mask_pre = mask_pre.reshape((class_num,240,240)) # 降维
            
            # 统计每个像素的对应通道最大值所在通道即为对应类
            mask_pro = mask_pre.argmax(axis=0) # 计算每个batch的预测结果最大值，单通道,元素值0-8
            mask_pro += 1 # 元素值变为1-9
            
            # 计算dice值
            dices += dice_coeff(mask_pro, mask) # 单通道np(240)（1-9），多通道np(240,240)（1-9）
            IoUs += IoU_coeff(mask_pro, mask)
        # 除以slice数
        dices /= (step+1)
        IoUs /= (step+1)
        if test:
            print("Test -- Dice: {}, IoU: {}".format(dices, IoUs))
        else:
            print("Train -- Dice: {}, IoU: {}".format(dices, IoUs))
    
    return dices, IoUs
 


