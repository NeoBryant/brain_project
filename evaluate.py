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

# def dice_coeff(input, target, class_num):
#     """Dice coeff
#     包括背景0类， 
#     input：预测结果，np，（240，240），0-9
#     target：标签，np，（240，240），0-9
#     """
#     eps = 0.0001
#     same = (input==target) # 相同的为1,不同的为0
#     inter = float(same.sum()) # 交集
#     union  = float(input.shape[0]*input.shape[1]+target.shape[0]*target.shape[1]) # 并集
#     dice = (2 * inter + eps) / union

#     return dice

def dice_coeff(input, target, class_num):
    """Dice coeff
    包括背景0类， 
    input：预测结果，np，（240，240），0-9
    target：标签，np，（240，240），0-9
    """
    eps = 0.0001
    dices = np.zeros((class_num))
    # count = 0 # 计数有几类是在预测结果或标签中的
    for i in range(class_num):
        hasClass = False # 判断是否有该类
        input_temp = (input == i)
        target_temp = (target == i)
        union = float(input_temp.sum() + target_temp.sum()) # 分母
        same = input_temp * target_temp # 两个矩阵相交
        inter = float(same.sum())  # 交集，分子
        if union == 0:
            dices[i] = 1
        else:
            dices[i] = (2 * inter) / union    
    return dices

# def IoU_coeff(input, target, class_num):
#     """IoU coeff
#     包括背景0类， 
#     input：预测结果，np，（240，240）,0-9
#     target：标签，np，（240，240）,0-9
#     """
#     eps = 0.0001
#     same = (input==target) # 相同的为1,不同的为0
#     inter = float(same.sum()) # 交集
#     union  = float(input.shape[0]*input.shape[1]) # 并集
#     IoU = (inter + eps) / union
#     return IoU

def IoU_coeff(input, target, class_num):
    """IoU coeff
    包括背景0类， 
    input：预测结果，np，（240，240），0-9
    target：标签，np，（240，240），0-9
    """
    eps = 0.0001
    IoUs = np.zeros((class_num))
    # count = 0 # 计数有几类是在预测结果或标签中的
    for i in range(class_num):
        hasClass = False # 判断是否有该类
        input_temp = (input == i)
        target_temp = (target == i)
        # union = float(input_temp.sum() + target_temp.sum()) # 分母
        inter = input_temp * target_temp # 两个矩阵相交
        inter = float(inter.sum())  # 交集，分子
        union = input_temp+target_temp
        union = (union != 0)
        union = float(union.sum())
        if union == 0:
            IoUs[i] = 1
        else:
            IoUs[i] = inter / union
    # IoU = float(IoUs.sum())/class_num
    
    return IoUs

def evaluate(model, val_loader, device, class_num, test=True):
    """评估模型，dice值等"""
    # 评估模型
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        dices = np.zeros((class_num)) # 记录dice值
        IoUs = 0 # 记录IoU值
        for step, (patch, mask, _) in enumerate(val_loader):
            # 标签
            # mask = label2multichannel(mask.cpu()) # 单通道(0-8)变多通道
            mask = mask.cpu().numpy().reshape((240,240)) # 单通道1-10,(1,1,240,240)->(240,240)
            mask -= 1 # 单通道0-9
            # 预测
            patch = patch.to(device)
            model.forward(patch, None, training=False)
            mask_pre = model.sample(testing=True).cpu().numpy() # 预测结果,(batch_size,9,240,240)，元素值为连续变量
            mask_pre = mask_pre.reshape((class_num,240,240)) # 降维
            
            # 统计每个像素的对应通道最大值所在通道即为对应类
            mask_pro = mask_pre.argmax(axis=0) # 计算每个batch的预测结果最大值，单通道,元素值0-9
            # mask_pro += 1 # 元素值变为1-10
            
            # 计算dice值
            dices += dice_coeff(mask_pro, mask, class_num) # 单通道np(240)（0-9），多通道np(240,240)（0-9）
            IoUs += IoU_coeff(mask_pro, mask, class_num)

        # 除以slice数
        dices /= (step+1)
        IoUs /= (step+1)
        if test:
            print("Test: \n-- Dice: {},{}\n-- IoU: {},{}".format(np.mean(dices),dices.tolist(), 
                np.mean(IoUs),IoUs.tolist()))
        else:
            print("Train: \n-- Dice: {},{}\n-- IoU: {},{}".format(np.mean(dices),dices.tolist(), 
                np.mean(IoUs),IoUs.tolist()))
    
    return dices, IoUs
 


