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

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    result = s/(i+1)
    result = float(result[0])
    return result


def evaluate(model, val_loader, device, test=True):
    """评估模型，dice值等"""
    # 评估模型
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        dices = 0 # 记录dices
        for step, (patch, mask, _) in enumerate(val_loader):
            # 标签
            mask = label2multichannel(mask.cpu()) # 单通道变多通道

            # 预测
            patch = patch.to(device)
            model.forward(patch, None, training=False)
            mask_pre = model.sample(testing=True).cpu() # 预测结果,(batch_size,9,240,240)
            mask_pre_np = mask_pre.detach().numpy() # torch变numpy(batch_size,9,240,240)
            mask_pre_np = mask_pre_np.reshape((9,240,240)) # 降维
            
            ## 统计每个像素的对应通道最大值所在通道即为对应类
            mask_por = mask_pre_np.argmax(axis=0) # 计算每个batch的预测结果最大值，单通道,元素值0-8
            mask_por += 1 # 元素值变为1-9
            mask_por = mask_por.reshape((1,1,240,240)) # 变为多通道（batch_size,1,240,240）
            mask_por = torch.from_numpy(mask_por)
            mask_por = label2multichannel(mask_por) # 单通道变多通道
            
            # 计算dice值
            dices += dice_coeff(mask_por, mask)
            # print(type(dices),dices.shape,dices)
        dices /= (step+1)
        if test:
            print("Dice (test): {}".format(dices))
        else:
            print("Dice (train): {}".format(dices))

    return dices
 


