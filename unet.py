from unet_blocks import *
import torch.nn.functional as F

class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels        # 输入图像通道数
        self.num_classes = num_classes              # 分类任务类别数
        self.num_filters = num_filters              # u-net每一层的滤波器
        self.padding = padding                      # padding大小
        self.activation_maps = []                   # 激活层
        self.apply_last_layer = apply_last_layer    # 最后一层
        self.contracting_path = nn.ModuleList()     # 链接每一层

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output  # 第一层对应通道数，其他层的输入为前一层的输出
            output = self.num_filters[i]                       # 每一层的输出

            if i == 0: pool = False  # 第一层不pooling
            else: pool = True   # 除第一层外其他层都有pooling
            
            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        # 最后一层
        if self.apply_last_layer: 
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            #nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            #nn.init.normal_(self.last_layer.bias)


    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])
 
        del blocks

        #Used for saving the activations and plotting 用于保存激活和绘图
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)
        
        
        return x
