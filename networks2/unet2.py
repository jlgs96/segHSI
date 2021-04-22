import torch.nn as nn
from .utils import unetConv2, unetUp
import torch.nn.functional as F
from .utils import ChannelSELayer



def filters2filtersboxconv(filters, n_boxes=4):
    newfilters = []; enc=False; mul=0
    for f in filters:
        if f % n_boxes == 0 and enc == False:
            newfilters.append(f)
            enc=True
            mul = f
        else:
            if enc:
                newfilters.append(mul*2)
                mul*=2
            else:
                while f % n_boxes != 0: f += 1
                mul = f
                newfilters.append(f)
                enc=True
    return newfilters


class unet(nn.Module):
    def __init__(self,in_channels = 3,out_channels = 21, feature_scale=4,use_SE = False, use_PReLU = False,  is_deconv=True, is_batchnorm=True, max_input_h=64, max_input_w=64, use_boxconv=False):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.use_SE = use_SE
        self.use_PReLU = use_PReLU
        self.use_boxconv = use_boxconv
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        print(filters)
        filters = filters2filtersboxconv(filters)
        print(filters)
        #exit()
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2
        
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2
        
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2
        
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h = max_input_h, max_input_w = max_input_w, use_boxconv=self.use_boxconv)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
class unetm(nn.Module):
    '''
    mini U-Net architecture with 2 downsampling & upsampling blocks and one bottleneck
    with Squeeze and Excitation layers
    
    Parameters:
        in_channels     -- number of input channelsfrom .selayer import ChannelSELayer

        out_channels    -- number of output channels
        feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
        is_deconv       -- boolean flag to indicate if interpolation or de-convolution
                            should be used for up-sampling
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
        use_SE          -- boolean flag to indicate SE blocks usage
        use_PReLU       -- boolean flag to indicate activation between linear layers in SE 
                            (relu vs. prelu)
    '''
    def __init__(self, in_channels=3, out_channels = 21, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, use_SE = False, use_PReLU = False, max_input_h=64, max_input_w=64, use_boxconv=False):
        super(unetm, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.use_SE = use_SE
        self.use_PReLU = use_PReLU
        self.use_boxconv = use_boxconv
        filters = [64, 128, 256, 512, 1024]
        #filters = [128, 256, 512, 1024, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        print(filters)
        filters = filters2filtersboxconv(filters)
        print(filters)
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU,  max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h //2 
        max_input_w = max_input_w //2 

        self.center = unetConv2(filters[1], filters[2], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, use_boxconv=self.use_boxconv, max_input_h=max_input_h, max_input_w=max_input_w)

        # upsampling
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

        
    
        
    def forward(self, inputs):
        
        conv1 = F.relu(self.conv1(inputs))
        maxpool1 = self.maxpool1(conv1)

        conv2 = F.relu(self.conv2(maxpool1))
        maxpool2 = self.maxpool2(conv2)

        center = self.center(maxpool2)
        up2 = self.up_concat2(conv2, center)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
  
