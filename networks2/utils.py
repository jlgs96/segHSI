import torch
import torch.nn as nn
import torch.nn.functional as F
from box_convolution import BoxConv2d
class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2, act = 'relu'):
        
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        if act == 'relu':
            self.relu = nn.ReLU()
        elif act == 'prelu':
            self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, use_se = False, use_prelu = False, max_input_h=64, max_input_w=64, use_boxconv=False):
        super(unetConv2, self).__init__()
        if use_boxconv:
            n_boxes =4
            reparam_factor = 0.860
            if is_batchnorm:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),nn.BatchNorm2d(out_size//n_boxes), nn.ReLU(), 
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.BatchNorm2d(out_size),#nn.Dropout(p = 0.5)
                                           
                )
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),nn.BatchNorm2d(out_size//n_boxes), nn.ReLU(), 
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.BatchNorm2d(out_size),#nn.Dropout(p = 0.5)
                                           
                )
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.ReLU()
                )
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.ReLU())
        else:
            if is_batchnorm:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
                )
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())


        if use_se == True and use_prelu == True:
            self.se_layer1 = ChannelSELayer(out_size, act = 'prelu')
            self.se_layer2 = ChannelSELayer(out_size, act = 'prelu')
        elif use_se == True and use_prelu == False:
            self.se_layer1 = ChannelSELayer(out_size, act = 'ptsemseg.modelsrelu')
            self.se_layer2 = ChannelSELayer(out_size, act = 'relu')
        else:
            self.se_layer1 = None
            self.se_layer2 = None


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if self.se_layer1 is not None:
            outputs = self.se_layer1(outputs)
        outputs = self.conv2(outputs)
        if self.se_layer2 is not None:
            outputs = self.se_layer2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
