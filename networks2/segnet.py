from box_convolution import BoxConv2d
import torch.nn as nn
import torch.nn.functional as F
#DEFINICION DE UNA FCNN CON ARQUITECTURA SEGNET
class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            #UTILIZA UNA INICIALIZACION KAIMING NORMAL PARA EVITAR LUEGO LOS PROBLEMAS DE VANISHING Y EXPLODING GRADIENT Y QUE LA RED EN ESENCIA PETE O DEJE DE APRENDER
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, num_boxes, max_input_h=64, max_input_w=64, use_boxconv = False):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        self.max_input_h =max_input_h
        self.max_input_w =max_input_w
        num_boxes = 4
        reparam_factor = 0.860
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        max_input_h=max_input_h//2
        max_input_w=max_input_w//2
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        max_input_h=max_input_h//2
        max_input_w=max_input_w//2
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        max_input_h=max_input_h//2
        max_input_w=max_input_w//2
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        max_input_h=max_input_h//2
        max_input_w=max_input_w//2
        
        
        #####  AQUI MIRAR DONDE INSERTAR BOXCONVOLUTIONS!!!! #####
        
        if(use_boxconv):
            self.conv5_1 = nn.Conv2d(512, 512//num_boxes, 1, padding=0)
            self.conv5_1_bn = nn.BatchNorm2d(512//num_boxes)
            self.conv5_2 = BoxConv2d(512//num_boxes, num_boxes, max_input_h,max_input_w,reparametrization_factor=reparam_factor)
            self.conv5_2_bn = nn.BatchNorm2d(512)
            self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_3_bn = nn.BatchNorm2d(512)
            
        else:
            self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_1_bn = nn.BatchNorm2d(512)
            self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_2_bn = nn.BatchNorm2d(512)
            self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_3_bn = nn.BatchNorm2d(512)
        
        if(use_boxconv):
            self.conv5_3_D = nn.Conv2d(512, 512//num_boxes, 1, padding=0)
            self.conv5_3_D_bn = nn.BatchNorm2d(512//num_boxes)
            self.conv5_2_D = BoxConv2d(512//num_boxes, num_boxes, max_input_h,max_input_w,reparametrization_factor=reparam_factor)
            self.conv5_2_D_bn = nn.BatchNorm2d(512)
            self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_1_D_bn = nn.BatchNorm2d(512)
        else:
            self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_3_D_bn = nn.BatchNorm2d(512)
            self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_2_D_bn = nn.BatchNorm2d(512)
            self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
            self.conv5_1_D_bn = nn.BatchNorm2d(512)
            
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
    #DEFINICION DE LA FUNCION DE PROPAGACIÓN DE LOS DIFERENTES LAYERS    
    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = F.log_softmax(self.conv1_1_D(x), dim=1)
        x = self.conv1_1_D(x)
        return x
