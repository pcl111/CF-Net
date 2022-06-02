import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from vgg_atrou import vgg16_bn
from modules.bn import InPlaceABNSync as BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        
        channels =512
        psp_out_feature=1024
        self.backbone = vgg16_bn()
        self.PSP = PSPModule(channels, out_channels=psp_out_feature, batch_norm=True)
        h_psp_out_feature = int(psp_out_feature / 2)
        q_psp_out_feature = int(psp_out_feature / 4)
        e_psp_out_feature = int(psp_out_feature / 8)
        self.upsampling1 = PSPUpsampling(psp_out_feature, h_psp_out_feature, batch_norm=True)
        self.upsampling2 = PSPUpsampling(h_psp_out_feature, q_psp_out_feature, batch_norm=True)
        self.upsampling3 = PSPUpsampling(q_psp_out_feature,  e_psp_out_feature, batch_norm=True)

        self.classifier = BiSeNetOutput(e_psp_out_feature, n_classes)
        ## here self.sp is deleted
#        self.ffm = FeatureFusionModule(256, 256)
#        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
#        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x,deep):
        H, W = x.size()[2:]
        _,_,_,x = self.backbone(x)
        o = self.PSP(x)
        o = self.upsampling1(o)
        o = self.upsampling2(o)
        o = self.upsampling3(o)

        o = F.upsample(o, size=(H, W), mode='bilinear')
        o = self.classifier(o)

        return o,o
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if  isinstance(child, Layer2):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class BiSeNetOutput(nn.Module):
    def __init__(self, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
    
class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_factors=(1, 2, 3, 6), batch_norm=True):
        super().__init__()
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = ConvBNReLU(512,512,ks=1,stride=1,paddding=0)
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.conv2 = ConvBNReLU(512,512,ks=1,stride=1,paddding=0)
        self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.conv3 = ConvBNReLU(512,512,ks=1,stride=1,paddding=0)
        self.avgpool4 = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.conv4 = ConvBNReLU(512,512,ks=1,stride=1,paddding=0)
        
        self.out = ConvBNReLU(2048,1024,ks=1,stride=1,paddding=0)
        

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1 = self.conv1(self.avgpool1(x))
        x2 = self.conv2(self.avgpool2(x))
        x3 = self.conv3(self.avgpool3(x))
        x4 = self.conv4(self.avgpool4(x))
        x1 = F.upsample(x1, size=(h, w), mode='bilinear')
        x2 = F.upsample(x2, size=(h, w), mode='bilinear')
        x3 = F.upsample(x3, size=(h, w), mode='bilinear')
        x4 = F.upsample(x4, size=(h, w), mode='bilinear')
        
        o = torch.cat([x1,x2,x3,x4],dim=1)
        o = self.out(o)
        return o

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class PSPUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.layer = ConvBNReLU(in_chan=in_channels,out_chan=out_channels,ks=3,stride=1,paddding=1)
        self.init_weight()

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(x, size=(h, w), mode='bilinear')
        return self.layer(p)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
    
    
class Layer2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Layer2, self).__init__()
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 6, 1)  # get (nB, 5, 36, 100)
        )

    def forward(self, x):
        x = self.layer2(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params 

class MS(nn.Module):
    def __init__(self, ms_ks=9, *args, **kwargs):
        super(MS, self).__init__()
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        
    def forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params    

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)
