import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net model for image segmentation, comprising symmetric down- and upsampling paths with skip connections.
    """
    def __init__(self):
        super(UNet, self).__init__()

        self.l1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1), nn.ReLU())
        self.p1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.l3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1), nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1), nn.ReLU())
        self.p2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(8)
        
        self.l5 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1), nn.ReLU())
        self.l6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.ReLU())
        self.p3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.l7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.ReLU())
        self.l8 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.l9 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.p4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.l10 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.l11 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.l12 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.ReLU())
        self.bn5 = nn.BatchNorm2d(256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.l13 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1), nn.ReLU())
        self.l14 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.ReLU())
        self.l15 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1), nn.ReLU())
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.l16 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1), nn.ReLU())
        self.l17 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), nn.ReLU())
  
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.l18 = self.l18 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=20, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.p1(out)
        out = self.bn1(out)
        
        out = self.l3(out)
        out = self.l4(out)
        out = self.p2(out)
        out = self.bn2(out)

        out = self.l5(out)
        out = self.l6(out)
        out = self.p3(out)
        out = self.bn3(out)
    
        out = self.l7(out)
        out = self.l8(out)
        out = self.l9(out)
        out = self.p4(out)
        out = self.bn4(out)
        
        out = self.up1(out)
        out = self.l10(out)
        out = self.l11(out)
        out = self.l12(out)
        out = self.bn5(out)
        
        out = self.up2(out)
        out = self.l13(out)
        out = self.l14(out)
        out = self.l15(out)
        
        out = self.up3(out)
        out = self.l16(out)
        out = self.l17(out)

        out = self.up4(out)
        out = self.l18(out)
        
        return out

