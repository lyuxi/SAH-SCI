import torch
import torch.nn as nn
import torch.nn.functional as F
def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 28,
                 num_classes: int = 28,
                 bilinear: bool = False,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits


def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class ADMM_net(nn.Module):

    def __init__(self,gamma=0.01):
        super(ADMM_net, self).__init__()
        self.unet1 = UNet(28, 28)
        self.unet2 = UNet(28, 28)
        self.unet3 = UNet(28, 28)
        self.unet4 = UNet(28, 28)
        self.unet5 = UNet(28, 28)
        self.unet6 = UNet(28, 28)
        self.unet7 = UNet(28, 28)
        self.gamma = gamma 

    def forward(self, y, input_mask=None):
        if input_mask == None:
            Phi = torch.rand((1, 28, 256, 310)).cuda()
            Phi_s = torch.rand((1, 256, 310)).cuda()
        else:
            Phi, Phi_s = input_mask
        x_list = []
        theta = At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet1(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet2(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet3(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet4(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet5(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet6(x1)
        theta = shift_3d(theta)
        b = b- (x-theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        theta = self.unet7(x1)
        theta = shift_3d(theta)
        return theta[:, :, :, 0:256]
