import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.parameter import Parameter
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a   

class CrossResolutionWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio = 16):
        super(CrossResolutionWeighting, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(
            channels, 
            int(channels / ratio), 
            kernel_size= 1, 
            stride = 1,
            bias=False) 
        self.conv2 = nn.Conv2d(
            int(channels / ratio),
            channels,
            kernel_size = 1, 
            stride = 1,
            bias=False) 
        self.relu1 = nn.LeakyReLU()
        self.sigmoid1 = nn.Sigmoid()
    def forward(self, x):
        _, _, h,w = x.shape
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.sigmoid1(out)
        out = x * out
        return out

class SplitSpatialSpectralConvWithShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size = 3, spectral_kernel_size = 3):
        super(SplitSpatialSpectralConvWithShuffle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = gcd(in_channels,out_channels)  
        self.spatial_conv = nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=spatial_kernel_size, 
                                      padding=spatial_kernel_size//2,
                                      groups = gcd(in_channels,out_channels))
        self.spectral_conv = nn.Conv2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=1)

        self.crossResolutionWeighting = CrossResolutionWeighting(out_channels,16)

    def forward(self, x):
        batch_size, _, height , width  = x.shape
        spatial_out = self.spatial_conv(x)
        spectral_out = self.spectral_conv(x)
        combined = spatial_out + spectral_out 
        combined = self.channel_shuffle(combined, self.group)
        combined = self.crossResolutionWeighting(combined)
        
        return combined

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x1 = x
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        
        return x

class SAH(nn.Module):
    def __init__(self, in_channels=28, out_channels=28, residual=False, circular_padding=True, cat=True):
        super(SAH, self).__init__()
        """compact unet (4 levels)"""
        self.name = 'unet'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.b1 = Parameter(torch.ones(1), requires_grad=True)
        self.b2 = Parameter(torch.ones(1), requires_grad=True)
        self.s1 = Parameter(torch.ones(1), requires_grad=True)
        self.s2 = Parameter(torch.ones(1), requires_grad=True)
        torch.nn.init.normal_(self.b1, mean=1, std=0.01)
        torch.nn.init.normal_(self.b2, mean=1, std=0.01)
        torch.nn.init.normal_(self.s1, mean=1, std=0.01)
        torch.nn.init.normal_(self.s2, mean=1, std=0.01)
        self.residual = residual
        self.cat = cat
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        def conv_block(ch_in, ch_out, circular_padding=False):
            return nn.Sequential(
                SplitSpatialSpectralConvWithShuffle(ch_in, ch_out),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(),
                SplitSpatialSpectralConvWithShuffle(ch_out, ch_out),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(),
            )

        def up_conv(ch_in, ch_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                SplitSpatialSpectralConvWithShuffle(ch_in, ch_out),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(),
            )

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        cat_dim = 1
        s1 = self.s1
        s2 = self.s2
        b1 = self.b1
        b2 = self.b2
        x1 = self.Conv1(x) # 28->64
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)# 64->128
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)# 128->256
        d3 = self.Up3(x3)# 256->128
        if self.cat:
            diff_y = x2.size()[2] - d3.size()[2]
            diff_x = x2.size()[3] - d3.size()[3]
            d3 = F.pad(d3, [diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2])
            result_d3 = torch.zeros_like(d3)
            if x2.shape[1] == 128:
                result_d3[:,:128] = d3 [:,:128] * b1
                result_d3[:,128:256] = d3[:,128:256]
                x2 = Fourier_filter(x2, threshold=1, scale=s1)
            d3 = torch.cat((x2, result_d3), dim=cat_dim) 
            d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        if self.cat:
            diff_y = x1.size()[2] - d2.size()[2]
            diff_x = x1.size()[3] - d2.size()[3]
            d2 = F.pad(d2, [diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2])
            result = torch.zeros_like(d2)
            if x1.shape[1] == 64:
                result[:,:64] = d2 [:,:64] * b2
                result[:,64:128] = d2[:,64:128]
                x1 = Fourier_filter(x1, threshold=1, scale=s2)
            d2 = torch.cat((x1, result), dim=cat_dim)
            d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = d1 + x 
        return (out)[:,:,:,0:256]