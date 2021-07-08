import torch
import torch.nn as nn


class ConvLRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.activation(x)
        return x

    
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            ConvLRelu(in_channels, out_channels),
            ConvLRelu(out_channels, out_channels),
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = DoubleConvBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
    def forward(self, x):
        before_pool = self.conv_block(x)
        x = self.max_pool(before_pool)
        return x, before_pool
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()              
        self.conv_block = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, y):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv_block(torch.cat([x, y], dim=1))

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_filters=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = EncoderBlock(in_channels, n_filters)
        self.enc2 = EncoderBlock(n_filters, n_filters * 2)
        self.enc3 = EncoderBlock(n_filters * 2, n_filters * 4)
        self.enc4 = EncoderBlock(n_filters * 4, n_filters * 8)
        
        self.center = DoubleConvBlock(n_filters * 8, n_filters * 16)
        
        self.dec4 = DecoderBlock(n_filters * (16 + 8), n_filters * 8)
        self.dec3 = DecoderBlock(n_filters * (8 + 4), n_filters * 4)
        self.dec2 = DecoderBlock(n_filters * (4 + 2), n_filters * 2)
        self.dec1 = DecoderBlock(n_filters * (2 + 1), n_filters)

        self.final = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.float()
        x, enc1 = self.enc1(x)
        x, enc2 = self.enc2(x)
        x, enc3 = self.enc3(x)
        x, enc4 = self.enc4(x)

        center = self.center(x)

        dec4 = self.dec4(center, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        
        final = self.final(dec1)

        return final
        