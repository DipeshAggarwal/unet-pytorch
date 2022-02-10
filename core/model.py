from torch.nn import ConvTranspose2d
from torch.nn import ModuleList
from torch.nn import MaxPool2d
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import ReLU

from torchvision.transforms import CenterCrop
from torch.nn import functional as F

import torch

class Block(Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class Encoder(Module):
    
    def __init__(self, channels=(3, 16, 32, 64, 128)):
        super().__init__()
        self.enc_blocks = ModuleList(
            [
            Block(channels[i], channels[i+1]) for i in range(len(channels)-1)    
            ]
        )
        self.pool = MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        block_outputs = []
        
        for block in self.enc_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
            
        return block_outputs
    
class Decoder(Module):
    
    def __init__(self, channels=(128, 64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.up_convs = ModuleList(
            [
                ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
                for i in range(len(channels)-1)
            ]
        )
        self.dec_blocks = ModuleList(
            [
                Block(channels[i], channels[i+1]) for i in range(len(channels)-1)
            ]
        )
        
    def forward(self, x, enc_features):
        for i in range(len(self.channels)-1):
            x = self.up_convs[i](x)
            
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
            
        return x
    
    def crop(self, enc_features, x):
        H, W = x.shape[2:]
        enc_features = CenterCrop([H, W])(enc_features)
        
        return enc_features
    
class UNet(Module):
    
    def __init__(self,
                 enc_channels=(3, 16, 32, 64, 128),
                 dec_channels=(128, 64, 32, 16),
                 seg_classes=1,
                 retain_dim=True,
                 out_size=(64, 64)
    ):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        
        self.head = Conv2d(dec_channels[-1], seg_classes, kernel_size=1)
        self.retain_dim = retain_dim
        self.out_size = out_size
        
    def forward(self, x):
        enc_features = self.encoder(x)
        
        # Pass the encoder features to decoder
        dec_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        x = self.head(dec_features)
        
        if self.retain_dim:
            x = F.interpolate(x, self.out_size)
            
        return x
        
