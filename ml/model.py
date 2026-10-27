import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> LeakyReLU) * 2 with optional residual connection"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class OceanEmbedCNN(nn.Module):
    """
    CNN Encoder-Decoder with Skip Connections (U-Net style) and Multi-Scale Receptive Field.
    Specifically designed for satellite-to-subsurface ocean temperature reconstruction (PS 26066).
    
    Input: (B, 5, 101, 241) - Surface Satellite Observations (SST, SSS, SSH, Current Speed, Wind Speed)
    Output: (B, 15, 101, 241) - 3D Subsurface Temperature Field across 15 standard depths (0m to 1000m)
    """
    def __init__(self, in_channels=5, out_channels=15, base_channels=32):
        super(OceanEmbedCNN, self).__init__()
        
        # --- ENCODER ---
        # Level 1: Full resolution (101, 241)
        self.enc1 = DoubleConv(in_channels, base_channels)
        
        # Level 2: Downsampled by 2 (~50, 120)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = DoubleConv(base_channels * 2, base_channels * 2)
        
        # Level 3: Downsampled by 4 (~25, 60)
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 4)
        
        # --- BOTTLENECK ---
        # Dilated convolutions to capture basin-wide teleconnections (Rossby & Kelvin waves)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # --- DECODER ---
        # Up 1: Back to Level 2 resolution
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = DoubleConv(base_channels * 4, base_channels * 2) # base_channels*2 from up + base_channels*2 from enc2
        
        # Up 2: Back to Level 1 resolution
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = DoubleConv(base_channels * 2, base_channels) # base_channels from up + base_channels from enc1
        
        # --- SUBSURFACE STRATIFICATION HEAD ---
        # Maps multi-scale decoded representations to 15 vertical ocean depth levels
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        
        # Encoder passes with skip connections
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)
        
        # Bottleneck with multi-scale dilation
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        if u1.shape[-2] != e2.shape[-2] or u1.shape[-1] != e2.shape[-1]:
            u1 = F.interpolate(u1, size=(e2.shape[-2], e2.shape[-1]), mode='bilinear', align_corners=False)
        m1 = torch.cat([u1, e2], dim=1)
        c1 = self.dec1(m1)
        
        u2 = self.up2(c1)
        if u2.shape[-2] != e1.shape[-2] or u2.shape[-1] != e1.shape[-1]:
            u2 = F.interpolate(u2, size=(e1.shape[-2], e1.shape[-1]), mode='bilinear', align_corners=False)
        m2 = torch.cat([u2, e1], dim=1)
        c2 = self.dec2(m2)
        
        out = self.head(c2)
        
        # Match exact spatial target shape
        if out.shape[-2] != orig_h or out.shape[-1] != orig_w:
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            
        return out


class OceanEmbedViT(nn.Module):
    """
    Patch-based Vision Transformer for Subsurface Ocean Temperature Reconstruction.
    Tokenizes spatial fields into patch representations and applies Multi-Head Self-Attention.
    """
    def __init__(self, in_channels=5, out_channels=15, embed_dim=64, num_heads=4, num_layers=2):
        super(OceanEmbedViT, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.05, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_channels, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        patches = self.patch_embed(x)
        B, C, H, W = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        trans_tokens = self.transformer(tokens)
        feat = trans_tokens.transpose(1, 2).view(B, C, H, W)
        out = self.decoder(feat)
        if out.shape[-2] != orig_h or out.shape[-1] != orig_w:
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return out
