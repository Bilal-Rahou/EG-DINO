import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class GatedFusionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Conv2d(1, in_channels, kernel_size=1) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, edge):
        edge_resized = F.interpolate(edge, size=x.shape[2:], mode='bilinear', align_corners=False)
        attention = self.sigmoid(self.edge_conv(edge_resized))
        return x + (x * attention)

class ParallelEdgePyramidDecoder(nn.Module):
    def __init__(self, encoder_dim, num_classes, target_h, target_w):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        
        self.grid_h = target_h // 14
        self.grid_w = target_w // 14
        
        self.conv1 = nn.Sequential(nn.Conv2d(encoder_dim, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU())
        self.gate1 = GatedFusionModule(256)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU())
        self.gate2 = GatedFusionModule(128)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU())
        self.gate3 = GatedFusionModule(64)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.GELU())
        self.gate4 = GatedFusionModule(32)
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, edge_map):
        h1, w1 = self.grid_h * 2, self.grid_w * 2
        x = self.conv1(x)
        x = F.interpolate(x, size=(h1, w1), mode='bilinear', align_corners=False)
        x = self.gate1(x, edge_map)
        
        h2, w2 = self.grid_h * 4, self.grid_w * 4
        x = self.conv2(x)
        x = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=False)
        x = self.gate2(x, edge_map)
        
        h3, w3 = self.grid_h * 8, self.grid_w * 8
        x = self.conv3(x)
        x = F.interpolate(x, size=(h3, w3), mode='bilinear', align_corners=False)
        x = self.gate3(x, edge_map)
        
        x = self.conv4(x)
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False) 
        x = self.gate4(x, edge_map)
        
        return self.final_conv(x)

class DINOv2EdgeAware(nn.Module):
    def __init__(self, model_name, num_classes, img_height, img_width):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size 
        self.decoder = ParallelEdgePyramidDecoder(self.encoder_dim, num_classes, img_height, img_width)
        self.patch_size = 14

    def forward(self, x, edge):
        outputs = self.encoder(x)
        patch_tokens = outputs.last_hidden_state[:, 1:] 
        
        B, L, D = patch_tokens.shape
        H_grid = self.img_height // self.patch_size
        W_grid = self.img_width // self.patch_size
        
        feature_map = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H_grid, W_grid)
        return self.decoder(feature_map, edge)