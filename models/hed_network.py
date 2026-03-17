import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1); self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1); self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1); self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1); self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1); self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1); self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1); self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1); self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.dsn1 = nn.Conv2d(64, 1, 1); self.dsn2 = nn.Conv2d(128, 1, 1); self.dsn3 = nn.Conv2d(256, 1, 1); self.dsn4 = nn.Conv2d(512, 1, 1); self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1_1(x)); x1 = F.relu(self.conv1_2(x1))
        x2 = F.max_pool2d(x1, 2, stride=2); x2 = F.relu(self.conv2_1(x2)); x2 = F.relu(self.conv2_2(x2))
        x3 = F.max_pool2d(x2, 2, stride=2); x3 = F.relu(self.conv3_1(x3)); x3 = F.relu(self.conv3_2(x3)); x3 = F.relu(self.conv3_3(x3))
        x4 = F.max_pool2d(x3, 2, stride=2); x4 = F.relu(self.conv4_1(x4)); x4 = F.relu(self.conv4_2(x4)); x4 = F.relu(self.conv4_3(x4))
        x5 = F.max_pool2d(x4, 2, stride=2); x5 = F.relu(self.conv5_1(x5)); x5 = F.relu(self.conv5_2(x5)); x5 = F.relu(self.conv5_3(x5))
        d1 = self.dsn1(x1); d2 = F.interpolate(self.dsn2(x2), size=x.shape[2:], mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.dsn3(x3), size=x.shape[2:], mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.dsn4(x4), size=x.shape[2:], mode='bilinear', align_corners=True)
        d5 = F.interpolate(self.dsn5(x5), size=x.shape[2:], mode='bilinear', align_corners=True)
        return torch.sigmoid(self.fuse(torch.cat((d1, d2, d3, d4, d5), 1)))

def load_pretrained_hed(model_path='./weights/hed_pretrained.pth'):
    model = HED()
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve("http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch", model_path)
    state = torch.load(model_path, map_location="cpu")
    new_state = {}
    mapping = {
        "moduleVggOne.0": "conv1_1", "moduleVggOne.2": "conv1_2", "moduleVggTwo.1": "conv2_1", "moduleVggTwo.3": "conv2_2",
        "moduleVggThr.1": "conv3_1", "moduleVggThr.3": "conv3_2", "moduleVggThr.5": "conv3_3",
        "moduleVggFou.1": "conv4_1", "moduleVggFou.3": "conv4_2", "moduleVggFou.5": "conv4_3",
        "moduleVggFiv.1": "conv5_1", "moduleVggFiv.3": "conv5_2", "moduleVggFiv.5": "conv5_3",
        "moduleScoreOne": "dsn1", "moduleScoreTwo": "dsn2", "moduleScoreThr": "dsn3", "moduleScoreFou": "dsn4", "moduleScoreFiv": "dsn5", "moduleCombine.0": "fuse",
    }
    for k, v in mapping.items():
        if k + ".weight" in state:
            new_state[v + ".weight"] = state[k + ".weight"]
            new_state[v + ".bias"] = state[k + ".bias"]
    model.load_state_dict(new_state)
    return model