import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import torchvision.models as models

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)
    
class DeblurGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_res_blocks=6):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        model += [ResBlock(256) for _ in range(num_res_blocks)]
        model += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.clamp(x + self.model(x),min=-1, max=1)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_feat, out_feat, norm=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return self.loss(pred_features, target_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = DeblurGenerator().to(device)
discriminator = Discriminator().to(device)

adv_criterion = nn.MSELoss()
content_criterion = VGGPerceptualLoss().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
os.makedirs("checkpoints", exist_ok=True)