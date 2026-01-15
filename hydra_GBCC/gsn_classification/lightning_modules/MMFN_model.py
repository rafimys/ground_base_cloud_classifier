from torchvision.models import resnet50
import torch
from torch import nn, tensor

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        base = resnet50(weights="IMAGENET1K_V2" if pretrained else None)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

            for param in self.layer2.parameters():
                param.requires_grad = True

            for param in self.layer3.parameters():
                param.requires_grad = True

class AttentiveNetwork(nn.Module):

    def __init__(self, in_channels=512, out_dim=2048, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(in_channels // reduction, 8)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x2):
        b, c, _, _ = x2.shape
        v = self.pool(x2).view(b, c)
        att = self.mlp(v)
        return att

class MultiModalNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2048)
        )

    def forward(self, x):
        return self.net(x)

class MMFN(nn.Module):
    def __init__(self, num_classes, tabular_dim, ratio):
        super().__init__()

        self.backbone = ResNet50Backbone(pretrained=True)
        self.att_net = AttentiveNetwork()
        self.mm_net = MultiModalNetwork(tabular_dim)
        self.ratio = ratio
        self.fc = nn.Linear(2048 * 2, num_classes)

    def forward(self, image, tabular):
        # === Backbone ===
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)  # conv3_x (512, 28, 28)

        # === Attentive branch ===
        att_feat = self.att_net(x2)  # (B, 2048)

        # === Main branch ===
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        main_feat = self.backbone.avgpool(x4).flatten(1)  # (B, 2048)

        # === Fusion ===
        fused_visual = main_feat * att_feat
        mm_feat = self.mm_net(tabular)

        fused = torch.cat([self.ratio*fused_visual, (1-self.ratio)*mm_feat], dim=1)

        return self.fc(fused)