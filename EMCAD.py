import timm
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


# Dataset class
class MySegmentationDataset(Dataset):
    def __init__(self, data_images, data_masks, transform=None):
        self.images = data_images  # Now these are PIL Images or tensors
        self.masks = data_masks
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # If they're PIL Images, you can directly transform them
        item_image = self.transform(self.images[idx])
        item_mask = (self.transform(self.masks[idx]) > 0).float()  # Binarize mask

        return item_image, item_mask


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(in_channels, reduced_channels, 1, bias=False), nn.ReLU(inplace=True),
                                        nn.Conv2d(reduced_channels, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttentionBlock(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, avg_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale


class MultiScaleDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.in_channels = in_channels
        self.branches = nn.ModuleList()

        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True)))

    def forward(self, x):
        for branch in self.branches:
            assert branch[1].num_features == x.shape[1], f"Expected {branch[1].num_features} channels, got {x.shape[1]}"
        out = sum(branch(x) for branch in self.branches)
        return out


class MSCB(nn.Module):
    def __init__(self, in_channels, expansion_factor=2, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.in_channels = in_channels
        expanded_channels = in_channels * expansion_factor

        self.expand = nn.Sequential(nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                                    nn.BatchNorm2d(expanded_channels), nn.ReLU6(inplace=True))

        self.msd_conv = MultiScaleDepthwiseConv(expanded_channels, kernel_sizes)

        self.project = nn.Sequential(nn.Conv2d(expanded_channels, in_channels, 1, bias=False),
                                     nn.BatchNorm2d(in_channels))

    def forward(self, x):
        return self.project(self.msd_conv(self.expand(x)))


class MSCAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cab = ChannelAttentionBlock(in_channels)
        self.sab = SpatialAttentionBlock()
        self.mscb = MSCB(in_channels)

    def forward(self, x):
        return self.mscb(self.sab(self.cab(x)))


class LGAG(nn.Module):
    def __init__(self, in_channels, groups=4):
        super().__init__()
        self.gc_g = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=groups, bias=False)
        self.gc_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=groups, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)  # <--- NEW batch-norm for 1 channel
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g_bn = self.bn(self.gc_g(g))
        x_bn = self.bn(self.gc_x(x))
        attn = self.relu(g_bn + x_bn)
        attn = self.bn1(self.conv1(attn))  # <--- use bn1
        attn = self.sigmoid(attn)
        return x * attn


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=1),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 1))

    def forward(self, x):
        return self.block(x)


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EMCADDecoder(nn.Module):
    def __init__(self, channels, out_channels=1):
        super().__init__()
        self.mscams = nn.ModuleList([MSCAM(c) for c in channels])
        self.seg_heads = nn.ModuleList([SegHead(c, out_channels) for c in channels])
        self.eucbs = nn.ModuleList([EUCB(channels[i], channels[i - 1]) for i in range(len(channels) - 1, 0, -1)])
        self.lgags = nn.ModuleList([LGAG(channels[i - 1]) for i in range(len(channels) - 1, 0, -1)])

        self.final_seghead = SegHead(channels[0], out_channels)

    def forward(self, feats):
        feats = [msc(f) for msc, f in zip(self.mscams, feats)]
        predictions = [seg(f) for seg, f in zip(self.seg_heads, feats)]
        x = feats[-1]

        for i in reversed(range(3)):
            x_up = self.eucbs[2 - i](x)
            x = self.lgags[2 - i](feats[i], x_up) + x_up

        # Dynamically match the target size to the first feature map
        x = self.final_seghead(x)
        return predictions, x


class PVTEMCAD(nn.Module):
    def __init__(self, encoder_name='pvt_v2_b2', out_channels=1):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        # Get the actual channel sizes from the encoder
        self.decoder = EMCADDecoder(channels=self.encoder.feature_info.channels(), out_channels=out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))[1]
