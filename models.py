"""SRUG: a CRRB feature pyramid and Nested Multi-Scale Decoder (NMD).

The NMD borrows nested skip connectivity from U-Net++; SRUG does not instantiate
the original segmentation network. Module registration keys are retained so
existing state_dict checkpoints load without renaming weights.
"""

import torch
import torch.nn as nn


ENCODER_TYPES = ("crrb", "without_se")
DECODER_TYPES = ("nmd", "plain")


def normalize_encoder_type(value):
    name = str(value).lower()
    name = {"seresnet": "crrb", "resnet": "without_se"}.get(name, name)
    if name not in ENCODER_TYPES:
        raise ValueError(f"Unknown encoder_type: {value}")
    return name


def normalize_decoder_type(value):
    name = str(value).lower()
    name = {"unetpp": "nmd", "unet": "plain"}.get(name, name)
    if name not in DECODER_TYPES:
        raise ValueError(f"Unknown decoder_type: {value}")
    return name


class FeatureFusionBlock(nn.Module):
    """Two convolutions used in the full-resolution stem and fusion nodes."""
    def __init__(self, in_channels, middle_channels, out_channels, use_attention=False):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = DecoderChannelAttention(out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return self.attention(out)


class DecoderChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))

class CRRB(nn.Module):
    """Channel-Recalibrated Residual Block with SE inside the residual branch."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, r=16):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, stride=stride,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * CRRB.expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * CRRB.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * CRRB.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * CRRB.expansion,
                          stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * CRRB.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * CRRB.expansion,
                      out_channels * CRRB.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * CRRB.expansion // r,
                      out_channels * CRRB.expansion),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_function(x)

        bs, c, _, _ = residual.size()
        y = self.squeeze(residual).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        residual = residual * y.expand_as(residual)

        out = residual + self.shortcut(x)
        return self.relu(out)


class ResidualBlock(nn.Module):
    """Matched residual block without SE, used for the encoder ablation."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResidualBlock.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * ResidualBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * ResidualBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * ResidualBlock.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * ResidualBlock.expansion),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual_function(x) + self.shortcut(x)
        return self.relu(out)


class PyramidEncoder(nn.Module):
    """Full-resolution stem followed by four pooling-separated residual stages."""

    def __init__(self, block, layers, input_channels=3, freeze_swin=True):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        self.in_channels = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = FeatureFusionBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)


    def _make_layer(self, block, middle_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, middle_channels, stride))
            self.in_channels = middle_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        return [x0_0, x1_0, x2_0, x3_0, x4_0]



def crrb_encoder():
    return PyramidEncoder(block=CRRB, layers=[3, 4, 6, 3])


def encoder_without_se():
    return PyramidEncoder(block=ResidualBlock, layers=[3, 4, 6, 3])


def encoder_block_for(encoder_type):
    encoder_type = normalize_encoder_type(encoder_type)
    if encoder_type == "crrb":
        return CRRB
    if encoder_type == "without_se":
        return ResidualBlock
    raise ValueError(f"Unknown encoder_type: {encoder_type}")


def build_encoder(encoder_type):
    encoder_type = normalize_encoder_type(encoder_type)
    if encoder_type == "crrb":
        return crrb_encoder()
    if encoder_type == "without_se":
        return encoder_without_se()
    raise ValueError(f"Unknown encoder_type: {encoder_type}")


class SRUG(nn.Module):
    """Direct MRI synthesis with a CRRB encoder and NMD by default.

    NMD fusion nodes remain registered at the generator's top level to preserve
    checkpoint keys. ``plain`` selects the sequential decoder ablation.
    """

    def __init__(
        self,
        block=CRRB,
        num_classes=3,
        input_channels=3,
        deep_supervision=False,
        decoder_attention=False,
        encoder_type="crrb",
        decoder_type="nmd",
    ):
        super().__init__()

        encoder_type = normalize_encoder_type(encoder_type)
        decoder_type = normalize_decoder_type(decoder_type)
        if decoder_type not in ("plain", "nmd"):
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        if decoder_type == "nmd":
            decoder_is_nested = True
        else:
            decoder_is_nested = False
        if deep_supervision and decoder_type == "plain":
            raise ValueError("deep_supervision is only supported for decoder_type='nmd'")
        block = encoder_block_for(encoder_type)

        nb_filter = [64, 128, 256, 512, 1024]
        self.in_channels = nb_filter[0]
        self.relu = nn.ReLU()
        # The historical 'se' state_dict prefix is part of the checkpoint format.
        self.se = build_encoder(encoder_type)
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.model_config = {
            "encoder_type": encoder_type,
            "decoder_type": decoder_type,
            "decoder_attention": decoder_attention,
            "decoder_is_nested": decoder_is_nested,
        }
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = FeatureFusionBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv0_1 = FeatureFusionBlock(
            nb_filter[0] + nb_filter[1] * block.expansion,
            nb_filter[0],
            nb_filter[0],
            use_attention=decoder_attention,
        )
        self.conv1_1 = FeatureFusionBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion, use_attention=decoder_attention)
        self.conv2_1 = FeatureFusionBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion, use_attention=decoder_attention)
        self.conv3_1 = FeatureFusionBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion, use_attention=decoder_attention)
        self.conv0_2 = FeatureFusionBlock(
            nb_filter[0] * 2 + nb_filter[1] * block.expansion,
            nb_filter[0],
            nb_filter[0],
            use_attention=decoder_attention,
        )
        self.conv1_2 = FeatureFusionBlock((nb_filter[1] * 2 + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion, use_attention=decoder_attention)

        self.conv2_2 = FeatureFusionBlock((nb_filter[2] * 2 + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion, use_attention=decoder_attention)

        self.conv0_3 = FeatureFusionBlock(
            nb_filter[0] * 3 + nb_filter[1] * block.expansion,
            nb_filter[0],
            nb_filter[0],
            use_attention=decoder_attention,
        )
        self.conv1_3 = FeatureFusionBlock((nb_filter[1] * 3 + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion, use_attention=decoder_attention)
        self.conv0_4 = FeatureFusionBlock(
            nb_filter[0] * 4 + nb_filter[1] * block.expansion,
            nb_filter[0],
            nb_filter[0],
            use_attention=decoder_attention,
        )
        if decoder_type == "plain":
            # Keep the historical parameter keys for plain-decoder checkpoints.
            self.unet_conv3_1 = FeatureFusionBlock(
                (nb_filter[3] + nb_filter[4]) * block.expansion,
                nb_filter[3],
                nb_filter[3] * block.expansion,
                use_attention=decoder_attention,
            )
            self.unet_conv2_2 = FeatureFusionBlock(
                (nb_filter[2] + nb_filter[3]) * block.expansion,
                nb_filter[2],
                nb_filter[2] * block.expansion,
                use_attention=decoder_attention,
            )
            self.unet_conv1_3 = FeatureFusionBlock(
                (nb_filter[1] + nb_filter[2]) * block.expansion,
                nb_filter[1],
                nb_filter[1] * block.expansion,
                use_attention=decoder_attention,
            )
            self.unet_conv0_4 = FeatureFusionBlock(
                nb_filter[0] + nb_filter[1] * block.expansion,
                nb_filter[0],
                nb_filter[0],
                use_attention=decoder_attention,
            )
            for module in (
                self.conv0_1,
                self.conv1_1,
                self.conv2_1,
                self.conv3_1,
                self.conv0_2,
                self.conv1_2,
                self.conv2_2,
                self.conv0_3,
                self.conv1_3,
                self.conv0_4,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, middle_channels, stride))
            self.in_channels = middle_channels * block.expansion
        return nn.Sequential(*layers)

    @property
    def encoder(self):
        """Expose the encoder without registering duplicate checkpoint keys."""
        return self.se

    def forward_nmd(self, input):
        """Fuse the feature pyramid through the nested multi-scale pathway."""

        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(input)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))


        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output

    def forward_plain(self, input):
        """Sequential decoder with one encoder skip at each reconstruction stage."""
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(input)
        x3_1 = self.unet_conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.unet_conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.unet_conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.unet_conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        return self.final(x0_4)

    def forward(self, input):
        if self.decoder_type == "plain":
            return self.forward_plain(input)
        if self.decoder_type == "nmd":
            return self.forward_nmd(input)
        raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

    # Compatibility for callers that used the previous method names.
    forward_unetpp = forward_nmd
    forward_unet = forward_plain


# Legacy imports remain valid; new code should use the manuscript terminology.
NestedUResnet = SRUG
SEBottleNeck = CRRB
BottleNeck = ResidualBlock
VGGBlock = FeatureFusionBlock
CNN_Encoder = PyramidEncoder
seresnet50_encoder = crrb_encoder
resnet50_encoder = encoder_without_se
