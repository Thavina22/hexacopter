import torch
import torch.nn as nn


# ============================================================
# Utility
# ============================================================

def _make_divisible(v, divisor=8, min_value=None):
    """
    Make sure the number of channels is divisible by divisor.
    """
    if min_value is None:
        min_value = divisor

    new_v = max(
        min_value,
        int(v + divisor / 2) // divisor * divisor
    )

    # Prevent rounding down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


# ============================================================
# Hard Sigmoid
# ============================================================

class HardSigmoid(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(
            x + 3.0,
            0.0,
            6.0
        ) / 6.0


# ============================================================
# Hard Swish
# ============================================================

class HardSwish(nn.Module):

    def __init__(self):
        super().__init__()

        self.hardsigmoid = HardSigmoid()

    def forward(self, x):

        return x * self.hardsigmoid(x)


# ============================================================
# Squeeze-and-Excitation
# ============================================================

class SELayer(nn.Module):

    def __init__(
        self,
        channel,
        reduction=4
    ):
        super().__init__()

        reduced_channels = max(
            1,
            channel // reduction
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channel,
                reduced_channels,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                reduced_channels,
                channel,
                kernel_size=1
            ),

            HardSigmoid()
        )

    def forward(self, x):

        y = self.avg_pool(x)

        y = self.fc(y)

        return x * y


# ============================================================
# Depthwise Separable Convolution
# ============================================================

class DepSepConv(nn.Module):

    def __init__(
        self,
        inp,
        oup,
        kernel_size,
        stride,
        use_se=False
    ):
        super().__init__()

        padding = kernel_size // 2

        # ----------------------------------------------------
        # Depthwise convolution
        # ----------------------------------------------------

        self.dw = nn.Conv2d(
            inp,
            inp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=inp,
            bias=False
        )

        self.dw_bn = nn.BatchNorm2d(inp)

        self.dw_act = HardSwish()

        # ----------------------------------------------------
        # SE attention
        # ----------------------------------------------------

        self.use_se = use_se

        if use_se:

            self.se = SELayer(
                inp,
                reduction=4
            )

        # ----------------------------------------------------
        # Pointwise convolution
        # ----------------------------------------------------

        self.pw = nn.Conv2d(
            inp,
            oup,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.pw_bn = nn.BatchNorm2d(oup)

        self.pw_act = HardSwish()

    def forward(self, x):

        x = self.dw(x)

        x = self.dw_bn(x)

        x = self.dw_act(x)

        if self.use_se:

            x = self.se(x)

        x = self.pw(x)

        x = self.pw_bn(x)

        x = self.pw_act(x)

        return x


# ============================================================
# LCNet Backbone
# ============================================================

class LCBackbone(nn.Module):

    def __init__(
        self,
        scale=0.75
    ):
        super().__init__()

        self.scale = scale

        # ----------------------------------------------------
        # PP-LCNet configuration
        #
        # [kernel, channels, stride, SE]
        # ----------------------------------------------------

        self.cfgs = [

            [3, 32, 1, 0],

            [3, 64, 2, 0],
            [3, 64, 1, 0],

            [3, 128, 2, 0],
            [3, 128, 1, 0],

            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],

            [5, 512, 2, 1],
            [5, 512, 1, 1],
        ]

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        input_channel = _make_divisible(
            16 * scale
        )

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                input_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                input_channel
            ),

            HardSwish()
        )

        # ----------------------------------------------------
        # Blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList()

        for (
            kernel_size,
            channels,
            stride,
            use_se
        ) in self.cfgs:

            output_channel = _make_divisible(
                channels * scale
            )

            self.blocks.append(

                DepSepConv(
                    inp=input_channel,
                    oup=output_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    use_se=bool(use_se)
                )
            )

            input_channel = output_channel

    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        x = self.stem(x)

        # ----------------------------------------------------
        # Feature outputs
        #
        # Block 01 → P2
        # Block 03 → P3
        # Block 10 → P4
        # Block 12 → P5
        # ----------------------------------------------------

        p2 = None
        p3 = None
        p4 = None
        p5 = None

        for i, layer in enumerate(self.blocks):

            x = layer(x)

            # P2
            if i == 1:

                p2 = x

            # P3
            elif i == 3:

                p3 = x

            # P4
            elif i == 10:

                p4 = x

            # P5
            elif i == 12:

                p5 = x

        return p2, p3, p4, p5


# ============================================================
# LCNet 0.75
# ============================================================

def lcnet_075():

    return LCBackbone(
        scale=0.75
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LCBACKBONE TEST")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    model = LCBackbone(
        scale=0.75
    ).to(device)

    model.eval()

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device
    )

    with torch.no_grad():

        p2, p3, p4, p5 = model(x)

    print("\nInput:")
    print(x.shape)

    print("\nBackbone outputs:")

    print("P2:", p2.shape)
    print("P3:", p3.shape)
    print("P4:", p4.shape)
    print("P5:", p5.shape)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nParameters:",
        f"{params:,}"
    )

    print("\nTest completed successfully!")