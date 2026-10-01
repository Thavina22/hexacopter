import sys
import torch
import torch.nn as nn

# Add project root
sys.path.insert(0, r"C:\Users\thavi\Downloads\Drone_Bird_Detection")

from custom_modules.lcnet import lcnet_075

# Import MAFR from your existing implementation
sys.path.insert(
    0,
    r"C:\Users\thavi\Downloads\Drone_Bird_Detection\LMWP-YOLO-full"
)

from mafr_test import MAFR


class P2P5MAFRNeck(nn.Module):
    def __init__(self):
        super().__init__()

        # LCNet channels
        self.p2_proj = nn.Conv2d(48, 128, 1)
        self.p3_proj = nn.Conv2d(96, 128, 1)
        self.p4_proj = nn.Conv2d(192, 256, 1)
        self.p5_proj = nn.Conv2d(384, 256, 1)

        # Top-down fusion
        self.p5_to_p4 = nn.Conv2d(256, 256, 1)
        self.p4_to_p3 = nn.Conv2d(256, 128, 1)
        self.p3_to_p2 = nn.Conv2d(128, 128, 1)

        # MAFR at high-resolution P2 feature
        self.mafr = MAFR(128)

    def forward(self, p2, p3, p4, p5):

        # Project channels
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5 = self.p5_proj(p5)

        # P5 -> P4
        p5_up = nn.functional.interpolate(
            self.p5_to_p4(p5),
            size=p4.shape[-2:],
            mode="nearest"
        )
        p4_out = p4 + p5_up

        # P4 -> P3
        p4_up = nn.functional.interpolate(
            self.p4_to_p3(p4_out),
            size=p3.shape[-2:],
            mode="nearest"
        )
        p3_out = p3 + p4_up

        # P3 -> P2
        p3_up = nn.functional.interpolate(
            self.p3_to_p2(p3_out),
            size=p2.shape[-2:],
            mode="nearest"
        )
        p2_out = p2 + p3_up

        # MAFR on high-resolution feature
        p2_out = self.mafr(p2_out)

        return p2_out, p3_out, p4_out, p5


def main():

    print("=" * 60)
    print("LCBACKBONE + P2-P5 + MAFR TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device :", device)

    # Backbone
    backbone = lcnet_075().to(device)

    # Neck
    neck = P2P5MAFRNeck().to(device)

    # Evaluation mode
    backbone.eval()
    neck.eval()

    # Dummy image
    x = torch.randn(1, 3, 640, 640).to(device)

    with torch.no_grad():

        # Backbone
        p2, p3, p4, p5 = backbone(x)

        print("\nBackbone:")
        print("P2:", p2.shape)
        print("P3:", p3.shape)
        print("P4:", p4.shape)
        print("P5:", p5.shape)

        # Neck
        n2, n3, n4, n5 = neck(p2, p3, p4, p5)

        print("\nNeck outputs:")
        print("N2:", n2.shape)
        print("N3:", n3.shape)
        print("N4:", n4.shape)
        print("N5:", n5.shape)

    # Parameter count
    backbone_params = sum(
        p.numel() for p in backbone.parameters()
    )

    neck_params = sum(
        p.numel() for p in neck.parameters()
    )

    print("\nParameters:")
    print("Backbone:", f"{backbone_params:,}")
    print("Neck    :", f"{neck_params:,}")
    print("Total   :", f"{backbone_params + neck_params:,}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()