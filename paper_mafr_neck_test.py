import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(
    0,
    r"C:\Users\thavi\Downloads\Drone_Bird_Detection"
)

from custom_modules.lcnet import lcnet_075

sys.path.insert(
    0,
    r"C:\Users\thavi\Downloads\Drone_Bird_Detection\LMWP-YOLO-full"
)

from mafr_test import MAFR


class PaperStyleMAFRNeck(nn.Module):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # Input channels from LCBackbone
        #
        # P2 = 48
        # P3 = 96
        # P4 = 192
        # P5 = 384
        # --------------------------------------------------

        # P5 -> 256
        self.p5_proj = nn.Conv2d(384, 256, 1)

        # P4 -> 256
        self.p4_proj = nn.Conv2d(192, 256, 1)

        # P3 -> 128
        self.p3_proj = nn.Conv2d(96, 128, 1)

        # P2 -> 128
        self.p2_proj = nn.Conv2d(48, 128, 1)

        # --------------------------------------------------
        # Top-down path
        # --------------------------------------------------

        # P5 -> P4
        self.p5_to_p4 = nn.Conv2d(256, 256, 3, padding=1)

        # P4 fusion
        self.p4_fusion = nn.Conv2d(
            256 + 256,
            256,
            1
        )

        # P4 -> P3
        self.p4_to_p3 = nn.Conv2d(
            256,
            128,
            3,
            padding=1
        )

        # P3 fusion
        self.p3_fusion = nn.Conv2d(
            128 + 128,
            128,
            1
        )

        # --------------------------------------------------
        # P3 -> P2
        # --------------------------------------------------

        self.p3_to_p2 = nn.Conv2d(
            128,
            128,
            3,
            padding=1
        )

        # P2 fusion
        self.p2_fusion = nn.Conv2d(
            128 + 128,
            128,
            1
        )

        # --------------------------------------------------
        # Bottom-up refinement
        # --------------------------------------------------

        # P2 -> P3
        self.p2_down = nn.Conv2d(
            128,
            256,
            3,
            stride=2,
            padding=1
        )

        # P3 refinement
        self.p3_refine = nn.Conv2d(
            128 + 256,
            256,
            1
        )

        # --------------------------------------------------
        # MAFR
        #
        # Published YAML explicitly specifies MAFR [256]
        # --------------------------------------------------

        self.mafr = MAFR(256)

    def forward(self, p2, p3, p4, p5):

        # --------------------------------------------------
        # Initial projections
        # --------------------------------------------------

        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5 = self.p5_proj(p5)

        # --------------------------------------------------
        # P5 -> P4
        # --------------------------------------------------

        p5_up = F.interpolate(
            self.p5_to_p4(p5),
            size=p4.shape[-2:],
            mode="nearest"
        )

        p4_out = self.p4_fusion(
            torch.cat(
                [p4, p5_up],
                dim=1
            )
        )

        # --------------------------------------------------
        # P4 -> P3
        # --------------------------------------------------

        p4_up = F.interpolate(
            self.p4_to_p3(p4_out),
            size=p3.shape[-2:],
            mode="nearest"
        )

        p3_out = self.p3_fusion(
            torch.cat(
                [p3, p4_up],
                dim=1
            )
        )

        # --------------------------------------------------
        # P3 -> P2
        # --------------------------------------------------

        p3_up = F.interpolate(
            self.p3_to_p2(p3_out),
            size=p2.shape[-2:],
            mode="nearest"
        )

        p2_out = self.p2_fusion(
            torch.cat(
                [p2, p3_up],
                dim=1
            )
        )

        # --------------------------------------------------
        # P2 -> P3
        # --------------------------------------------------

        p2_down = self.p2_down(p2_out)

        p3_final = self.p3_refine(
            torch.cat(
                [p3_out, p2_down],
                dim=1
            )
        )

        # --------------------------------------------------
        # MAFR
        # --------------------------------------------------

        mafr_out = self.mafr(p3_final)

        return (
            p2_out,
            p3_final,
            p4_out,
            mafr_out
        )


def main():

    print("=" * 65)
    print("PAPER-STYLE P2-P5 MAFR NECK TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    backbone = lcnet_075().to(device)
    neck = PaperStyleMAFRNeck().to(device)

    backbone.eval()
    neck.eval()

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device
    )

    with torch.no_grad():

        # Backbone
        p2, p3, p4, p5 = backbone(x)

        print("\nBackbone:")
        print("P2:", p2.shape)
        print("P3:", p3.shape)
        print("P4:", p4.shape)
        print("P5:", p5.shape)

        # Neck
        n2, n3, n4, mafr = neck(
            p2,
            p3,
            p4,
            p5
        )

        print("\nNeck outputs:")
        print("N2 :", n2.shape)
        print("N3 :", n3.shape)
        print("N4 :", n4.shape)
        print("MAFR:", mafr.shape)

    backbone_params = sum(
        p.numel()
        for p in backbone.parameters()
    )

    neck_params = sum(
        p.numel()
        for p in neck.parameters()
    )

    print("\nParameters:")
    print("Backbone :", f"{backbone_params:,}")
    print("Neck     :", f"{neck_params:,}")
    print(
        "Total    :",
        f"{backbone_params + neck_params:,}"
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()