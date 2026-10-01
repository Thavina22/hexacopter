import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\thavi\Downloads\Drone_Bird_Detection")

from custom_modules.lcnet import lcnet_075

sys.path.insert(
    0,
    r"C:\Users\thavi\Downloads\Drone_Bird_Detection\LMWP-YOLO-full"
)

from mafr_test import MAFR

from ultralytics.nn.modules.head import Detect


class P2P5MAFRNeck(nn.Module):
    def __init__(self):
        super().__init__()

        self.p2_proj = nn.Conv2d(48, 128, 1)
        self.p3_proj = nn.Conv2d(96, 128, 1)
        self.p4_proj = nn.Conv2d(192, 256, 1)
        self.p5_proj = nn.Conv2d(384, 256, 1)

        self.p5_to_p4 = nn.Conv2d(256, 256, 1)
        self.p4_to_p3 = nn.Conv2d(256, 128, 1)
        self.p3_to_p2 = nn.Conv2d(128, 128, 1)

        self.mafr = MAFR(128)

    def forward(self, p2, p3, p4, p5):

        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5 = self.p5_proj(p5)

        p5_up = F.interpolate(
            self.p5_to_p4(p5),
            size=p4.shape[-2:],
            mode="nearest"
        )
        p4_out = p4 + p5_up

        p4_up = F.interpolate(
            self.p4_to_p3(p4_out),
            size=p3.shape[-2:],
            mode="nearest"
        )
        p3_out = p3 + p4_up

        p3_up = F.interpolate(
            self.p3_to_p2(p3_out),
            size=p2.shape[-2:],
            mode="nearest"
        )
        p2_out = p2 + p3_up

        p2_out = self.mafr(p2_out)

        return p2_out, p3_out, p4_out, p5


class LMWPTestModel(nn.Module):

    def __init__(self, nc=1):
        super().__init__()

        self.backbone = lcnet_075()
        self.neck = P2P5MAFRNeck()

        # P2, P3, P4, P5 channels
        self.detect = Detect(
            nc=nc,
            ch=(128, 128, 256, 256)
        )

    def forward(self, x):

        p2, p3, p4, p5 = self.backbone(x)

        n2, n3, n4, n5 = self.neck(
            p2, p3, p4, p5
        )

        outputs = self.detect(
            [n2, n3, n4, n5]
        )

        return outputs


def main():

    print("=" * 65)
    print("LMWP-YOLO P2-P5 + MAFR + YOLO DETECT TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device :", device)

    model = LMWPTestModel(nc=1).to(device)

    model.eval()

    x = torch.randn(
        1, 3, 640, 640,
        device=device
    )

    with torch.no_grad():

        outputs = model(x)

    print("\nDetection output:")

    if isinstance(outputs, tuple):

        print("Output type : tuple")
        print("Tuple length:", len(outputs))

        for i, output in enumerate(outputs):

            if isinstance(output, list):

                print(
                    f"Output[{i}]: list "
                    f"(length={len(output)})"
                )

                for j, item in enumerate(output):

                    print(
                        f"  [{j}] {item.shape}"
                    )

            elif torch.is_tensor(output):

                print(
                    f"Output[{i}]: {output.shape}"
                )

            else:

                print(
                    f"Output[{i}]: {type(output)}"
                )

    elif isinstance(outputs, list):

        print("Output type : list")
        print("Length      :", len(outputs))

        for i, output in enumerate(outputs):

            print(
                f"Output[{i}]: {output.shape}"
            )

    elif torch.is_tensor(outputs):

        print("Output type:", outputs.shape)

    else:

        print("Output type:", type(outputs))

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:")
    print(f"{total_params:,}")

    print("\nCUDA memory allocated:")

    if torch.cuda.is_available():

        print(
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()