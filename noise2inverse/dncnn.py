import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=20):
        # From zhang-2017-beyon-gauss-denois:
        #
        #  Thus, for Gaussian denoising with a certain noise level, we
        #  set the receptive field size of DnCNN to 35 × 35 with the
        #  corresponding depth of 17. For other general image denoising
        #  tasks, we adopt a larger receptive field and set the depth
        #  to be 20.

        # Hence, we set the standard depth to 20.

        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(
            in_channels=channels,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
