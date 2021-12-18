from torch import nn


class StridedConv(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride):
        super(StridedConv, self).__init__()
        # kernel should be an odd number and stride an even number
        self.conv = nn.Sequential(
            nn.ReflectionPad1d(kernel_size // 2),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # input and output of shape [bs,in_channels,L] --> [bs,out_channels,L//stride]
        return self.conv(x)


class ResidualConv(nn.Module):
    def __init__(self, channels, n_blocks=3):
        super(ResidualConv, self).__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad1d(3 ** i),
                    nn.Conv1d(channels, channels, kernel_size=3, dilation=3 ** i),
                    nn.BatchNorm1d(channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(channels, channels, kernel_size=1),
                    nn.BatchNorm1d(channels),
                )
                for i in range(n_blocks)
            ]
        )
        self.shortcuts = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.BatchNorm1d(channels))
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        # input and output of shape [bs,channels,L]
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, norm="BN"):
        super(LinearBlock, self).__init__()
        if norm == "BN":
            self.block = nn.Sequential(nn.Linear(in_size, out_size), nn.BatchNorm1d(out_size), nn.LeakyReLU(0.2))
        if norm == "LN":
            self.block = nn.Sequential(nn.Linear(in_size, out_size), nn.LayerNorm(out_size), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.block(x)
