"""Shared residual conv building blocks (used by the query pair aggregator)."""

import torch.nn as nn


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class ResNet(nn.Module):
    def __init__(self, layer_num, input_channel=None, use_bn=False, output_channel=None, kernel_size=3, use_dims=None):
        super().__init__()

        self.layer_num = layer_num
        self.output_channel = output_channel
        self.use_bn = use_bn
        self.kernel_size = kernel_size
        self.use_dims = use_dims
        self.input_channel = len(self.use_dims) if self.use_dims is not None else input_channel

        layers = []
        if self.output_channel is None:
            for idx in range(self.layer_num):
                layers.append(ResidualConvUnit(features=self.input_channel, activation=nn.ReLU(), bn=self.use_bn))
        else:
            if isinstance(self.output_channel, int):
                layers.append(
                    nn.Conv2d(
                        self.input_channel,
                        self.output_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=True,
                        groups=1,
                    )
                )
                for idx in range(self.layer_num):
                    layers.append(ResidualConvUnit(features=self.output_channel, activation=nn.ReLU(), bn=self.use_bn))
            else:
                input_channel = self.input_channel
                for output_channel in self.output_channel:
                    layers.append(
                        nn.Conv2d(
                            input_channel,
                            output_channel,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2,
                            bias=True,
                            groups=1,
                        )
                    )
                    for idx in range(self.layer_num):
                        layers.append(ResidualConvUnit(features=output_channel, activation=nn.ReLU(), bn=self.use_bn))
                    input_channel = output_channel

        assert len(layers) > 0
        self.convs = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if self.use_dims is not None:
            x = x[:, self.use_dims]
        return self.convs(x)
