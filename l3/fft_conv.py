"""
FFTConv2d — drop-in replacement for nn.Conv2d that performs convolution
via element-wise multiplication in the DFT domain.

The filter weights are registered as a non-trainable buffer so they are
frozen from the moment the layer is created (matches the lab requirement:
"filter bank that does not change during training").
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _next_power_of_two(n: int) -> int:
    return 1 << math.ceil(math.log2(n))


class FFTConv2d(nn.Module):
    """
    Convolution via FFT.  Equivalent to nn.Conv2d with padding='same'
    (zero-padded to avoid circular artefacts) but uses rfft2/irfft2.

    Parameters mirror nn.Conv2d; bias is supported.
    The weight tensor is a *buffer* (frozen), not a parameter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        # Weight as buffer → frozen (no gradient)
        weight = torch.empty(out_channels, in_channels // groups, *kernel_size)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.register_buffer("weight", weight)

        if bias:
            # Bias IS trainable
            fan_in = (in_channels // groups) * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-bound, bound))
        else:
            self.bias = None

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d) -> "FFTConv2d":
        """Copy weights from an existing nn.Conv2d into a frozen FFTConv2d."""
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            layer.weight.copy_(conv.weight.data)
            if conv.bias is not None and layer.bias is not None:
                layer.bias.data.copy_(conv.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        g = self.groups

        # Padding to avoid circular convolution artefacts
        pad_h = kH - 1
        pad_w = kW - 1
        x_pad = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                           pad_h // 2, pad_h - pad_h // 2))

        fft_h = _next_power_of_two(x_pad.shape[-2])
        fft_w = _next_power_of_two(x_pad.shape[-1])

        # x_pad: (B, C, H', W')
        X = torch.fft.rfft2(x_pad, s=(fft_h, fft_w))  # (B, C, fft_h, fft_w//2+1)

        # weight: (out_ch, in_ch/g, kH, kW)
        # pad kernel to fft size
        W_freq = torch.fft.rfft2(self.weight, s=(fft_h, fft_w))  # (out_ch, in_ch/g, ...)

        # Group convolution in frequency domain
        out_chunks = []
        in_per_g = C // g
        out_per_g = self.out_channels // g
        for grp in range(g):
            x_grp = X[:, grp * in_per_g:(grp + 1) * in_per_g]   # (B, in_per_g, fH, fW)
            w_grp = W_freq[grp * out_per_g:(grp + 1) * out_per_g]  # (out_per_g, in_per_g, fH, fW)
            # einsum: b i h w, o i h w -> b o h w
            y_grp = torch.einsum("bihw,oihw->bohw", x_grp, w_grp)
            out_chunks.append(y_grp)
        Y = torch.cat(out_chunks, dim=1)  # (B, out_ch, fH, fW)

        out = torch.fft.irfft2(Y, s=(fft_h, fft_w))  # (B, out_ch, fft_h, fft_w)
        out = out[:, :, :H, :W]  # crop back

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out