import torch

from torch import nn


def l2normed(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()

        self.module = module
        self.name = name
        self.power_iterations = power_iterations

        self.height = None
        self.width = None

        self.u_inited = False

    def _update_u_v(self):
        if not self.u_inited:
            self._make_params()

        w = getattr(self.module, self.name)

        height = w.data.shape[0]

        if self.conv_f:
            with torch.no_grad():
                self.conv = nn.Conv2d(in_channels=self.module.in_channels,
                                      out_channels=self.module.out_channels,
                                      kernel_size=self.module.kernel_size,
                                      stride=self.module.stride,
                                      padding=self.module.padding,
                                      dilation=self.module.dilation,
                                      bias=False)

                self.transposed_conv = nn.ConvTranspose2d(in_channels=self.module.out_channels,
                                                          out_channels=self.module.in_channels,
                                                          kernel_size=self.module.kernel_size,
                                                          stride=self.module.stride,
                                                          padding=self.module.padding,
                                                          dilation=self.module.dilation, bias=False)

                # print('conv weight', self.conv.weight.data.shape)
                self.conv.weight.data = w.data
                # print('transposed conv_weight', self.transposed_conv.weight.data.shape)
                self.transposed_conv.weight.data = w.data
                output_channels = w.shape[0]

                self.u = l2normed(w.data.new(1, output_channels, self.height, self.width).normal_(0, 1))

                for _ in range(self.power_iterations):
                    v = l2normed(self.transposed_conv(self.u))
                    conv_res = self.conv(v)
                    self.u = l2normed(conv_res)

                sigma = torch.norm(conv_res, 2)
        else:
            for _ in range(self.power_iterations):
                v = l2normed(torch.mv(torch.t(w.view(height,-1).data), self.u))
                self.u = l2normed(torch.mv(w.view(height, -1).data, v))
            sigma = torch.dot(self.u, torch.mv(w.view(height, -1).data, v))
            # print('sigma form FC', sigma)
        # setattr(self.module, self.name + "_u", u)
        self.saves_sigma = sigma.item()

        self.module.weight.data = self.module.weight.data / (sigma.item())

    def _make_params(self):
        w = getattr(self.module, self.name)

        if len(w.shape) == 4:
            self.conv_f = True
            in_channels, out_channels = w.shape[:2]

            self.u = l2normed(w.data.new(1, out_channels, self.height, self.width).normal_(0, 1))
        else:
            self.conv_f = False
            height = w.data.shape[0]

            self.u = l2normed(w.data.new(height).normal_(0, 1))

        self.u_inited = True

    def forward(self, tensor):
        # to get w
        w = getattr(self.module, self.name)

        if len(w.shape) == 4:
            batch_size, channels, self.height, self.width = tensor.shape

        self._update_u_v()

        return self.module.forward(tensor)
