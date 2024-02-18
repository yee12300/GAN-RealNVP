import torch
import torch.nn as nn
import numpy as np

class AffineCoupling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(in_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim // 2),
            nn.Tanh()
        )
        self.translation = nn.Sequential(
            nn.Linear(in_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim // 2)
        )

    def forward(self, x, mode, reverse):
        if not reverse:
            x1, x2 = x.chunk(2, dim=1)
            if mode == 0:
                s = self.scale(x1)
                t = self.translation(x1)
                y1 = x1
                y2 = x2 * torch.exp(s) + t

            elif mode == 1:
                s = self.scale(x2)
                t = self.translation(x2)
                y1 = x1 * torch.exp(s) + t
                y2 = x2

            else:
                raise Exception("mode에는 1 또는 2만 들어갈 수 있습니다")

            log_det = s.sum()
            return torch.cat((y1, y2), dim=1), log_det

        else:
            y1, y2 = x.chunk(2, dim=1)
            if mode == 0:
                s = self.scale(y1)
                t = self.translation(y1)
                x1 = y1
                x2 = (y2 - t) * torch.exp(-s)

            elif mode == 1:
                s = self.scale(y2)
                t = self.translation(y2)
                x1 = (y1 - t) * torch.exp(-s)
                x2 = y2

            else:
                raise Exception("mode에는 1 또는 2만 들어갈 수 있습니다")

            log_det = s.sum()
            return torch.cat((x1, x2), dim=1), log_det

class RealNVP(nn.Module):
    def __init__(self, num_block, in_dim, hidden_dim):
        super().__init__()
        self.coupling_layers = nn.ModuleList([
            AffineCoupling(in_dim, hidden_dim) for _ in range(num_block)
        ])
        self.mask_mode = [i % 2 for i in range(num_block)]

    def forward(self, x, reverse=False):
        if not reverse:
            log_det_sum = 0
            for i, layer in enumerate(self.coupling_layers):
                x, log_det = layer(x, self.mask_mode[i], reverse)
                log_det_sum += log_det
            return x, log_det_sum
        else:
            mask_mode_reverse = reversed(self.mask_mode)
            for i, layer in enumerate(reversed(self.coupling_layers)):
                x, _ = layer(x, mask_mode_reverse[i], reverse)
            return x, 0


def RealNVPLoss(z, log_det_sum):
    prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
    prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
               - np.log(256) * np.prod(z.size()[1:])
    ll = prior_ll + log_det_sum
    nll = -ll.mean()

    return nll





