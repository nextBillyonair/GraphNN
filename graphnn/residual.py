import torch
from torch.nn import Module, Linear, Bilinear

# Lernable Residual
class AdditiveResidual(Module):

    def __init__(self, input_size, hidden_size=128, bias=True):
        super().__init__()
        self.wa = Linear(2 * input_size, hidden_size, bias=bias)
        self.va = Linear(hidden_size, 1, bias=bias)

    def forward(self, old, new, return_weights=True):
        # old == new == (B, *, S, E)
        combine = torch.cat((old, new), dim=-1)
        weights = torch.sigmoid(self.va(torch.tanh(self.wa(combine))))
        attended = weights * old + (1 - weights) * new
        # out == (B, *, S, E), weight == (B, *, S, 2)
        if return_weights:
            return attended, torch.cat((weights, (1 - weights)), dim=-1)
        return attended



class GeneralResidual(Module):

    def __init__(self, input_size, hidden_size=128, bias=True):
        super().__init__()
        self.wa = Bilinear(input_size, input_size, 1, bias=bias)

    def forward(self, old, new, return_weights=True):
        # old == new == (B, *, S, E)
        weights = torch.sigmoid(self.wa(old, new))
        attended = weights * old + (1 - weights) * new
        # out == (B, *, S, E), weight == (B, *, S, 2)
        if return_weights:
            return attended, torch.cat((weights, (1 - weights)), dim=-1)
        return attended


class DotResidual(Module):

    def __init__(self, input_size, hidden_size=128, bias=True):
        super().__init__()

    def forward(self, old, new, return_weights=True):
        # old == new == (B, *, S, E)
        bz, vert = old.size()[:2]
        o = old.view(bz*vert, -1).unsqueeze(-2)
        n = new.view(bz*vert, -1).unsqueeze(-1)
        weights = torch.sigmoid(torch.bmm(o, n).view(bz, vert, -1))
        attended = weights * old + (1 - weights) * new
        # out == (B, *, S, E), weight == (B, *, S, 2)
        if return_weights:
            return attended, torch.cat((weights, (1 - weights)), dim=-1)
        return attended
