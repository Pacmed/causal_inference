import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Calculates Empirical maximum mean discrepancy (MMD).

     The lower the result the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

def mmd_loss(x, kernel):
    x_control = x[x[:, 0] == 0, 1:]
    x_treated = x[x[:, 0] == 1, 1:]

    nb_control = x_control.shape[0]
    nb_treated = x_treated.shape[0]

    if nb_control == 0:
        print("No control to compare...")
        return torch.tensor(MMD(x_treated, torch.rand(x_treated.shape[0], x_treated.shape[1]), kernel), requires_grad=True)

    if nb_treated == 0:
        print("No treated to compare...")
        return torch.tensor(MMD(x_control, torch.rand(x_control.shape[0], x_control.shape[1]), kernel), requires_grad=True)

    if nb_control < nb_treated:
        rand_rows = torch.randperm(nb_treated)[:nb_control]
        x_treated = x_treated[rand_rows, :]
    if nb_control > nb_treated:
        rand_rows = torch.randperm(nb_control)[:nb_treated]
        x_control = x_control[rand_rows, :]

    distance = torch.tensor(MMD(x_treated, x_control, kernel), requires_grad=True)

    return distance


class MMDLoss(nn.Module):
    def __init__(self, kernel = 'multiscale'):
        super().__init__()
        self.kernel = kernel
    def forward(self, inputs, targets = 0):
        return mmd_loss(inputs, kernel=self.kernel)

