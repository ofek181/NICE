"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np


class AdditiveCoupling(nn.Module):
    """Additive coupling layer.
    """

    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()

        # input size is half of the in_out dimension.
        input_size = in_out_dim // 2

        # create a Sequential linear block with ReLu activation.
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, mid_dim),
            nn.ReLU())

        # create a ModuleList of Sequential blocks with ReLu activations.
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for i in range(hidden - 1)])

        # output size is half the in_out dimension.
        out_size = in_out_dim // 2

        # create a linear output block
        self.output_layer = nn.Linear(mid_dim, out_size)

        # mask-config determines units to transform on.
        self.mask = mask_config

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # reshape x to match NICE shapes.
        size0, size1 = x.size()

        x = x.reshape((x.shape[0], x.shape[1] // 2, 2))

        # determine units to transform on using mask-config.
        x1, x2 = x[:, :, 1], x[:, :, 0]
        if self.mask:
            x1, x2 = x[:, :, 0], x[:, :, 1]

        # transform half of the data using the predefines model
        out = self.input_layer(x2)
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
        out = self.output_layer(out)

        # apply additive function
        if reverse:
            x1 = x1 - out
        else:
            x1 = x1 + out

        # return x with proper stack
        x = torch.stack((x2, x1), dim=2)
        if self.mask:
            x = torch.stack((x1, x2), dim=2)

        out_param = x.reshape((size0, size1)), log_det_J
        return out_param


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        # TODO fill in

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # TODO fill in


class Scaling(nn.Module):
    """Log-scaling layer.
    """

    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale)
        log_det_j = torch.sum(self.scale) + self.eps

        # scale the data by the Jacobian
        if reverse:
            x = x * (scale ** -1)
        else:
            x = x * scale

        return x, log_det_j


class NICE(nn.Module):
    """NICE main model.
    """

    def __init__(self, prior, coupling, coupling_type, in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        # choose device
        self.device = device

        # choose type of distribution
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))

        elif prior == 'logistic':
            self.prior = TransformedDistribution(Uniform(0, 1),
                                                 [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

        else:
            raise ValueError('Prior not implemented.')

        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.coupling = coupling
        self.hidden = hidden
        self.coupling_type = coupling_type
        self.scaling = Scaling(self.in_out_dim)

        # choose coupling type
        if self.coupling_type == "additive":
            self.coupling = nn.ModuleList([
                AdditiveCoupling(in_out_dim=self.in_out_dim,
                                 mid_dim=self.mid_dim,
                                 hidden=self.hidden,
                                 mask_config=i % 2) for i in range(coupling)])

        elif self.coupling_type == 'adaptive':
            self.coupling = nn.ModuleList([
                AffineCoupling(in_out_dim=self.in_out_dim,
                               mid_dim=self.mid_dim,
                               hidden=self.hidden,
                               mask_config=i % 2) for i in range(coupling)])

        else:
            raise ValueError('Coupling type not implemented.')

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        # reverse scaling
        x, log_det_j = self.scaling(z, reverse=True)
        # reverse coupling layers
        for i in reversed(range(len(self.coupling))):
            x, log_det_j = self.coupling[i](x, 0, reverse=True)
        # return reversed value -> x
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        # initiate log det of Jacobian
        log_det_j = 0
        # pipe into the coupling layers
        for i in range(len(self.coupling)):
            x, log_det_j = self.coupling[i](x, log_det_j)
        # return scaled x and log det of Jacobian
        return self.scaling(x, reverse=False)

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256) * self.in_out_dim  # log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
