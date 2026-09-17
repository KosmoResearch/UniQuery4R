"""MLP head with per-output activations (depth, confidence, warp3d, ...)."""

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(
        self,
        names,
        acts=None,
        in_chan=1024,
        hidden_dim=256,
        elu=False,
        num_hidden_layers=2,
    ):
        super(Head, self).__init__()

        self.names = names
        self.acts = acts
        self.in_chan = in_chan
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = sum(self.names.values())

        if self.num_hidden_layers < 1:
            raise ValueError(
                f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}"
            )

        layers = []
        in_dim = self.in_chan
        for _ in range(self.num_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        if elu:
            layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, meta_data=None):
        x = self.mlp(x)

        results = dict()
        for name, dim in self.names.items():
            cur_x = x[..., :dim]
            cur_x = self._apply_activation_single(cur_x, self.acts.get(name, "") if self.acts is not None else "")
            results[name] = cur_x

            x = x[..., dim:]

        return results

    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """Apply activation to a single output branch.

        Supports: exp / log / expp1 / expm1 / relu / sigmoid / softplus / tanh /
        inv_log / norm / linear.
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "log":
            return torch.log(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        if act == "inv_log":
            return torch.sign(x) * (torch.expm1(torch.abs(x)))
        # Unit L2 on the last dim (e.g. 3-D direction).
        if act == "norm":
            eps = 1e-4
            denom = (x.square().sum(dim=-1, keepdim=True) + eps * eps).sqrt()
            return x / denom
        # Default linear
        return x
