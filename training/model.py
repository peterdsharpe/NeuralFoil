"""
The PyTorch training-side wrapper around NeuralFoil's model core.

The model mathematics (MLP forward, alpha-flip symmetry fusion, Mahalanobis
confidence correction) live in neuralfoil/_core.py and are shared verbatim
with inference; this module only registers the parameters with torch and
binds the core to the torch backend. If you change the model, change the
core - both training and inference will follow.
"""

import torch

from neuralfoil import _backend_torch, _core, _spec


class Net(torch.nn.Module):
    def __init__(
        self,
        mean_inputs_scaled: torch.Tensor,
        cov_inputs_scaled: torch.Tensor,
        n_hidden_layers: int,
        width: int,
    ):
        """
        Args:
            mean_inputs_scaled: Mean of the training data in the input latent
                space, shape (N_INPUTS,).
            cov_inputs_scaled: Covariance of the training data in the input
                latent space, shape (N_INPUTS, N_INPUTS).
            n_hidden_layers: Number of width-to-width hidden blocks. The total
                number of hidden neuron layers is n_hidden_layers + 1 (due to
                the input-to-width layer).
            width: Neurons per hidden layer.
        """
        super().__init__()

        layers = [
            torch.nn.Linear(_spec.N_INPUTS, width),
            torch.nn.SiLU(),
        ]
        for _ in range(n_hidden_layers):
            layers += [
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
            ]
        layers += [
            torch.nn.Linear(width, _spec.N_OUTPUTS),
        ]
        self.net = torch.nn.Sequential(*layers)

        # Registered as non-persistent buffers: they move with .to(device),
        # but stay out of the state_dict, so checkpoints contain only the
        # "net.*" parameter keys (and older checkpoints load cleanly).
        self.register_buffer(
            "mean_inputs_scaled", mean_inputs_scaled, persistent=False
        )
        self.register_buffer(
            "inv_cov_inputs_scaled", torch.inverse(cov_inputs_scaled), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _core.evaluate_fused(
            x,
            layer_weights_and_biases=[
                (module.weight, module.bias)
                for module in self.net
                if isinstance(module, torch.nn.Linear)
            ],
            input_distribution={
                "mean_inputs_scaled": self.mean_inputs_scaled,
                "inv_cov_inputs_scaled": self.inv_cov_inputs_scaled,
            },
            backend=_backend_torch,
        )
